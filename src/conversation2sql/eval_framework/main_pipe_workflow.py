import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Callable

import tqdm
import tqdm.asyncio
import yaml

from conversation2sql.config_input import (
    ConfigReader,
    ConfigPredictor,
    ConfigUserSimulator,
    ConfigPipeline,
)
from conversation2sql.eval_framework.agents import (
    run_agent_bird_baseline,
    run_agent_deep_agent,
    run_baseline_no_tool,
)
from conversation2sql.eval_framework.agents.utils import utils_create_model
from conversation2sql.eval_framework.dataset_readers import load_bird_interact_as_tasks
from conversation2sql.eval_framework.state import TaskData
from conversation2sql.logger import get_logger

logger = get_logger(__name__)



def _resolve_baseline_settings(baseline: str) -> tuple[bool, Callable, bool]:
    """Map ConfigPipeline.baseline → (make_data_ambiguous, runner, needs_user_sim).

    no_tool    -> clean query, no agent loop, no user-sim
    tools_only -> clean query, agent without ask_user, no user-sim
    tools_user -> clean query, agent with ask_user, user-sim required
    bird_full  -> ambiguous query, agent with ask_user, user-sim required
    """
    table = {
        "no_tool": (False, run_baseline_no_tool, False),
        "tools_only": (False, run_agent_bird_baseline, False),
        "tools_user": (False, run_agent_bird_baseline, True),
        "bird_full": (True, run_agent_bird_baseline, True),
        # deep_agent parallels bird_full (ambiguous query + ask_user) but swaps
        # the schema tools for the deepagents virtual filesystem.
        "deep_agent": (True, run_agent_deep_agent, True),
    }
    if baseline not in table:
        raise ValueError(
            f"Unknown baseline: {baseline!r}; expected one of {list(table)}"
        )
    return table[baseline]


def _resolve_iterations(num_iterations: int, predictor_temperature: float) -> int:
    """Collapse to a single iteration when the predictor is deterministic.

    With temperature <= 0 the agent's output is deterministic, so repeating the
    dataset adds no information — we run it once regardless of num_iterations.
    """
    if predictor_temperature <= 0 and num_iterations > 1:
        logger.warning(
            "predictor temperature=%s; iterations are deterministic — "
            "collapsing num_iterations=%s to 1",
            predictor_temperature,
            num_iterations,
        )
        return 1
    return num_iterations


def workflow_evaluation_pipeline(
    config_pipeline: ConfigPipeline,
    config_reader: ConfigReader,
    config_predictor: ConfigPredictor,
    config_user: ConfigUserSimulator,
) -> None:
    logger.info(f"config_pipeline: {config_pipeline}")
    logger.info(f"config_reader: {config_reader}")
    logger.info(f"config_predictor: {config_predictor}")
    logger.info(f"config_user: {config_user}")

    # Resolve baseline settings and override make_data_ambiguous
    forced_amb, runner, needs_user_sim = _resolve_baseline_settings(
        config_pipeline.baseline
    )
    if config_reader.make_data_ambiguous != forced_amb:
        logger.warning(
            "baseline=%s forces make_data_ambiguous=%s; overriding ConfigReader.make_data_ambiguous=%s",
            config_pipeline.baseline,
            forced_amb,
            config_reader.make_data_ambiguous,
        )
    config_reader = config_reader.model_copy(update={"make_data_ambiguous": forced_amb})

    output_folder = Path(config_pipeline.output_folder)

    # save config in the output folder; on resume keep the original snapshot intact
    snapshot_name = (
        f"config_resume_{datetime.now().strftime('%H_%M_%S')}.yaml"
        if config_pipeline.resume
        else "config.yaml"
    )
    _save_configs_as_yaml(
        output_folder=output_folder,
        config_pipeline=config_pipeline,
        config_reader=config_reader,
        config_predictor=config_predictor,
        config_user=config_user,
        filename=snapshot_name,
    )

    # Register this run in the experiments index (best-effort: never abort a run).
    # The repo root must be importable for the top-level `explorer` package, which
    # is not part of the installed `conversation2sql` distribution.
    try:
        import sys

        repo_root = str(Path(__file__).resolve().parents[3])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from explorer.index import append_stub

        append_stub(output_folder)
    except Exception as e:  # noqa: BLE001 - indexing must never break evaluation
        logger.warning("Could not append run to experiments index: %s", e)

    # initialize models (API based)
    model_agent, (model_user_parsing, model_user_generator) = _init_models(
        config_predictor,
        config_user,
        needs_user_sim=needs_user_sim,
        concurrency=config_pipeline.concurrency,
    )

    effective_iterations = _resolve_iterations(
        config_pipeline.num_iterations, config_predictor.temperature
    )

    # read dataset
    dataset: list[TaskData] = load_bird_interact_as_tasks(**config_reader.model_dump())

    if config_pipeline.debug:
        dataset = dataset[:10]
        logger.info("Debug mode is ON - using only the first 10 tasks from the dataset")

    try:
        asyncio.run(
            _run_tasks_concurrently(
                dataset=dataset,
                runner=runner,
                model_agent=model_agent,
                model_user_parsing=model_user_parsing,
                model_user_generator=model_user_generator,
                baseline=config_pipeline.baseline,
                config_predictor=config_predictor,
                config_user=config_user,
                config_pipeline=config_pipeline,
                config_reader=config_reader,
                output_folder=output_folder,
                concurrency=config_pipeline.concurrency,
                num_iterations=effective_iterations,
                resume=config_pipeline.resume,
            )
        )
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        response_error = {"error": str(e)}
        output = output_folder / "results_error.jsonl"
        _save_record(response=response_error, output_path_jsonl=output)
        logger.info(f"Saved ERROR to {output}")
        raise e


def _load_completed_pairs(
    output_folder: Path, num_iterations: int
) -> set[tuple[str, int]]:
    """Successful (instance_id, iteration) pairs already on disk.

    Reads output_folder/results_iter{i}.jsonl for i in range(num_iterations).
    Parsing is line-by-line and defensive: blank lines, malformed JSON (a crash
    can leave a truncated trailing line), and records missing instance_id are
    skipped. Error records live only in results_error.jsonl and are deliberately
    NOT counted as completed, so errored/never-reached tasks are re-run.
    """
    completed: set[tuple[str, int]] = set()
    for iteration in range(num_iterations):
        path = output_folder / f"results_iter{iteration}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                instance_id = record.get("instance_id")
                if instance_id is not None:
                    completed.add((instance_id, iteration))
    return completed


async def _run_tasks_concurrently(
    dataset: list[TaskData],
    runner: Callable,
    model_agent,
    model_user_parsing,
    model_user_generator,
    baseline: str,
    config_predictor: ConfigPredictor,
    config_user: ConfigUserSimulator,
    config_pipeline: ConfigPipeline,
    config_reader: ConfigReader,
    output_folder: Path,
    concurrency: int,
    num_iterations: int,
    resume: bool = False,
) -> None:
    # Each task runs the whole synchronous agent loop via `asyncio.to_thread`,
    # which dispatches to the running loop's default executor. That default pool
    # caps at min(32, cpu_count()+4) threads, so it silently throttles concurrency
    # to ~32 regardless of the `concurrency` setting. Size the executor to the
    # requested concurrency so the semaphore is the only real limit (these threads
    # are I/O-bound on the LLM server + Postgres, so they're cheap to oversubscribe).
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=concurrency))

    sem = asyncio.Semaphore(concurrency)
    file_lock = threading.Lock()

    async def _process_one(task: TaskData, iteration: int) -> None:
        try:
            async with sem:
                if baseline == "no_tool":
                    response = await asyncio.to_thread(runner, task, model_agent)
                else:
                    response = await asyncio.to_thread(
                        runner,
                        task,
                        model_agent,
                        model_user_parsing,
                        model_user_generator,
                        enable_ask_user=(baseline in ("tools_user", "bird_full", "deep_agent")),
                    )
        except Exception as e:
            # Isolate per-task failures: one wedged/erroring task is logged to
            # results_error.jsonl and skipped, instead of propagating out of the
            # gather and aborting every remaining (task, iteration) pair.
            import traceback
            logger.error(
                f"Error on task_id={task.instance_id} iteration={iteration}: {e}\n{traceback.format_exc()}"
            )
            error_record = {
                "instance_id": task.instance_id,
                "iteration": iteration,
                "error": str(e),
            }
            with file_lock:
                _save_record(
                    response=error_record,
                    output_path_jsonl=output_folder / "results_error.jsonl",
                )
            return
        task_output = {
            "config_predictor": config_predictor.model_dump(),
            "config_user": config_user.model_dump(),
            "config_pipeline": config_pipeline.model_dump(),
            "config_reader": config_reader.model_dump(),
            **task.model_dump(),
            **response,
            "iteration": iteration,
        }
        file_result = output_folder / f"results_iter{iteration}.jsonl"
        with file_lock:
            _save_record(response=task_output, output_path_jsonl=file_result)
        logger.info(
            f"Saved response for task_id={task.instance_id} iteration={iteration} to {file_result}"
        )

    # Flatten all (iteration, task) pairs into one shared semaphore-bounded gather
    # so the concurrency pool stays saturated across iteration boundaries.
    # Per-task errors are swallowed inside _process_one (logged to
    # results_error.jsonl), so a single failure never aborts the gather.
    completed: set[tuple[str, int]] = (
        _load_completed_pairs(output_folder, num_iterations) if resume else set()
    )
    pairs = [
        (task, iteration)
        for iteration in range(num_iterations)
        for task in dataset
        if (task.instance_id, iteration) not in completed
    ]
    if resume:
        total = num_iterations * len(dataset)
        logger.info(
            "resume: skipping %d completed, running %d/%d (instance_id, iteration) pairs",
            total - len(pairs),
            len(pairs),
            total,
        )
    coros = [_process_one(task, iteration) for task, iteration in pairs]
    await tqdm.asyncio.tqdm.gather(
        *coros, desc=f"Inference with {baseline} x{num_iterations}"
    )


def _build_vllm_http_client(concurrency: int):
    """A LiteLLM sync HTTP client whose connection pool is sized to `concurrency`.

    LiteLLM's hosted_vllm path otherwise reuses a cached httpx client with the
    httpx default pool (max_connections=100), which caps in-flight requests at
    ~100 even when more tasks are running. Sizing the pool to the concurrency lets
    every worker hold a connection so the vLLM-side `max_num_seqs` is the only
    limit. Returns None on import failure so model creation still works.
    """
    try:
        import httpx
        from litellm.llms.custom_httpx.http_handler import HTTPHandler
    except Exception as e:  # noqa: BLE001 - never let pooling break model init
        logger.warning("Could not build sized vLLM HTTP client: %s", e)
        return None
    limits = httpx.Limits(
        max_connections=concurrency, max_keepalive_connections=concurrency
    )
    return HTTPHandler(client=httpx.Client(limits=limits))


def _init_models(
    config_predictor: ConfigPredictor,
    config_user: ConfigUserSimulator,
    needs_user_sim: bool,
    concurrency: int = 1,
) -> tuple:
    http_client = _build_vllm_http_client(concurrency)
    model_agent = utils_create_model(
        model_name=config_predictor.model_name,
        model_provider=config_predictor.model_provider,
        temperature=config_predictor.temperature,
        max_tokens=config_predictor.max_new_tokens,
        top_p=config_predictor.top_p,
        top_k=config_predictor.top_k,
        api_base=config_predictor.predictor_vllm_api_base,
        min_p=config_predictor.min_p,
        presence_penalty=config_predictor.presence_penalty,
        repetition_penalty=config_predictor.repetition_penalty,
        enable_thinking=config_predictor.enable_thinking,
        reasoning_effort=config_predictor.reasoning_effort,
        request_timeout=config_predictor.request_timeout,
        num_retries=config_predictor.num_retries,
        http_client=http_client,
    )
    if not needs_user_sim:
        return model_agent, (None, None)

    model_user_parsing = utils_create_model(
        model_name=config_user.model_name,
        model_provider=config_user.model_provider,
        temperature=config_user.temperature,
        max_tokens=config_user.max_new_tokens,
        api_base=config_user.user_simulator_vllm_api_base,
        request_timeout=config_user.request_timeout,
        num_retries=config_user.num_retries,
        http_client=http_client,
    )
    model_user_generator = utils_create_model(
        model_name=config_user.model_name,
        model_provider=config_user.model_provider,
        temperature=config_user.temperature,
        max_tokens=config_user.max_new_tokens,
        request_timeout=config_user.request_timeout,
        num_retries=config_user.num_retries,
    )
    return model_agent, (model_user_parsing, model_user_generator)


def _save_configs_as_yaml(
    output_folder: Path,
    config_pipeline: ConfigPipeline,
    config_reader: ConfigReader,
    config_predictor: ConfigPredictor,
    config_user: ConfigUserSimulator,
    filename: str = "config.yaml",
) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)
    if config_reader.make_data_ambiguous:
        task_budget_formula = (
            "task_budget = 6 + 2*m_amb + 2*user_patience_budget, where "
            "m_amb = len(critical_ambiguity) + len(knowledge_ambiguity) per task"
        )
    else:
        task_budget_formula = (
            "task_budget = 6 + 2*user_patience_budget "
            "(ambiguity not counted because make_data_ambiguous=False)"
        )
    configs = {
        "task_budget_formula": task_budget_formula,
        "pipeline": config_pipeline.model_dump(mode="json"),
        "reader": config_reader.model_dump(mode="json"),
        "predictor": config_predictor.model_dump(mode="json"),
        "user_simulator": config_user.model_dump(mode="json"),
    }
    config_path = output_folder / filename
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(configs, f, sort_keys=False, allow_unicode=True)
    logger.info(f"Saved configs to {config_path}")


def _save_record(response: dict, output_path_jsonl: Path):
    with output_path_jsonl.open(
        "a", encoding="utf-8"
    ) as f:  # "a" = append line by line
        f.write(json.dumps(response, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    _config_pipeline = ConfigPipeline()
    _config_reader = ConfigReader()
    _config_predictor = ConfigPredictor()
    _config_user = ConfigUserSimulator()
    workflow_evaluation_pipeline(
        config_pipeline=_config_pipeline,
        config_reader=_config_reader,
        config_predictor=_config_predictor,
        config_user=_config_user,
    )
