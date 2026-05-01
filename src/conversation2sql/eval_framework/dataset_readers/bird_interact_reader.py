import json
from functools import cache
from pathlib import Path

import tqdm

from conversation2sql.eval_framework.agents.bird_baseline.tools.utils_db_execute import _execute_query
from conversation2sql.eval_framework.state import TaskData, ColumnMeaningEntry, ExternalKnowledgeEntry, FollowUpPayload
from conversation2sql.logger import get_logger

logger = get_logger(__name__)


def _get_db_schema_path(dataset_path: Path, db_name: str) -> Path:
    return dataset_path / db_name / f"{db_name}_schema.txt"


def _get_column_meanings_path(dataset_path: Path, db_name: str) -> Path:
    return dataset_path / db_name / f"{db_name}_column_meaning_base.json"


def _get_kb_path(dataset_path: Path, db_name: str) -> Path:
    return dataset_path / db_name / f"{db_name}_kb.jsonl"


@cache
def _get_schema(dataset_path: Path, db_name: str) -> str:
    with open(_get_db_schema_path(dataset_path, db_name)) as f:
        return f.read()


@cache
def _get_column_meanings(dataset_path: Path, db_name: str) -> dict[str, ColumnMeaningEntry]:
    all_column_meanings = {}
    with open(_get_column_meanings_path(dataset_path, db_name)) as f:
        for key, value in json.load(f).items():
            entry = ColumnMeaningEntry(column_meaning=value) \
                if isinstance(value, str) else ColumnMeaningEntry(**value)
            all_column_meanings[key.lower()] = entry
    return all_column_meanings


@cache
def _get_external_knowledge(dataset_path: Path, db_name: str) -> dict[str, ExternalKnowledgeEntry]:
    kb = {}
    with open(_get_kb_path(dataset_path, db_name)) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line.strip())
            if entry['children_knowledge'] == -1:
                entry['children_knowledge'] = []
            kb[entry["knowledge"]] = ExternalKnowledgeEntry(**entry)
    return kb


def _calculate_initial_budget(line: dict, user_patience: int) -> float:
    """a-interact budget in bird-coins (per task, paper Section 3.2).

    Formula: 6 + 2 * m_amb + 2 * patience
      - 6 = ENV_INTERACT(3) + SUBMIT(3) base budget
      - 2 * m_amb = one ask_user (cost=2) per ambiguity point
      - 2 * patience = extra exploration tolerance
      - patience=3 in config = patience_budget=6 in reference (we multiply by 2)
    """
    critical = len(line.get("user_query_ambiguity", {}).get("critical_ambiguity", []))
    knowledge = len(line.get("knowledge_ambiguity", []))
    m_amb = critical + knowledge
    return 6.0 + 2.0 * m_amb + 2.0 * user_patience


def _get_masked_agent_kb(
        knowledge_ambiguity: list,
        full_kb: dict[str, ExternalKnowledgeEntry],
) -> dict[str, ExternalKnowledgeEntry]:
    if not full_kb:
        return {}
    masked_agent_kb = full_kb.copy()
    if not knowledge_ambiguity:
        return masked_agent_kb
    for amb in knowledge_ambiguity:
        dk = amb['deleted_knowledge']
        if dk is not None and dk in masked_agent_kb:
            del masked_agent_kb[dk]
    return masked_agent_kb


def _is_query_empty(sql, db_dsn):
    result, cur = _execute_query(sql, db_dsn)

    return result is None or len(result) == 0 or (
            len(result) == 1 and (
            result[0][cur[0][0]] is None or result[0][cur[0][0]] == "" or result[0][cur[0][0]] == "None")
    )


def load_bird_interact_as_tasks(dataset_path: str,
                                dataset_name_jsonl: str,
                                filter_query_category: bool,
                                db_dsn_template: str,
                                user_patience_budget: int,
                                make_data_ambiguous: bool = True,
                                *args, **kwargs) -> list[TaskData]:
    """
    Note that Bird-Interact contains also follow-up questions but we are only interested in the initial questions for evaluation,
    so we will ignore the follow-up questions.

     https://github.com/bird-bench/BIRD-Interact/blob/48805f00ff427983a57d7137650a8a04b8e5ffad/combine_public_with_gt.py#L65
    """
    dataset_path = Path(dataset_path)
    samples = []
    skipped_instance_id = {'solar_panel_1', 'solar_panel_17', 'virtual_idol_10', 'households_12', 'fake_account_18',
                           'cold_chain_pharma_compliance_14'}

    if not make_data_ambiguous:
        logger.warning("make_data_ambiguous is set to False, the task will be not ambiguous:"
                       " 1. KB will be full not masked; 2. the user query will be not ambiguous")
    skipped_not_query = 0
    skipped_empty = []
    not_ambig_query_mapping = {}
    if 'full' in dataset_name_jsonl:
        not_ambig_query_mapping = _load_not_ambig_query_from_livesqlbench()
    total = 0
    with (open(dataset_name_jsonl, "r", encoding="utf-8") as f):
        for i, raw_line in tqdm.tqdm(enumerate(f, start=1), desc="processing dataset"):
            total += 1
            if not raw_line:
                logger.warning(f"Empty line at index {i}")
                continue

            line: dict = json.loads(raw_line.strip())
            if filter_query_category:
                if line["category"] != "Query":
                    skipped_not_query += 1
                    continue
            db_name = line.pop("selected_database")
            if line['instance_id'] in skipped_instance_id:
                # or _is_query_empty(line['sol_sql'][0], db_dsn_template.format(database=db_name)):
                skipped_empty.append(line['instance_id'])
                continue

            schema = _get_schema(dataset_path, db_name)
            column_meanings = _get_column_meanings(dataset_path, db_name)

            kb_full = _get_external_knowledge(dataset_path, db_name)
            masked_agent_kb = _get_masked_agent_kb(
                line.pop('knowledge_ambiguity'),
                kb_full
            ) if make_data_ambiguous else kb_full

            external_knowledge = line.pop("external_knowledge")

            ambig_question = line.pop("amb_user_query")
            not_ambiguos_query = line.pop("query") if "query" in line else not_ambig_query_mapping.get(
                line['instance_id'])

            sample = TaskData(
                instance_id=line.pop("instance_id"),
                selected_database=db_name,
                task_question=ambig_question if make_data_ambiguous else not_ambiguos_query,
                amb_user_query=ambig_question,
                sol_sql=line.pop('sol_sql'),
                not_ambiguos_query=not_ambiguos_query,
                follow_up=FollowUpPayload(**line.pop("follow_up")),
                task_budget=_calculate_initial_budget(line, user_patience_budget),
                db_dsn=db_dsn_template.format(database=db_name),
                database_engine='postgresql',
                ddl_database_schema=schema,
                full_knowledge_base=kb_full,
                masked_agent_kb=masked_agent_kb,
                gt_knowledge_base={k: v for k, v in kb_full.items() if v.id in external_knowledge},
                column_meanings=column_meanings,
                user_query_ambiguity=line.pop("user_query_ambiguity"),
                preprocess_sql=line.pop("preprocess_sql"),
                test_cases=line.pop('test_cases'),
                clean_up_sqls=line.pop("clean_up_sqls"),
                category=line.pop("category"),
                sql_query_conditions=line.pop("conditions"),
                **line,
            )
            samples.append(sample)

    logger.info(
        f'Skipped {skipped_not_query} sample since filter_query_category is set to {filter_query_category}'
    )
    logger.info(
        f'Skipped sample since the query is empty\n [{skipped_empty}]'
    )
    logger.info(
        f'Finally loaded {len(samples)}/{total} samples from {dataset_name_jsonl}'
    )
    return samples


def _load_not_ambig_query_from_livesqlbench() -> dict:
    path = 'data/livesqlbench-base-full-v1/livesqlbench_data.jsonl'
    output = {}
    with (open(path, "r", encoding="utf-8") as f):
        for i, raw_line in tqdm.tqdm(enumerate(f, start=1), desc="processing Livesqlbench"):
            if not raw_line:
                logger.warning(f"Empty line at index {i}")
                continue

            line: dict = json.loads(raw_line.strip())
            instance_id = line['instance_id']
            not_ambig_query = line['normal_query']
            if instance_id in output:
                raise KeyError('Value is already present')

            output[instance_id] = not_ambig_query
    return output


if __name__ == '__main__':
    _dataset_path = 'data/bird_interact/bird-interact-full'
    _dataset_name_jsonl = 'data/bird_interact/bird-interact-full/bird_interact_data_GT.jsonl'
    _filter_query_category = True
    _db_dsn_template = 'postgresql://root:123123@localhost:5433/{database}'
    _user_patience = 10

    samples = load_bird_interact_as_tasks(
        _dataset_path,
        _dataset_name_jsonl,
        _filter_query_category,
        _db_dsn_template,
        _user_patience,
    )
