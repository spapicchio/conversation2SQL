from conversation2sql.eval_framework.dataset_readers.utils_parse_schema import (
    _extract_tables_from_toon_format,
)
from conversation2sql.eval_framework.dataset_readers.utils_parse_schema import (
    _extract_tables_from_ddl_format,
)
import copy
import json
from functools import cache
from pathlib import Path

import tqdm

from conversation2sql.eval_framework.agents.bird_baseline.tools.utils_db_execute import (
    _execute_query,
)
from conversation2sql.eval_framework.dataset_readers.sql_usage_extractor import (
    extract_table_in_gt_sql,
)
from conversation2sql.eval_framework.state import (
    TaskData,
    ColumnMeaningEntry,
    ExternalKnowledgeEntry,
    FollowUpPayload,
)
from conversation2sql.logger import get_logger

logger = get_logger(__name__)


def _get_db_schema_path(
    dataset_path: Path, db_name: str, database_schema_type: str
) -> Path:
    return dataset_path / db_name / f"{db_name}_{database_schema_type}.txt"


def _get_column_meanings_path(dataset_path: Path, db_name: str) -> Path:
    return dataset_path / db_name / f"{db_name}_column_meaning_base.json"


def _get_kb_path(dataset_path: Path, db_name: str) -> Path:
    return dataset_path / db_name / f"{db_name}_kb.jsonl"


@cache
def _get_schema(dataset_path: Path, db_name: str, database_schema_type: str) -> str:
    with open(_get_db_schema_path(dataset_path, db_name, database_schema_type)) as f:
        return f.read()


@cache
def _get_column_meanings(
    dataset_path: Path, db_name: str
) -> dict[str, ColumnMeaningEntry]:
    all_column_meanings = {}
    with open(_get_column_meanings_path(dataset_path, db_name)) as f:
        for key, value in json.load(f).items():
            entry = (
                ColumnMeaningEntry(column_meaning=value)
                if isinstance(value, str)
                else ColumnMeaningEntry(**value)
            )
            all_column_meanings[key.lower()] = entry
    return all_column_meanings


@cache
def _get_external_knowledge(
    dataset_path: Path, db_name: str
) -> dict[str, ExternalKnowledgeEntry]:
    kb = {}
    with open(_get_kb_path(dataset_path, db_name)) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line.strip())
            if entry["children_knowledge"] == -1:
                entry["children_knowledge"] = []
            kb[entry["knowledge"]] = ExternalKnowledgeEntry(**entry)
    return kb


def _build_table_to_columns(
    column_meanings: dict[str, ColumnMeaningEntry],
    db_name: str,
) -> dict[str, set[str]]:
    table_to_columns: dict[str, set[str]] = {}
    prefix = f"{db_name.lower()}|"
    for key in column_meanings.keys():
        if not key.startswith(prefix):
            continue
        parts = key.split("|")
        if len(parts) != 3:
            continue
        _, table, column = parts
        table_to_columns.setdefault(table, set()).add(column)
    return table_to_columns


def _extract_gt_sql_table_usage(
    sol_sqls: list[str] | str,
    table_to_columns: dict[str, set[str]],
    ddl_schema: str | None,
) -> tuple[dict[str, list[str]], str | None]:
    sql_list = [sol_sqls] if isinstance(sol_sqls, str) else list(sol_sqls)
    combined: dict[str, set[str]] = {}
    errors: list[str] = []

    # remove CREATE TYPE statements from ddl_schema since they can cause parsing error but they are not relevant for table usage extraction
    ddl_schema = (
        "\n".join(
            [
                line
                for line in ddl_schema.splitlines()
                if not line.startswith("CREATE TYPE")
            ]
        )
        if ddl_schema
        else None
    )

    for idx, sql in enumerate(sql_list):
        usage, error = extract_table_in_gt_sql(
            sql,
            table_to_columns,
            ddl_schema=ddl_schema,
        )
        if error:
            errors.append(f"sql_index={idx}: {error}")
            continue
        for table, columns in usage.items():
            combined.setdefault(table, set()).update(columns)

    table_in_gt_sql = {table: sorted(columns) for table, columns in combined.items()}
    parse_error = " | ".join(errors) if errors else None
    return table_in_gt_sql, parse_error


def _calculate_initial_budget(
    line: dict, user_patience: int, count_ambiguity: bool = True
) -> float:
    """Task budget in bird-coins (paper Section 3.2).

    With count_ambiguity=True (default): 6 + 2*m_amb + 2*patience
    With count_ambiguity=False: 6 + 2*patience (no-ambiguity ablations).
    """
    if not count_ambiguity:
        return 6.0 + 2.0 * user_patience
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
    masked_agent_kb = copy.deepcopy(full_kb)
    if not knowledge_ambiguity:
        return masked_agent_kb
    for amb in knowledge_ambiguity:
        dk = amb["deleted_knowledge"]
        if dk is not None and dk in masked_agent_kb:
            del masked_agent_kb[dk]
    return masked_agent_kb


def _resolve_db_context(
    dataset_path: Path,
    db_name: str,
    database_schema_type: str,
    sol_sqls: list[str],
    read_only_gt_tables: bool,
) -> tuple[str, dict[str, ColumnMeaningEntry], dict[str, list[str]], str | None]:
    """Load schema and column metadata, compute GT SQL table usage, optionally filter schema."""
    schema = _get_schema(dataset_path, db_name, database_schema_type)
    column_meanings = _get_column_meanings(dataset_path, db_name)
    table_to_columns = _build_table_to_columns(column_meanings, db_name)

    table_in_gt_sql, parse_error = _extract_gt_sql_table_usage(
        sol_sqls, table_to_columns, schema
    )

    if read_only_gt_tables:
        tables_no_cte = [
            tbl
            for tbl in table_in_gt_sql.keys()
            if not tbl.startswith("cte") and not tbl.startswith("unknown")
        ]
        if database_schema_type == "ddl":
            schema = _extract_tables_from_ddl_format(schema, tables_no_cte)
        else:
            schema = _extract_tables_from_toon_format(schema, tables_no_cte)

    return schema, column_meanings, table_in_gt_sql, parse_error


def _resolve_kb_context(
    dataset_path: Path,
    db_name: str,
    external_knowledge_ids: list,
    knowledge_ambiguity: list,
    make_data_ambiguous: bool,
    read_only_gt_kb: bool,
) -> tuple[
    dict[str, ExternalKnowledgeEntry],
    dict[str, ExternalKnowledgeEntry],
    dict[str, ExternalKnowledgeEntry],
]:
    """Load full KB, derive GT KB subset, and build the masked KB seen by the agent."""
    kb_full = _get_external_knowledge(dataset_path, db_name)
    gt_knowledge_base = {
        k: v for k, v in kb_full.items() if v.id in external_knowledge_ids
    }

    if read_only_gt_kb:
        kb_full = copy.deepcopy(gt_knowledge_base)

    masked_agent_kb = (
        _get_masked_agent_kb(knowledge_ambiguity, kb_full)
        if make_data_ambiguous
        else kb_full
    )

    return kb_full, masked_agent_kb, gt_knowledge_base


def _is_query_empty(sql, db_dsn):
    try:
        result, cur = _execute_query(sql, db_dsn)
    except Exception as e:
        logger.warning(
            f"Error executing SQL `{db_dsn}`, sql: `{sql}` to check if query is empty: {e}"
        )
        return True

    return (
        result is None
        or len(result) == 0
        or (
            len(result) == 1
            and (
                result[0][cur[0][0]] is None
                or result[0][cur[0][0]] == ""
                or result[0][cur[0][0]] == "None"
            )
        )
    )


def load_bird_interact_as_tasks(
    dataset_path: str | Path,
    dataset_name_jsonl: str,
    filter_query_category: bool,
    db_dsn_template: str,
    user_patience_budget: int,
    make_data_ambiguous: bool = True,
    read_only_gt_tables: bool = False,
    read_only_gt_kb: bool = False,
    database_schema_type: str = "ddl",
    is_kb_linearized: bool = False,
    *args,
    **kwargs,
) -> list[TaskData]:
    """
    Note that Bird-Interact contains also follow-up questions but we are only interested in the initial questions for evaluation,
    so we will ignore the follow-up questions.

     https://github.com/bird-bench/BIRD-Interact/blob/48805f00ff427983a57d7137650a8a04b8e5ffad/combine_public_with_gt.py#L65
    """
    dataset_path = Path(dataset_path)
    samples = []
    skipped_instance_id = {
        # DB FULL
        "solar_panel_1",
        "solar_panel_17",
        "virtual_idol_10",
        "households_12",
        "fake_account_18",
        "cold_chain_pharma_compliance_14",
        # DB LITE
        "archeology_7",  # Error in the SQL
        "insider_1",
        "vaccine_2",
        "vaccine_7",
        "vaccine_10",
        "virtual_2",
    }

    if not make_data_ambiguous:
        logger.warning(
            "make_data_ambiguous is set to False, the task will be not ambiguous:"
            " 1. KB will be full not masked; 2. the user query will be not ambiguous"
        )
    if read_only_gt_tables:
        logger.warning(
            "read_only_gt_tables is set to True, the agent will read from GT tables only (schema linking is performed automatically)"
        )
    if read_only_gt_kb:
        logger.warning(
            "read_only_gt_kb is set to True, the agent will read from the GT KB only"
        )

    skipped_not_query = 0
    skipped_empty = []
    not_ambig_query_mapping = {}
    if "full" in dataset_name_jsonl:
        not_ambig_query_mapping = _load_not_ambig_query_from_livesqlbench()

    total = 0
    with open(dataset_name_jsonl, "r", encoding="utf-8") as f:
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
            if line["instance_id"] in skipped_instance_id:
                #  or _is_query_empty(line['sol_sql'][0], db_dsn_template.format(database=db_name)):
                skipped_empty.append(line["instance_id"])
                continue

            ambig_question = line.pop("amb_user_query")
            not_ambiguos_query = (
                line.pop("query")
                if "query" in line
                else not_ambig_query_mapping.get(line["instance_id"])
            )
            sol_sqls = line.pop("sol_sql")

            schema, column_meanings, table_in_gt_sql, table_in_gt_sql_parse_error = (
                _resolve_db_context(
                    dataset_path,
                    db_name,
                    database_schema_type,
                    sol_sqls,
                    read_only_gt_tables,
                )
            )

            kb_full, masked_agent_kb, gt_knowledge_base = _resolve_kb_context(
                dataset_path,
                db_name,
                external_knowledge_ids=line.pop("external_knowledge"),
                knowledge_ambiguity=line.pop("knowledge_ambiguity"),
                make_data_ambiguous=make_data_ambiguous,
                read_only_gt_kb=read_only_gt_kb,
            )

            if table_in_gt_sql_parse_error:
                logger.warning(
                    "table_in_gt_sql parse error for instance_id=%s: %s",
                    line.get("instance_id"),
                    table_in_gt_sql_parse_error,
                )
            sample = TaskData(
                instance_id=line.pop("instance_id"),
                selected_database=db_name,
                task_question=ambig_question  # pyrefly: ignore
                if make_data_ambiguous
                else not_ambiguos_query,
                amb_user_query=ambig_question,
                sol_sql=sol_sqls,
                not_ambiguos_query=not_ambiguos_query,  # pyrefly: ignore
                follow_up=FollowUpPayload(**line.pop("follow_up")),
                task_budget=_calculate_initial_budget(
                    line, user_patience_budget, count_ambiguity=make_data_ambiguous
                ),
                db_dsn=db_dsn_template.format(database=db_name),
                database_engine="postgresql",
                ddl_database_schema=schema,
                full_knowledge_base=kb_full,
                masked_agent_kb=masked_agent_kb,
                gt_knowledge_base=gt_knowledge_base,
                column_meanings=column_meanings,
                user_query_ambiguity=line.pop("user_query_ambiguity"),
                preprocess_sql=line.pop("preprocess_sql"),
                test_cases=line.pop("test_cases"),
                clean_up_sqls=line.pop("clean_up_sqls"),
                category=line.pop("category"),
                sql_query_conditions=line.pop("conditions"),
                table_in_gt_sql=table_in_gt_sql,
                table_in_gt_sql_parse_error=table_in_gt_sql_parse_error,
                is_kb_linearized=is_kb_linearized,
                **line,
            )
            samples.append(sample)

    logger.info(
        f"Skipped {skipped_not_query} sample since filter_query_category is set to {filter_query_category}"
    )
    logger.info(f"Skipped sample since the query is empty\n [{skipped_empty}]")
    logger.info(
        f"Finally loaded {len(samples)}/{total} samples from {dataset_name_jsonl}"
    )
    return samples


def _load_not_ambig_query_from_livesqlbench() -> dict:
    path = "data/livesqlbench-base-full-v1/livesqlbench_data.jsonl"
    output = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, raw_line in tqdm.tqdm(
            enumerate(f, start=1), desc="processing Livesqlbench"
        ):
            if not raw_line:
                logger.warning(f"Empty line at index {i}")
                continue

            line: dict = json.loads(raw_line.strip())
            instance_id = line["instance_id"]
            not_ambig_query = line["normal_query"]
            if instance_id in output:
                raise KeyError("Value is already present")

            output[instance_id] = not_ambig_query
    return output


if __name__ == "__main__":
    _dataset_path = "data/bird_interact/bird-interact-lite"
    _dataset_name_jsonl = (
        "data/bird_interact/bird-interact-lite/bird_interact_data_GT.jsonl"
    )
    _filter_query_category = True
    _db_dsn_template = "postgresql://root:123123@localhost:5432/{database}"
    _user_patience = 10
    # plants, electrical_performance, plant_record, warranty_risk_plants
    samples = load_bird_interact_as_tasks(
        _dataset_path,
        _dataset_name_jsonl,
        _filter_query_category,
        _db_dsn_template,
        _user_patience,
        make_data_ambiguous=True,
        read_only_gt_tables=True,
        read_only_gt_kb=True,
        database_schema_type="toon",
    )

    print(f"Loaded {len(samples)} samples")
    print("Example sample:")
    print(samples[0].table_in_gt_sql)
    print(samples[0].sol_sql[0])
    print(samples[0].masked_agent_kb)
