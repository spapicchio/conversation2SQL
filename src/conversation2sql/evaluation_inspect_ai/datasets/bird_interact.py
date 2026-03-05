import copy
import json
from typing import Any

import inspect_ai.dataset as inspect_ai_dataset
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from conversation2sql.logger import get_logger

logger = get_logger(__name__)


class BirdInteractMetadata(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    selected_database: str = Field(description="Name of the selected database")
    query: str = Field(
        description="The unambiguous user query (comes from query field in LiveSQLBench-Base-Lite)."
    )
    user_query_ambiguity: dict = Field(
        description="Ambiguity annotations for the user query, as a dictionary mapping 'term' to 'sql snippet'. Differentiate between 'non_critical_ambiguity' and 'critical_ambiguity'."
    )
    knowledge_ambiguity: list[dict] = Field(
        description="Ambiguity annotations for the knowledge, as a dictionary mapping 'term' to 'sql snippet'. Differentiate between 'non_critical_ambiguity' and 'critical_ambiguity'."
    )
    preprocess_sql: list[str] = Field(
        description="SQL queries to run before executing the solution or prediction."
    )
    test_cases: list[str] = Field(
        description="A set of test cases to validate the predicted corrected SQL."
    )
    external_knowledge: list = Field(
        description="The external knowledge related to the specific task."
    )


def _read_gt_bird_interact(gt_jsonl_path):
    gt_data: dict[str, dict] = {}
    logger.info(f"Reading GT data from `{gt_jsonl_path}`")
    with open(gt_jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                gt_entry = json.loads(line)
                instance_id = gt_entry.get("instance_id")
                if instance_id:
                    gt_data[instance_id] = gt_entry
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing GT line: {e}")
                continue

    logger.info(f"Loaded GT data for {len(gt_data)} instances.")
    return gt_data


def read_bird_interact(
        hf_dataset_path: str = "birdsql/bird-interact-lite",
        gt_jsonl_path: str = "data/bird_interact/bird_interact_mini_gt_kg_testcases_1008.jsonl",
        *args: Any,
        **kwargs: Any,
) -> inspect_ai_dataset.Dataset:
    # https://huggingface.co/datasets/birdsql/bird-interact-lite
    gt_data: dict[str, dict] = _read_gt_bird_interact(gt_jsonl_path)
    logger.info(f"Reading GT data from `{gt_jsonl_path}`")
    if "lite" in hf_dataset_path and "lite" not in gt_jsonl_path:
        raise ValueError(
            "GT JSONL path must contain 'lite' in the path when using a 'lite' dataset."
        )

    if "lite" not in hf_dataset_path and "lite" in gt_jsonl_path:
        raise ValueError(
            "GT JSONL path must not contain 'lite' in the path when using a non-'lite' dataset."
        )

    logger.info(f"Loaded GT data for {len(gt_data)} instances.")

    def record_to_sample(record) -> inspect_ai_dataset.Sample:
        record = copy.deepcopy(record)
        gt_entry = gt_data.get(record["instance_id"], {})

        record["sol_sql"] = gt_entry.get("sol_sql", record["sol_sql"])
        record["external_knowledge"] = gt_entry.get(
            "external_knowledge", record["external_knowledge"]
        )
        record["test_cases"] = gt_entry.get("test_cases", record["test_cases"])

        # Merge follow_up GT fields if present
        if "follow_up" in gt_entry and isinstance(gt_entry["follow_up"], dict):
            if "follow_up" not in record:
                record["follow_up"] = {}
            record["follow_up"]["sol_sql"] = gt_entry["follow_up"].get("sol_sql", [])
            record["follow_up"]["external_knowledge"] = gt_entry["follow_up"].get(
                "external_knowledge", []
            )
            record["follow_up"]["test_cases"] = gt_entry["follow_up"].get(
                "test_cases", []
            )

        return inspect_ai_dataset.Sample(
            input=record["amb_user_query"],
            target=record["sol_sql"],  # pyrefly: ignore
            id=record["instance_id"],
            metadata=BirdInteractMetadata(**dict(record.items())).model_dump(),
        )

    return inspect_ai_dataset.hf_dataset(
        hf_dataset_path,
        split="dev",
        sample_fields=record_to_sample,
    )


if __name__ == "__main__":
    dataset = read_bird_interact(
        hf_dataset_path="birdsql/bird-interact-lite",
        gt_jsonl_path="data/bird_interact/bird_interact_lite_gt_kg_testcases_1008.jsonl",
    )
    for sample in dataset:
        print(sample)
        break
