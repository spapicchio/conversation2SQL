import json
from functools import cache
from pathlib import Path

import datasets  # pyrefly: ignore
import tqdm  # pyrefly: ignore
from datasets import Sequence, Value  # pyrefly: ignore

from conversation2sql.config_input import ConfigReader
from conversation2sql.eval import reader_registry, Sample
from conversation2sql.eval.interfaces import BaseReader, ToolUserContext
from conversation2sql.eval.message_builder import build_agent_messages
from conversation2sql.eval.prompt_params import BirdInteractAgentParams, BirdInteractUserSimulatorParams
from conversation2sql.logger import get_logger
from conversation2sql.prompt_factory import PromptFactory


@reader_registry.register
class BirdInteractReader(BaseReader):
    """
    data: Each data instance contain the following main parts:
        selected_database: The name of the database.
        query: The unambiguous user query (comes from query field in LiveSQLBench-Base-Lite).
        amb_user_query: The user query with injected ambiguities.
        user_query_ambiguity: The ambiguities injected into the user query.
        non_critical_ambiguity: The non-critical ambiguities like order, limit, etc.
        knowledge_ambiguity: The ambiguities created by masked external knowledges.
        sol_sql: The ground truth SQL solution.
        preprocess_sql: SQL queries to run before executing the solution or prediction.
        clean_up_sql: SQL queries to run after the test cases to revert any changes made to the database.
        test_cases: A set of test cases to validate the predicted corrected SQL.
        follow_up: The labeled follow up questions.
        external_knowledge: The external knowledge related to the specific task.
    """

    def __init__(
            self, config_reader: ConfigReader, user_patience: int, *args, **kwargs
    ):
        super().__init__(config_reader, *args, **kwargs)
        self.logger = get_logger(__name__)

        # gt_path_jsonl: path to the jsonl file containing the ground truth samples
        # Bird Interact provides the GT files in jsonl format, different from the HF dataset
        self.user_patience = user_patience
        self.dataset_path = Path(config_reader.dataset_path)
        self.gt_file = (
            self.dataset_path / "bird_interact_lite_gt_kg_testcases_1008.jsonl"
            if "lite" in config_reader.dataset_name
            else self.dataset_path / "bird_interact_full_gt_kg_testcases_1008.jsonl"
        )

        self.hf_dataset_name = config_reader.dataset_name
        self.config_reader = config_reader
        self.prompt_factory = PromptFactory(prompt_dir=config_reader.prompt_dir)
        self.instance2sol = dict()
        with open(self.gt_file, "r") as f:
            for line in f:
                line = json.loads(line)
                assert len(
                    line['sol_sql']) == 1, f"sol_sql contains {len(line['sol_sql'])} for id {line['instance_id']}"

                line["sol_sql"] = line["sol_sql"][0]
                self.instance2sol[line["instance_id"]] = line

        if (
                ("lite" in config_reader.dataset_name and "lite" not in str(self.gt_file))
                or
                ("full" in config_reader.dataset_name and "full" not in str(self.gt_file))
        ):
            raise ValueError(
                "The GT file does not match the specified HF dataset (lite vs full)."
            )

        # Validate at init time that the agent prompt templates match the params
        # we will provide, so mismatches are caught early rather than per-sample.
        self._validate_agent_template_vars()

    def read(self) -> list[Sample]:
        """
        Note that Bird-Interact contains also follow-up questions but we are only interested in the initial questions for evaluation,
        so we will ignore the follow-up questions.

         https://github.com/bird-bench/BIRD-Interact/blob/48805f00ff427983a57d7137650a8a04b8e5ffad/combine_public_with_gt.py#L65
        """
        dataset = datasets.load_dataset(self.hf_dataset_name)
        dataset = dataset["dev"]
        features = dataset.features.copy()
        features["test_cases"] = Sequence(Sequence(Value("string")))
        dataset = dataset.cast(features)

        # filter out category of the dataset
        if self.config_reader.filter_query_category:
            dataset = dataset.filter(lambda x: x["category"] == "Query")

        samples = []
        for line in tqdm.tqdm(dataset.to_list(), desc='processing dataset'):
            # create the sample
            db_name = line["selected_database"]
            # load the KB for the database
            kb_database = self._load_external_knowledge(db_name)
            # specific nodes from the KB used for the sample
            external_knowledge = [
                kb_database[kb_id]
                for kb_id in self.instance2sol[line["instance_id"]]["external_knowledge"]
            ]

            # remove the old key in line. read them from a ground truth file.
            del line['external_knowledge']
            del line['test_cases']
            del line['sol_sql']  # renamed target in the dataset

            sol_sql = self.instance2sol[line["instance_id"]]["sol_sql"]
            agent_params = BirdInteractAgentParams(
                database_engine=self.config_reader.database_engine,
                user_query=line["amb_user_query"],
                total_budget=self.user_patience,
            )
            simulator_params = BirdInteractUserSimulatorParams(
                db_name=db_name,
                db_schema=self._get_db_schema(db_name),
                user_query=line["amb_user_query"],
                ambiguities_json=json.dumps(line["user_query_ambiguity"]),
                correct_sql=sol_sql,
                database_engine=self.config_reader.database_engine,
            )

            sample = Sample(
                sample_id=line["instance_id"],
                messages=build_agent_messages(
                    self.prompt_factory,
                    self.config_reader.system_prompt,
                    self.config_reader.user_prompt,
                    agent_params,
                ),
                target=sol_sql,
                db_dsn=self.config_reader.db_dsn_template.format(database=db_name),
                database_engine=self.config_reader.database_engine,
                external_knowledge=external_knowledge,
                column_meanings=self._load_column_meanings(db_name),
                ddl_database_schema=self._get_db_schema(db_name),
                user_patience=self.user_patience,
                test_cases=self.instance2sol[line["instance_id"]]["test_cases"],

                metadata={
                    "kb_database": kb_database,
                    "unambig_query": line["query"],
                    "knowledge_ambiguity": line["knowledge_ambiguity"],
                    "user_query_ambiguity": line["user_query_ambiguity"],
                    "preprocess_sql": line["preprocess_sql"], # SQL queries to run before executing the solution or prediction.
                    "clean_up_sqls": line["clean_up_sqls"],  # SQL queries to run after the test cases to revert any changes made to the database.
                },
                **line,
            )

            # create user context used to give context to the LLM as a user in tool
            tool_context = ToolUserContext(
                template_params=simulator_params,
                user_simulator_prompt_folder=self.config_reader.user_simulator_prompt_folder,
                user_simulator_system_prompt=self.config_reader.user_simulator_system_prompt,
                user_simulator_user_prompt=self.config_reader.user_simulator_user_prompt,
                sample=sample.model_copy(deep=True),
            )

            sample.user_context = tool_context
            samples.append(sample)
        return samples

    @cache
    def _get_db_schema(self, database_name: str) -> str:
        """Load the database schema for a given database name.

        Reads from {dataset_path}/{database_name}/{database_name}_schema.txt.
        Returns an empty string if the file does not exist.
        """
        path = self.dataset_path / database_name / f"{database_name}_schema.txt"
        if not path.exists():
            raise FileNotFoundError(f"Database schema file not found: {path}")

        with open(path) as f:
            return f.read()

    def _validate_agent_template_vars(self) -> None:
        """Verify that the configured agent templates only need variables that
        ``BirdInteractAgentParams`` provides.  Raises ``ValueError`` at init
        time (not per-sample) if there is a mismatch.
        """
        provided = set(BirdInteractAgentParams.model_fields)
        needed: set[str] = set()
        if self.config_reader.system_prompt:
            needed.update(self.prompt_factory.get_template_variables(self.config_reader.system_prompt))
        needed.update(self.prompt_factory.get_template_variables(self.config_reader.user_prompt))

        missing = needed - provided
        if missing:
            raise ValueError(
                f"Agent prompt templates require variables not present in "
                f"BirdInteractAgentParams: {missing}. "
                f"Add them to BirdInteractAgentParams or update the template."
            )

    @cache
    def _load_column_meanings(self, database_name: str) -> dict[str, dict[str, str]]:
        """Load column meanings for a database and return as {table: {column: meaning}}.

        Reads from {dataset_path}/{database_name}/{database_name}_column_meaning_base.json.
        The source file uses flat keys of the form "db|Table|Column".
        Returns an empty dict if the file does not exist.
        """
        path = (
                self.dataset_path
                / database_name
                / f"{database_name}_column_meaning_base.json"
        )
        if not path.exists():
            raise FileNotFoundError(f"Column meanings file not found: {path}")

        with open(path) as f:
            flat = json.load(f)

        nested: dict[str, dict[str, str]] = {}
        for key, meaning in flat.items():
            parts = key.split("|")
            if len(parts) != 3:
                continue
            _, table, column = parts
            nested.setdefault(table, {})[column] = meaning
        return nested

    @cache
    def _load_external_knowledge(self, database_name: str) -> dict:
        """Load the full knowledge base for a database.

        Reads from {dataset_path}/{database_name}/{database_name}_kb.jsonl.
        Returns an empty dict if the file does not exist.
        """
        path = self.dataset_path / database_name / f"{database_name}_kb.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Knowledge base file not found: {path}")
        entries = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    if entry['children_knowledge'] == -1:
                        entry['children_knowledge'] = list()
                    entries[entry["id"]] = entry
        return entries


if __name__ == "__main__":
    config = ConfigReader()
    reader = BirdInteractReader(config, user_patience=3)
    reader.read()
