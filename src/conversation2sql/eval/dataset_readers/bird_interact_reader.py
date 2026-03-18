import json
from functools import cache
from pathlib import Path

import datasets
import tqdm
from datasets import Sequence, Value
from frozendict import frozendict

from conversation2sql.config_input import ConfigReader
from conversation2sql.eval import reader_registry, Sample
from conversation2sql.eval.interfaces import BaseMessage, BaseReader, ToolUserContext
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
        self.set_needed_params = set()
        if config_reader.system_prompt:
            self.set_needed_params.update(
                self.prompt_factory.get_template_variables(config_reader.system_prompt)
            )
        if config_reader.reader_name:
            self.set_needed_params.update(
                self.prompt_factory.get_template_variables(config_reader.user_prompt)
            )

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

            # remove old key in line. read them from ground truth file.
            del line['external_knowledge']
            del line['test_cases']

            sample = Sample(
                sample_id=line["instance_id"],
                messages=self._build_messages(self.config_reader.database_engine,
                                              line["amb_user_query"],
                                              self.user_patience),
                target=self.instance2sol[line["instance_id"]]["sol_sql"],
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
                    "preprocess_sql": line["preprocess_sql"],
                    "clean_up_sqls": line["clean_up_sqls"],
                },
                **line,
            )

            # create user context used to give context to the LLM as a user in tool
            tool_context = ToolUserContext(
                template_params={
                    "db_schema": db_name,
                    "user_query": line["amb_user_query"],
                    "ambiguities_json": line["user_query_ambiguity"],
                    "correct_sql": self.instance2sol[line["instance_id"]]["sol_sql"],
                },
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

    def _build_messages(self, database_engine, amb_user_query, user_patience) -> list[BaseMessage] | str:
        template_params = {
            "database_engine": database_engine,
            "user_query": amb_user_query,
            "total_budget": user_patience,
        }

        if not self.set_needed_params.issubset(set(template_params.keys())):
            missing_params = self.set_needed_params - set(template_params.keys())
            raise ValueError(f"Missing parameters for prompt rendering: {missing_params}")

        system_prompt = []
        if self.config_reader.system_prompt:
            system_prompt = self.prompt_factory.render_template(self.config_reader.system_prompt,
                                                                frozendict(template_params))
            system_prompt = [BaseMessage(role="system", content=system_prompt)]

        user_prompt = [BaseMessage(
            role="user",
            content=self.prompt_factory.render_template(self.config_reader.user_prompt, frozendict(template_params))
        )]

        messages = system_prompt + user_prompt
        return messages

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
