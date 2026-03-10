import json

import datasets
from datasets import Sequence, Value

from conversation2sql.config_input import ConfigReader
from conversation2sql.eval import reader_registry, Sample
from conversation2sql.eval.interfaces import BaseMessage, BaseReader
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

    def __init__(self, config_reader: ConfigReader, *args, **kwargs):
        super().__init__(config_reader, *args, **kwargs)
        self.logger = get_logger(__name__)

        # gt_path_jsonl: path to the jsonl file containing the ground truth samples
        # Bird Interact provides the GT files in jsonl format, different from the HF dataset
        self.gt_file = config_reader.dataset_kwargs['gt_path_jsonl']
        self.hf_dataset_name = config_reader.dataset_name
        self.config_reader = config_reader
        self.prompt_factory = PromptFactory(prompt_dir=config_reader.prompt_dir)
        self.set_needed_params = set()
        if config_reader.system_prompt:
            self.set_needed_params.update(self.prompt_factory.get_template_variables(config_reader.system_prompt))
        if config_reader.reader_name:
            self.set_needed_params.update(self.prompt_factory.get_template_variables(config_reader.user_prompt))

        self.instance2sol = dict()
        with open(self.gt_file, 'r') as f:
            for line in f:
                line = json.loads(line)
                line['sol_sql'] = line['sol_sql'][0]
                self.instance2sol[line['instance_id']] = line

        if ('lite' in config_reader.dataset_name and 'lite' not in self.gt_file) or (
                'full' in config_reader.dataset_name and 'full' not in self.gt_file):
            raise ValueError("The GT file does not match the specified HF dataset (lite vs full).")

    def read(self) -> list[Sample]:
        """
        Note that Bird-Interact contains also follow-up questions but we are only interested in the initial questions for evaluation,
        so we will ignore the follow-up questions.

         https://github.com/bird-bench/BIRD-Interact/blob/48805f00ff427983a57d7137650a8a04b8e5ffad/combine_public_with_gt.py#L65
         """
        dataset = datasets.load_dataset(self.hf_dataset_name)
        dataset = dataset['dev']
        features = dataset.features.copy()
        features["test_cases"] = Sequence(Sequence(Value("string")))
        dataset = dataset.cast(features)

        dataset = dataset.map(self._process_line, num_proc=16, load_from_cache_file=False)

        if isinstance(dataset, datasets.DatasetDict):
            dataset = dataset['dev']
        return [
            Sample(
                sample_id=sample['instance_id'],
                predictor_input=sample['messages'],
                target=sample['sol_sql'],
                metadata={
                    'selected_database': sample['selected_database'],
                    'unambig_query': sample['query'],
                    'knowledge_ambiguity': sample['knowledge_ambiguity'],
                    'user_query_ambiguity': sample['user_query_ambiguity'],
                    'preprocess_sql': sample['preprocess_sql'],
                    'clean_up_sqls': sample['clean_up_sqls'],
                    'test_cases': sample['test_cases'],
                    'external_knowledge': sample['external_knowledge'],
                },
                **sample
            )
            for sample in dataset.to_list()
        ]

    def _process_line(self, line):
        line['sol_sql'] = self.instance2sol[line['instance_id']]['sol_sql']
        line['external_knowledge'] = self.instance2sol[line['instance_id']].get('external_knowledge', [])
        line['test_cases'] = self.instance2sol[line['instance_id']].get('test_cases', [])
        line['messages'] = self._build_predictor_input(line)
        return line

    def _build_predictor_input(self, line: dict) -> list[BaseMessage] | str:

        template_params = {
            'database_engine': self.config_reader.database_engine,
            'user_query': line['amb_user_query'],
            'total_budget': 10,  # TODO understand how to setup in config_input.py
            'interaction_history': ''
        }

        if not self.set_needed_params.issubset(set(template_params.keys())):
            missing_params = self.set_needed_params - set(template_params.keys())
            raise ValueError(f"Missing parameters for prompt rendering: {missing_params}")

        system_prompt = self.prompt_factory.render_template(self.config_reader.system_prompt,
                                                            **template_params) if self.config_reader.system_prompt else ''

        user_prompt = self.prompt_factory.render_template(self.config_reader.user_prompt, **template_params)

        if self.config_reader.is_chat_template:
            messages = [] if system_prompt == '' else [BaseMessage(role='system', content=system_prompt)]
            messages.append(BaseMessage(role='user', content=user_prompt))
            return messages

        return system_prompt + '\n\n' + user_prompt if system_prompt != '' else user_prompt
