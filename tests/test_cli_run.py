from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from conversation2sql.cli import app

runner = CliRunner()


def test_run_forwards_config_flag():
    with (
        patch("conversation2sql.cli.PydanticParser") as mock_cls,
        patch("conversation2sql.cli.workflow_evaluation_pipeline") as mock_wf,
    ):
        mock_parser = MagicMock()
        mock_parser.parse_args_and_config.return_value = (
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        mock_cls.return_value = mock_parser
        result = runner.invoke(app, ["run", "--config", "configs/eval_pipeline_config.yaml"])
    assert result.exit_code == 0, result.output
    mock_parser.parse_args_and_config.assert_called_once_with(
        ["--config", "configs/eval_pipeline_config.yaml"]
    )
    mock_wf.assert_called_once()


def test_run_forwards_extra_overrides():
    with (
        patch("conversation2sql.cli.PydanticParser") as mock_cls,
        patch("conversation2sql.cli.workflow_evaluation_pipeline"),
    ):
        mock_parser = MagicMock()
        mock_parser.parse_args_and_config.return_value = (
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        mock_cls.return_value = mock_parser
        result = runner.invoke(
            app, ["run", "--config", "foo.yaml", "--baseline", "no_tool", "--debug"]
        )
    assert result.exit_code == 0, result.output
    mock_parser.parse_args_and_config.assert_called_once_with(
        ["--config", "foo.yaml", "--baseline", "no_tool", "--debug"]
    )


def test_run_without_config():
    with (
        patch("conversation2sql.cli.PydanticParser") as mock_cls,
        patch("conversation2sql.cli.workflow_evaluation_pipeline"),
    ):
        mock_parser = MagicMock()
        mock_parser.parse_args_and_config.return_value = (
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        mock_cls.return_value = mock_parser
        result = runner.invoke(app, ["run", "--baseline", "no_tool"])
    assert result.exit_code == 0, result.output
    mock_parser.parse_args_and_config.assert_called_once_with(["--baseline", "no_tool"])


def test_run_expands_model_profile_and_variant():
    with (
        patch("conversation2sql.cli.PydanticParser") as mock_cls,
        patch("conversation2sql.cli.workflow_evaluation_pipeline"),
    ):
        mock_parser = MagicMock()
        mock_parser.parse_args_and_config.return_value = (
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        mock_cls.return_value = mock_parser
        result = runner.invoke(
            app,
            [
                "run", "--config", "foo.yaml",
                "--model-profile", "qwen35",
                "--variant", "all_db_all_kb",
                "--predictor_enable_thinking", "true",
            ],
        )
    assert result.exit_code == 0, result.output
    passed = mock_parser.parse_args_and_config.call_args.args[0]
    assert passed[:2] == ["--config", "foo.yaml"]
    assert "--predictor_model_name" in passed
    assert passed[passed.index("--predictor_model_name") + 1] == "Qwen/Qwen3.5-9B"
    assert passed[passed.index("--predictor_temperature") + 1] == "0.6"
    assert passed[passed.index("--database_schema_type") + 1] == "ddl"
    assert passed[-2:] == ["--predictor_enable_thinking", "true"]


def test_run_non_thinking_changes_sampling():
    with (
        patch("conversation2sql.cli.PydanticParser") as mock_cls,
        patch("conversation2sql.cli.workflow_evaluation_pipeline"),
    ):
        mock_parser = MagicMock()
        mock_parser.parse_args_and_config.return_value = (
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        mock_cls.return_value = mock_parser
        result = runner.invoke(
            app,
            ["run", "--model-profile", "qwen35",
             "--predictor_enable_thinking", "false"],
        )
    assert result.exit_code == 0, result.output
    passed = mock_parser.parse_args_and_config.call_args.args[0]
    assert passed[passed.index("--predictor_temperature") + 1] == "1.0"
    assert passed[passed.index("--predictor_presence_penalty") + 1] == "1.5"


def test_run_without_presets_unchanged():
    with (
        patch("conversation2sql.cli.PydanticParser") as mock_cls,
        patch("conversation2sql.cli.workflow_evaluation_pipeline"),
    ):
        mock_parser = MagicMock()
        mock_parser.parse_args_and_config.return_value = (
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        mock_cls.return_value = mock_parser
        result = runner.invoke(app, ["run", "--config", "foo.yaml", "--baseline", "no_tool"])
    assert result.exit_code == 0, result.output
    mock_parser.parse_args_and_config.assert_called_once_with(
        ["--config", "foo.yaml", "--baseline", "no_tool"]
    )
