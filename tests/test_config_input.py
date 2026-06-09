import pytest

from conversation2sql.config_input import ConfigReader


def test_psql_console_defaults_off():
    assert ConfigReader().enable_psql_console is False


def test_psql_console_alone_is_allowed():
    cfg = ConfigReader(enable_psql_console=True)
    assert cfg.enable_psql_console is True
    assert cfg.enable_table_schema_tools is False


def test_both_db_tool_ablations_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        ConfigReader(enable_psql_console=True, enable_table_schema_tools=True)
