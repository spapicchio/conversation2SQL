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


def test_dataset_variant_default_is_lite():
    cfg = ConfigReader()
    assert cfg.dataset_variant == "lite"
    assert cfg.dataset_path == "data/bird_interact/bird-interact-lite"
    assert (
        cfg.dataset_name_jsonl
        == "data/bird_interact/bird-interact-lite/bird_interact_data_GT.jsonl"
    )
    assert cfg.deep_catalog_root == "data/bird_interact/catalog_bird_interact_lite"
    assert cfg.db_dsn_template == "postgresql://root:123123@localhost:5432/{database}"


def test_dataset_variant_full_moves_all_paths_and_port():
    cfg = ConfigReader(dataset_variant="full")
    assert cfg.dataset_path == "data/bird_interact/bird-interact-full"
    assert (
        cfg.dataset_name_jsonl
        == "data/bird_interact/bird-interact-full/bird_interact_data_GT.jsonl"
    )
    assert cfg.deep_catalog_root == "data/bird_interact/catalog_bird_interact_full"
    assert cfg.db_dsn_template == "postgresql://root:123123@localhost:5433/{database}"


def test_db_host_credentials_override_flow_into_dsn():
    cfg = ConfigReader(
        dataset_variant="full", db_host="slurm-node-1", db_user="u", db_password="p"
    )
    assert cfg.db_dsn_template == "postgresql://u:p@slurm-node-1:5433/{database}"


def test_data_root_override_repaths_everything():
    cfg = ConfigReader(data_root="/custom/data")
    assert cfg.dataset_path == "/custom/data/bird-interact-lite"
    assert cfg.deep_catalog_root == "/custom/data/catalog_bird_interact_lite"


def test_computed_path_fields_present_in_model_dump():
    dumped = ConfigReader().model_dump()
    for key in ("dataset_path", "dataset_name_jsonl", "deep_catalog_root", "db_dsn_template"):
        assert key in dumped, f"{key} missing from model_dump()"


def test_dataset_variant_rejects_unknown_value():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ConfigReader(dataset_variant="medium")
