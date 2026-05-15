from conversation2sql.eval_framework.dataset_readers.sql_usage_extractor import (
    extract_table_in_gt_sql,
)


def _table_to_columns() -> dict[str, set[str]]:
    return {
        "users": {"id", "name", "email"},
        "orders": {"id", "user_id", "amount", "status"},
    }


def test_alias_join_dedup():
    sql = (
        "SELECT u.id, o.amount "
        "FROM users AS u JOIN orders o ON u.id = o.user_id "
        "WHERE o.status = 'paid'"
    )
    usage, error = extract_table_in_gt_sql(sql, _table_to_columns())

    assert error is None
    assert usage == {
        "users": ["id"],
        "orders": ["amount", "status", "user_id"],
    }


def test_table_star_expansion():
    sql = (
        "SELECT u.*, o.id "
        "FROM users u JOIN orders o ON u.id = o.user_id"
    )
    usage, error = extract_table_in_gt_sql(sql, _table_to_columns())

    assert error is None
    assert usage == {
        "users": ["email", "id", "name"],
        "orders": ["id", "user_id"],
    }


def test_bare_star_expansion():
    sql = "SELECT * FROM users u JOIN orders o ON u.id = o.user_id"
    usage, error = extract_table_in_gt_sql(sql, _table_to_columns())

    assert error is None
    assert usage == {
        "users": ["email", "id", "name"],
        "orders": ["amount", "id", "status", "user_id"],
    }


def test_ambiguous_unqualified_column_goes_unknown():
    sql = "SELECT id FROM users u CROSS JOIN orders o"
    usage, error = extract_table_in_gt_sql(sql, _table_to_columns())

    assert error is None
    assert usage == {
        "unknown": ["id"],
    }


def test_join_using_handles_list_identifiers():
    sql = "SELECT u.name FROM users u JOIN orders o USING (id)"
    usage, error = extract_table_in_gt_sql(sql, _table_to_columns())

    assert error is None
    assert usage == {
        "users": ["id", "name"],
        "orders": ["id"],
    }


def test_join_on_disambiguates_unqualified_column():
    sql = "SELECT id FROM users u JOIN orders o ON u.id = o.user_id"
    usage, error = extract_table_in_gt_sql(sql, _table_to_columns())

    assert error is None
    assert usage == {
        "users": ["id"],
        "orders": ["user_id"],
    }


def test_disambiguation_success():
    sql = "SELECT email FROM users u JOIN orders o ON u.id = o.user_id"
    usage, error = extract_table_in_gt_sql(sql, _table_to_columns())

    assert error is None
    assert usage == {
        "users": ["email", "id"],
        "orders": ["user_id"],
    }


def test_cte_coverage():
    sql = (
        "WITH recent_orders AS ("
        "SELECT o.user_id, o.amount, u.email "
        "FROM orders o JOIN users u ON u.id = o.user_id"
        ") "
        "SELECT ro.user_id, ro.amount "
        "FROM recent_orders ro WHERE ro.amount > 10"
    )
    usage, error = extract_table_in_gt_sql(sql, _table_to_columns())

    assert error is None
    assert usage == {
        "orders": ["amount", "user_id"],
        "users": ["email", "id"],
        "cte_recent_orders": ["amount", "user_id"],
    }


def test_parse_error_contract():
    sql = "SELECT FROM"
    usage, error = extract_table_in_gt_sql(sql, _table_to_columns())

    assert usage == {}
    assert error


def test_ddl_schema_adds_missing_columns():
    sql = "SELECT age FROM users"
    table_to_columns = {"users": {"id"}}
    ddl = "CREATE TABLE users (id int, age int);"

    usage, error = extract_table_in_gt_sql(
        sql,
        table_to_columns,
        ddl_schema=ddl,
    )

    assert error is None
    assert usage == {"users": ["age"]}
