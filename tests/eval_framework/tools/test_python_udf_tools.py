"""Tests for Python UDF creation and cleanup impl functions.

Requires a live PostgreSQL server at localhost:5433/solar_panel with
plpython3u available. Run:
    SELECT * FROM pg_available_extensions WHERE name = 'plpython3u';
to confirm before running these tests.
"""
from __future__ import annotations

import psycopg2
import pytest

from conversation2sql.eval_framework.agents.bird_baseline.tools.bird_interact_env_tools import (
    UDFParameter,
    _safe_instance_prefix,
    cleanup_python_udfs_impl,
    create_python_udf_impl,
)

DB_DSN = "postgresql://root:123123@localhost:5433/solar_panel"
TEST_PREFIX = "pytest_udf_test"


def _plpython3u_available() -> bool:
    """Return True iff plpython3u is listed in pg_available_extensions."""
    try:
        conn = psycopg2.connect(DB_DSN)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM pg_available_extensions WHERE name = 'plpython3u'"
                )
                return cur.fetchone() is not None
        finally:
            conn.close()
    except psycopg2.OperationalError:
        return False


requires_plpython3u = pytest.mark.skipif(
    not _plpython3u_available(),
    reason="plpython3u extension not available in this PostgreSQL server",
)


@pytest.fixture(autouse=True)
def cleanup_test_udfs():
    """Drop any leftover test UDFs before and after each test."""
    cleanup_python_udfs_impl(DB_DSN, TEST_PREFIX)
    yield
    cleanup_python_udfs_impl(DB_DSN, TEST_PREFIX)


# ---------------------------------------------------------------------------
# create_python_udf_impl
# ---------------------------------------------------------------------------

@requires_plpython3u
def test_create_python_udf_impl_returns_qualified_name():
    result = create_python_udf_impl(
        function_name="add_nums",
        parameters=[
            UDFParameter(name="x", pg_type="FLOAT8"),
            UDFParameter(name="y", pg_type="FLOAT8"),
        ],
        return_type="FLOAT8",
        python_body="return x + y",
        db_dsn=DB_DSN,
        instance_id="pytest_udf_test_run",
    )
    assert result == "pytest_udf_test_run_add_nums"


@requires_plpython3u
def test_create_python_udf_impl_callable_from_select():
    prefix_safe = "pytest_udf_test_run"
    create_python_udf_impl(
        function_name="add_nums",
        parameters=[
            UDFParameter(name="x", pg_type="FLOAT8"),
            UDFParameter(name="y", pg_type="FLOAT8"),
        ],
        return_type="FLOAT8",
        python_body="return x + y",
        db_dsn=DB_DSN,
        instance_id="pytest_udf_test_run",
    )
    conn = psycopg2.connect(DB_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT {prefix_safe}_add_nums(3.0, 4.0)")
            assert cur.fetchone()[0] == pytest.approx(7.0)
    finally:
        conn.close()


@requires_plpython3u
def test_create_python_udf_impl_no_params():
    result = create_python_udf_impl(
        function_name="get_pi",
        parameters=[],
        return_type="FLOAT8",
        python_body="import math\nreturn math.pi",
        db_dsn=DB_DSN,
        instance_id="pytest_udf_test_run",
    )
    assert result == "pytest_udf_test_run_get_pi"
    conn = psycopg2.connect(DB_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pytest_udf_test_run_get_pi()")
            assert cur.fetchone()[0] == pytest.approx(3.14159, rel=1e-5)
    finally:
        conn.close()


@requires_plpython3u
def test_create_python_udf_impl_sanitizes_function_name():
    result = create_python_udf_impl(
        function_name="My-Func!",
        parameters=[],
        return_type="INTEGER",
        python_body="return 42",
        db_dsn=DB_DSN,
        instance_id="pytest_udf_test_run",
    )
    assert result == "pytest_udf_test_run_my_func_"


@requires_plpython3u
def test_create_python_udf_impl_error_returns_message():
    result = create_python_udf_impl(
        function_name="bad_func",
        parameters=[UDFParameter(name="x", pg_type="NOT_A_REAL_TYPE")],
        return_type="INTEGER",
        python_body="return 1",
        db_dsn=DB_DSN,
        instance_id="pytest_udf_test_run",
    )
    assert isinstance(result, str)
    assert "error" in result.lower() or "type" in result.lower()


# ---------------------------------------------------------------------------
# cleanup_python_udfs_impl
# ---------------------------------------------------------------------------

@requires_plpython3u
def test_cleanup_removes_prefixed_functions():
    conn = psycopg2.connect(DB_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "CREATE EXTENSION IF NOT EXISTS plpython3u"
            )
            cur.execute(
                "CREATE OR REPLACE FUNCTION pytest_udf_test_fn1() "
                "RETURNS INTEGER AS $$ return 1 $$ LANGUAGE plpython3u"
            )
            cur.execute(
                "CREATE OR REPLACE FUNCTION pytest_udf_test_fn2() "
                "RETURNS INTEGER AS $$ return 2 $$ LANGUAGE plpython3u"
            )
        conn.commit()
    finally:
        conn.close()

    cleanup_python_udfs_impl(DB_DSN, TEST_PREFIX)

    conn = psycopg2.connect(DB_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT proname FROM pg_proc WHERE proname LIKE %s",
                (f"{TEST_PREFIX}_%",),
            )
            assert cur.fetchall() == []
    finally:
        conn.close()


@requires_plpython3u
def test_cleanup_does_not_touch_other_functions():
    conn = psycopg2.connect(DB_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "CREATE EXTENSION IF NOT EXISTS plpython3u"
            )
            cur.execute(
                "CREATE OR REPLACE FUNCTION other_prefix_fn() "
                "RETURNS INTEGER AS $$ return 99 $$ LANGUAGE plpython3u"
            )
        conn.commit()
    finally:
        conn.close()

    cleanup_python_udfs_impl(DB_DSN, TEST_PREFIX)

    conn = psycopg2.connect(DB_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT proname FROM pg_proc WHERE proname = 'other_prefix_fn'")
            assert cur.fetchone() is not None
        # manual teardown
        with conn.cursor() as cur:
            cur.execute("DROP FUNCTION IF EXISTS other_prefix_fn()")
        conn.commit()
    finally:
        conn.close()


def test_cleanup_noop_when_no_functions_exist():
    # Should not raise even when there's nothing to drop
    cleanup_python_udfs_impl(DB_DSN, "nonexistent_prefix_xyz")
