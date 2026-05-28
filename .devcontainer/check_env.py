"""
Environment sanity checks for the conversation2SQL dev container.
Run this after rebuilding the container to verify everything is wired up correctly.

    python .devcontainer/check_env.py
"""

import subprocess
import sys
import unittest


class TestPythonDependencies(unittest.TestCase):
    """Core Python packages required by the project."""

    def _assert_importable(self, module: str):
        try:
            __import__(module)
        except ImportError as e:
            self.fail(f"Could not import '{module}': {e}")

    def test_jinja2(self):
        self._assert_importable("jinja2")

    def test_pydantic(self):
        self._assert_importable("pydantic")

    def test_datasets(self):
        self._assert_importable("datasets")

    def test_langchain(self):
        self._assert_importable("langchain")

    def test_langgraph(self):
        self._assert_importable("langgraph")

    def test_litellm(self):
        self._assert_importable("litellm")

    def test_loguru(self):
        self._assert_importable("loguru")

    def test_pandas(self):
        self._assert_importable("pandas")

    def test_yaml(self):
        self._assert_importable("yaml")


class TestProjectPackage(unittest.TestCase):
    """The conversation2sql package itself must be installed and importable."""

    def test_conversation2sql_importable(self):
        try:
            import conversation2sql  # noqa: F401
        except ImportError as e:
            self.fail(
                f"conversation2sql package not importable: {e}\n"
                "Run: uv venv --system-site-packages && uv sync && source .venv/bin/activate"
            )

    def test_dataset_readers(self):
        try:
            from conversation2sql.eval_framework.dataset_readers import (  # noqa: F401
                load_bird_interact_as_tasks,
            )
        except ImportError as e:
            self.fail(f"dataset_readers not importable: {e}")


class TestSystemTools(unittest.TestCase):
    """CLI tools that must be available on PATH."""

    def _assert_command(self, cmd: list[str], name: str):
        result = subprocess.run(cmd, capture_output=True)
        self.assertEqual(
            result.returncode, 0,
            f"'{name}' not found or returned non-zero. Install it or check PATH."
        )

    def test_uv_available(self):
        self._assert_command(["uv", "--version"], "uv")

    def test_gpustat_available(self):
        self._assert_command(["gpustat", "--version"], "gpustat")

    def test_just_available(self):
        self._assert_command(["just", "--version"], "just")


class TestPostgresLite(unittest.TestCase):
    """PostgreSQL lite instance (port 5432) must be reachable."""

    HOST = "localhost"
    PORT = 5432
    USER = "root"
    PASSWORD = "123123"

    @classmethod
    def setUpClass(cls):
        try:
            import psycopg2  # noqa: F401
            cls.psycopg2_available = True
        except ImportError:
            cls.psycopg2_available = False

    def _connect(self, port: int):
        import psycopg2
        return psycopg2.connect(
            host=self.HOST,
            port=port,
            user=self.USER,
            password=self.PASSWORD,
            connect_timeout=5,
        )

    def test_psycopg2_available(self):
        self.assertTrue(
            self.psycopg2_available,
            "psycopg2 not installed. Run: uv pip install psycopg2-binary"
        )

    def test_postgres_lite_reachable(self):
        if not self.psycopg2_available:
            self.skipTest("psycopg2 not available")
        try:
            conn = self._connect(self.PORT)
            conn.close()
        except Exception as e:
            self.fail(
                f"Cannot connect to postgresql (lite) on port {self.PORT}: {e}\n"
                "Is the 'postgresql' container running? Check: docker ps"
            )

    def test_postgres_lite_has_databases(self):
        if not self.psycopg2_available:
            self.skipTest("psycopg2 not available")
        try:
            conn = self._connect(self.PORT)
            cur = conn.cursor()
            cur.execute("SELECT datname FROM pg_database WHERE datistemplate = false;")
            databases = [row[0] for row in cur.fetchall()]
            conn.close()
            self.assertGreater(
                len(databases), 0,
                "postgres (lite) connected but returned no user databases"
            )
        except Exception as e:
            self.fail(f"Query failed on postgres (lite): {e}")


class TestPostgresFull(unittest.TestCase):
    """PostgreSQL full instance (port 5433) must be reachable."""

    HOST = "localhost"
    PORT = 5433
    USER = "root"
    PASSWORD = "123123"

    @classmethod
    def setUpClass(cls):
        try:
            import psycopg2  # noqa: F401
            cls.psycopg2_available = True
        except ImportError:
            cls.psycopg2_available = False

    def _connect(self, port: int):
        import psycopg2
        return psycopg2.connect(
            host=self.HOST,
            port=port,
            user=self.USER,
            password=self.PASSWORD,
            connect_timeout=5,
        )

    def test_postgres_full_reachable(self):
        if not self.psycopg2_available:
            self.skipTest("psycopg2 not available")
        try:
            conn = self._connect(self.PORT)
            conn.close()
        except Exception as e:
            self.fail(
                f"Cannot connect to postgresql (full) on port {self.PORT}: {e}\n"
                "Is the 'postgresql_full' container running? Check: docker ps"
            )

    def test_postgres_full_has_databases(self):
        if not self.psycopg2_available:
            self.skipTest("psycopg2 not available")
        try:
            conn = self._connect(self.PORT)
            cur = conn.cursor()
            cur.execute("SELECT datname FROM pg_database WHERE datistemplate = false;")
            databases = [row[0] for row in cur.fetchall()]
            conn.close()
            self.assertGreater(
                len(databases), 0,
                "postgres (full) connected but returned no user databases"
            )
        except Exception as e:
            self.fail(f"Query failed on postgres (full): {e}")


if __name__ == "__main__":
    # Run with verbose output so each check is clearly listed
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(unittest.TestLoader().loadTestsFromModule(
        sys.modules[__name__]
    ))
    sys.exit(0 if result.wasSuccessful() else 1)
