"""SQL execution tool for the agent pipeline.

Reuses the database execution logic from :mod:`src.evaluation.evaluation_metrics`
to guarantee consistency between agent-time execution and final evaluation.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from src.evaluation.evaluation_metrics import execute_sql, result_sets_match
from src.pipeline.tools.base import BaseTool


class SQLExecutionTool(BaseTool):
    """Execute a SQL query against a specified SQLite database.

    This tool wraps the shared helpers in
    :mod:`src.evaluation.evaluation_metrics` so that the same execution and
    validation logic is used during both the agent loop and the final metric
    computation.
    """

    name: str = "sql_execution"
    description: str = (
        "Execute a SQL query on an SQLite database and return the results. "
        "Optionally validate the query against a ground-truth SQL."
    )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute a SQL query and optionally compare with ground truth.

        Keyword Args:
            sql (str): The SQL query to execute.  **Required.**
            database_path (str): Path to the SQLite database.  **Required.**
            ground_truth_sql (str | None): Optional gold SQL for validation.

        Returns:
            A dict with keys ``"success"`` (bool), ``"results"`` (list of
            tuples or ``None`` on error), ``"error"`` (str or ``None``), and
            optionally ``"matches_ground_truth"`` (bool).
        """
        sql: str = kwargs["sql"]
        database_path: str = kwargs["database_path"]
        ground_truth_sql: str | None = kwargs.get("ground_truth_sql")

        output: dict[str, Any] = {
            "success": False,
            "results": None,
            "error": None,
        }

        try:
            output["results"] = execute_sql(sql, database_path)
            output["success"] = True
        except sqlite3.Error as exc:
            output["error"] = str(exc)

        if ground_truth_sql is not None:
            output["matches_ground_truth"] = result_sets_match(
                sql, ground_truth_sql, database_path
            )

        return output
