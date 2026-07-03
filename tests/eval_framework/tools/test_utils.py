"""Unit tests for bird_baseline/tools/utils.py's SQL-normalization helpers."""
from __future__ import annotations

from conversation2sql.eval_framework.agents.bird_baseline.tools.utils import remove_round


class TestRemoveRound:
    def test_strips_plain_round_no_cast(self):
        sql = "SELECT ROUND(AVG(mcs), 2) FROM t"
        assert remove_round(sql) == "SELECT AVG(mcs) FROM t"

    def test_strips_inner_numeric_cast_within_round(self):
        # Postgres's two-arg ROUND(numeric, int) requires a numeric argument,
        # so a model rounding a double-precision aggregate must cast it —
        # AVG(x::numeric) and AVG(x) diverge in low-order decimal digits.
        sql = "SELECT ROUND(AVG(mcs::numeric), 2) FROM t"
        assert remove_round(sql) == "SELECT AVG(mcs) FROM t"

    def test_strips_outer_numeric_cast_within_round(self):
        sql = "SELECT ROUND(AVG(mcs)::numeric, 2) FROM t"
        assert remove_round(sql) == "SELECT AVG(mcs) FROM t"

    def test_preserves_numeric_cast_outside_round(self):
        # Only the cast inside the stripped ROUND(...) call is a rounding
        # artifact; a cast elsewhere in the query may be load-bearing and
        # must survive.
        sql = "SELECT x::numeric FROM t WHERE ROUND(y::numeric, 2) > 1"
        assert remove_round(sql) == "SELECT x::numeric FROM t WHERE y > 1"
