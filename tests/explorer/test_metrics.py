import pytest

from explorer.metrics import ReliabilityStats, passk_curve, reliability_metrics


def _samples(passes: int, total: int) -> list[dict]:
    """A group of `total` records, `passes` of which passed."""
    return [{"execution_accuracy": i < passes} for i in range(total)]


class TestPasskCurve:
    def test_all_pass(self):
        curve = passk_curve({"a": _samples(3, 3)})
        assert curve == {1: pytest.approx(1.0), 2: pytest.approx(1.0), 3: pytest.approx(1.0)}

    def test_known_values_n10_c3(self):
        curve = passk_curve({"a": _samples(3, 10)})
        assert curve[1] == pytest.approx(0.3)
        assert curve[5] == pytest.approx(0.9166666, abs=1e-6)
        assert curve[10] == pytest.approx(1.0)

    def test_empty(self):
        assert passk_curve({}) == {}

    def test_k_capped_by_smallest_group(self):
        # one instance has 2 samples, one has 3 → k=3 averages only the 3-sample instance
        curve = passk_curve({"a": _samples(0, 2), "b": _samples(3, 3)})
        assert set(curve) == {1, 2, 3}
        assert curve[3] == pytest.approx(1.0)  # only "b" qualifies for k=3


class TestReliabilityMetrics:
    def test_empty(self):
        rel = reliability_metrics({})
        assert rel == ReliabilityStats(0.0, 0.0, 0.0, 0.0, {})

    def test_all_pass_single_instance(self):
        rel = reliability_metrics({"a": _samples(3, 3)})
        assert rel.avg_performance == pytest.approx(1.0)
        assert rel.aptitude == pytest.approx(1.0)
        assert rel.unreliability == pytest.approx(0.0)
        assert rel.reliability == pytest.approx(1.0)

    def test_flaky_mix(self):
        # A: [1,1] all pass; B: [1,0] half pass
        rel = reliability_metrics({"a": _samples(2, 2), "b": _samples(1, 2)})
        assert rel.avg_performance == pytest.approx(0.75)        # (1.0 + 0.5)/2
        assert rel.aptitude == pytest.approx(0.95)               # (1.0 + 0.9)/2
        assert rel.unreliability == pytest.approx(0.4)           # (0.0 + 0.8)/2
        assert rel.reliability == pytest.approx(0.6)

    def test_empty_group_skipped(self):
        rel = reliability_metrics({"a": [], "b": _samples(1, 1)})
        assert rel.avg_performance == pytest.approx(1.0)
        assert rel.reliability == pytest.approx(1.0)
