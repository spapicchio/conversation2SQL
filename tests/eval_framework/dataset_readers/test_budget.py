from conversation2sql.eval_framework.dataset_readers.bird_interact_reader import (
    _calculate_initial_budget,
)


def _make_line(critical_count: int = 0, knowledge_count: int = 0) -> dict:
    return {
        "user_query_ambiguity": {
            "critical_ambiguity": [{"i": i} for i in range(critical_count)]
        },
        "knowledge_ambiguity": [{"j": j} for j in range(knowledge_count)],
    }


class TestCalculateInitialBudget:
    def test_with_ambiguity_counted(self):
        line = _make_line(critical_count=2, knowledge_count=1)
        # 6 + 2*3 + 2*10 = 32
        assert _calculate_initial_budget(line, user_patience=10, count_ambiguity=True) == 32.0

    def test_with_ambiguity_ignored(self):
        line = _make_line(critical_count=2, knowledge_count=1)
        # 6 + 2*10 = 26 (m_amb dropped)
        assert _calculate_initial_budget(line, user_patience=10, count_ambiguity=False) == 26.0

    def test_no_ambiguity_in_line(self):
        line = _make_line()
        # both flags give the same answer when m_amb=0
        assert _calculate_initial_budget(line, user_patience=5, count_ambiguity=True) == 16.0
        assert _calculate_initial_budget(line, user_patience=5, count_ambiguity=False) == 16.0

    def test_missing_keys_default_to_zero(self):
        line = {}
        assert _calculate_initial_budget(line, user_patience=3, count_ambiguity=True) == 12.0
        assert _calculate_initial_budget(line, user_patience=3, count_ambiguity=False) == 12.0
