from conversation2sql.eval import Sample, BaseReader
from conversation2sql.eval.registry import reader_registry


@reader_registry.register
class PlaceholderReader(BaseReader):
    """Returns two hard-coded samples — no file I/O required."""

    def read(self) -> list[Sample]:
        """Return the full list of samples for this evaluation run."""
        messages = [
            {"role": "system", "content": "You are a poetry expert"},
            {"role": "user", "content": "Write a haiku about spring"},
            {"role": "assistant", "content": "Cherry blossoms bloom..."}
        ]

        return [
            Sample(
                sample_id="sample-1",
                predictor_input=messages,
                target="SELECT * FROM employees;",
            ),
            Sample(
                sample_id="sample-2",
                predictor_input=messages,
                target="SELECT COUNT(*) FROM orders;",
            ),
        ]
