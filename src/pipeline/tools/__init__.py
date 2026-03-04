"""Tool registry for the BIRD-Interact agent pipeline."""

__all__: list[str] = [
    "BaseTool",
    "SQLExecutionTool",
    "UserSimulatorTool",
]


def __getattr__(name: str) -> object:
    """Lazily import tool classes to avoid loading heavy dependencies eagerly."""
    if name == "BaseTool":
        from src.pipeline.tools.base import BaseTool
        return BaseTool

    if name == "SQLExecutionTool":
        from src.pipeline.tools.sql_execution import SQLExecutionTool
        return SQLExecutionTool

    if name == "UserSimulatorTool":
        from src.pipeline.tools.user_simulator import UserSimulatorTool
        return UserSimulatorTool

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
