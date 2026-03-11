from langchain_core.tools import tool

from conversation2sql.eval.registry import tool_registry


# https://docs.langchain.com/oss/python/langchain/tools


@tool_registry.register(name='search')
@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


@tool_registry.register(name='get_weather')
@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"


@tool_registry.register(name='sql_query_tool')
@tool
def sql_query_tool(query: str) -> str:
    """Execute a SQL query and return results."""
    return f"SQL result for: {query}"


@tool_registry.register(name='db_query_tool')
@tool
def db_query_tool(query: str) -> str:
    """Execute a database query and return results."""
    return f"DB result for: {query}"
