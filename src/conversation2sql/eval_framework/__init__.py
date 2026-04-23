"""Modular evaluation framework for text-to-SQL with conversation.

Three independent MCP-based microservices:
  - DB Environment  (port 6002): SQL execution, schema, knowledge, SQL grading
  - User Simulator  (port 6001): Two-stage LLM pipeline that answers clarifications
  - System Agent    (port 6000): LangGraph ReAct LLM that calls the other two via MCP

All three can run on the same machine (stdio transport, HPC single-node arrays)
or on separate nodes (streamable-http transport, Docker, multi-node HPC).

Quick start — run each service, then the orchestrator:
    python -m conversation2sql.framework.db_environment.server
    python -m conversation2sql.framework.user_simulator.server
    python -m conversation2sql.framework.system_agent.server
    python -m conversation2sql.framework.orchestrator.runner --mode a-interact

Transport selection (env var):
    C2SQL_TRANSPORT=stdio             # HPC single-node: no networking
    C2SQL_TRANSPORT=streamable-http   # default: HTTP per tool call
"""
