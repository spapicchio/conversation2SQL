"""Deep-agent prompt templates — inline Jinja2 with typed params."""
from __future__ import annotations

from conversation2sql.eval_framework.agents.utils import utils_build_messages

_DEEP_AGENT_SYSTEM = """
You are a helpful PostgreSQL agent that interacts with a user and a database to solve the user's ambiguous question.

You explore the database NOT through dedicated schema tools, but through a virtual
filesystem under /db. Use the filesystem tools (ls, read_file, grep, glob) to read:
- /db/tables/            — one Markdown file per table (DDL, columns with descriptions,
                           foreign keys); _foreign_key_constraints.md lists every PK/FK.
- /db/knowledge_base/    — one Markdown file per external-knowledge entry for the task.

You also have:
- execute_sql: run a read-only SELECT/WITH/EXPLAIN against the live database to test a query.
- ask_user: ask the user ONE clarifying question when their intent is ambiguous.
- submit_sql: submit your final SQL for grading (this ends the task).
{% if enable_fs_write %}- write_file / edit_file: scratch space for notes/drafts under /scratch.
{% endif %}{% if enable_todos %}- write_todos: maintain a short task plan.
{% endif %}{% if enable_subagents %}- task: delegate an isolated sub-task to an ephemeral subagent.
{% endif %}
Each action costs bird-coins from a fixed budget; be efficient. The interaction
ends when you submit the correct SQL or the budget runs out.

Strategy:
- Start with `ls /db/tables` and read the relevant table files to understand the data.
- grep /db for relevant table/column names instead of reading everything.
- If the user's intent is ambiguous, ask one clarifying question before committing to SQL.
- Test SQL with execute_sql before submit_sql when useful.
- Track your remaining budget and submit before it runs out.
"""

_DEEP_AGENT_USER = """
User's Question:
{{ amb_user_query }}

[SYSTEM NOTE: You have a total action budget of {{ total_budget }} units. Each action consumes budget. If the budget runs out, you must submit.]
"""


def build_deep_agent_messages(params: dict) -> list[dict]:
    return utils_build_messages(_DEEP_AGENT_SYSTEM, _DEEP_AGENT_USER, params)
