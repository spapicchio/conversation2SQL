from conversation2sql.eval_framework.agents.deep_agent import agent_code
from langchain.agents.middleware import TodoListMiddleware, SummarizationMiddleware
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from deepagents import FilesystemMiddleware, SubAgentMiddleware


def _task(task_data, **flags):
    for k, v in flags.items():
        setattr(task_data, k, v)
    return task_data


def _fake_model() -> FakeListChatModel:
    # deepagents introspects the model when compiling a subagent, so the
    # middleware tests need a real BaseChatModel rather than a MagicMock.
    return FakeListChatModel(responses=["ok"])


def test_fs_middleware_read_only_by_default(task_data):
    mw = agent_code._build_fs_middleware(enable_fs_write=False)
    names = {t.name for t in mw.tools}
    assert {"ls", "read_file", "glob", "grep"} <= names
    assert "write_file" not in names and "edit_file" not in names
    assert "execute" not in names  # inert on StateBackend; hidden


def test_fs_middleware_adds_writes_when_enabled(task_data):
    mw = agent_code._build_fs_middleware(enable_fs_write=True)
    names = {t.name for t in mw.tools}
    assert "write_file" in names and "edit_file" in names


def test_middleware_minimal_by_default(task_data):
    mws = agent_code._build_deep_middleware(_task(task_data), _fake_model())
    types = {type(m) for m in mws}
    assert FilesystemMiddleware in types
    assert TodoListMiddleware not in types
    assert SummarizationMiddleware not in types
    assert SubAgentMiddleware not in types
    # patience middleware present (match by callable identity)
    from conversation2sql.eval_framework.agents.bird_baseline.agent_callback import (
        tool_wrapper_patience_and_submit,
    )
    assert tool_wrapper_patience_and_submit in mws


def test_middleware_flags_add_components(task_data):
    mws = agent_code._build_deep_middleware(
        _task(
            task_data,
            deep_enable_todos=True,
            deep_enable_summarization=True,
            deep_enable_subagents=True,
        ),
        _fake_model(),
    )
    types = {type(m) for m in mws}
    assert TodoListMiddleware in types
    assert SummarizationMiddleware in types
    assert SubAgentMiddleware in types


def test_tools_include_reused_and_ask_user(task_data, make_chat_model):
    tools = agent_code._build_deep_tools(
        _task(task_data),
        model_user_parsing=make_chat_model("<s>x</s>"),
        model_user_generator=make_chat_model("<s>y</s>"),
    )
    names = {t.name for t in tools}
    assert {"execute_sql", "ask_user", "submit_sql"} <= names
