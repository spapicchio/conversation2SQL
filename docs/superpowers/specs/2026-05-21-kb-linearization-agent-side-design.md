# Move KB linearization from reader to agents

## Problem

`load_bird_interact_as_tasks` currently pre-computes a per-entry linearized
string representation of the external knowledge base (the
`dict[str, str]` produced by `dataset_readers/utils_kb.py::linearize_kb`)
and stores it on `TaskData.masked_agent_kb_linearized`. The
`bird_interact_env_tools` then branch on `is_kb_linearized` to choose
between the pre-built strings and the raw `ExternalKnowledgeEntry` dict.

This has three problems:

1. **Redundant pre-computation** — the linearizer runs at dataset-load
   time for every task even when `is_kb_linearized=False`, and the
   `masked_agent_kb_linearized` payload is serialized into every result
   row even though nothing downstream of the tools uses it.
2. **The per-entry subgraph strategy duplicates shared prerequisites
   across keys** — see `docs/kb_linearization.md` for the analysis.
3. **Wrong layer** — turning a structured DAG into LLM-readable text is
   an agent concern (it depends on what the agent's prompt/tools want),
   not a dataset-reader concern.

We want the reader to expose the KB as a structured DAG dict only, and
to move the string-building logic into `agents/` where the agents and
tools that actually consume it live. The new linearization follows
**Strategy 1** of `docs/kb_linearization.md`: a single flat string for
the whole KB, with a dependency-edges block followed by definitions in
topological order.

## User-facing contract

No CLI surface changes. The `ConfigReader.is_kb_linearized` flag
behaves the same way from the user's perspective: when `True`, KB tool
output (and the no-tool baseline's prompt) is the linearized flat
string; when `False`, the raw `ExternalKnowledgeEntry` JSON is exposed.

The only observable change is in the **content** of the linearized
representation:

- Old (`is_kb_linearized=True`):
  `get_all_knowledge_definitions` returned a list of per-entry
  subgraph strings, one per knowledge name.
  `get_knowledge_definition(name)` returned the full subgraph string
  for that entry (its entry plus transitive prereqs).
- New (`is_kb_linearized=True`):
  `get_all_knowledge_definitions` returns a **single string** for the
  whole KB (dependency edges block + topologically-ordered
  definitions).
  `get_knowledge_definition(name)` returns **just the single
  `[TOKEN] name - desc - formula: …` line** for that entry, with no
  prerequisites bundled.

This is intentional per the brainstorming session: the agent is
expected to call `get_all_knowledge_definitions` once to obtain the
full DAG context, and use per-entry lookups only as a cheap follow-up.

## Code structure

### `state.py`

`src/conversation2sql/eval_framework/state.py`:

```python
class TaskData(BaseModel):
    ...
    masked_agent_kb: dict[str, ExternalKnowledgeEntry] = Field(default_factory=dict)
    # masked_agent_kb_linearized removed
    is_kb_linearized: bool = False
    ...
```

- Drop the `masked_agent_kb_linearized` field entirely.
- Tighten `masked_agent_kb` from `dict[str, ExternalKnowledgeEntry | str]`
  back to `dict[str, ExternalKnowledgeEntry]`.
- Keep `is_kb_linearized` — it now signals to the agent layer that it
  should call the linearizer when producing tool output / prompts.

### `bird_interact_reader.py`

`src/conversation2sql/eval_framework/dataset_readers/bird_interact_reader.py`:

- Remove `from conversation2sql.eval_framework.dataset_readers.utils_kb import linearize_kb`.
- Remove the `linearized_kb = linearize_kb(masked_agent_kb)` step.
- Remove the `if is_kb_linearized: masked_agent_kb = linearized_kb`
  in-place overwrite.
- Remove `masked_agent_kb_linearized=linearized_kb` from the `TaskData(...)`
  construction.
- Keep `is_kb_linearized: bool = False` as a function parameter and pass
  it through to `TaskData(is_kb_linearized=is_kb_linearized)`.

### Delete `dataset_readers/utils_kb.py`

The file is entirely about linearization. Its logic moves into the new
`agents/utils_kb_linearize.py`, adapted to Strategy 1.

### New module `agents/utils_kb_linearize.py`

`src/conversation2sql/eval_framework/agents/utils_kb_linearize.py`:

Two public functions plus the LaTeX/token helpers carried over from the
old `utils_kb.py`:

```python
def linearize_kb(masked_agent_kb: dict[str, ExternalKnowledgeEntry]) -> str:
    """Strategy 1: single flat string for the whole (masked) KB.

    Layout:
        # Dependency edges (prerequisite -> dependent)
        (A_TOKEN, prerequisite_of, B_TOKEN)
        ...

        # Definitions (topological order: leaves first)
        [A_TOKEN] A name - description - formula: ...
        [B_TOKEN] B name - description - formula: ...

    Empty KB -> empty string. Single-entry KB -> just the
    `# Definitions` block (the edges block is omitted when there are no
    edges).
    """


def format_entry_line(
    name: str,
    masked_agent_kb: dict[str, ExternalKnowledgeEntry],
) -> str:
    """Format one entry as a single `[TOKEN] name - desc - formula: …` line.

    Returns `"Knowledge not found."` when `name` is not in
    `masked_agent_kb` (masked away or simply absent). No transitive
    prereqs are included.
    """
```

Internal helpers carried over verbatim from the old `utils_kb.py`:
`simplify_latex`, `_extract_token`, `_strip_token`, `_id_index`,
`_topological_sort`, plus the LaTeX replacement tables. We drop
`_collect_subgraph` and `linearize_subgraph` — both were the per-entry
BFS that Strategy 1 removes.

The single-string builder is small enough to inline at the top of the
file:

```python
def linearize_kb(masked_agent_kb):
    if not masked_agent_kb:
        return ""
    entries = list(masked_agent_kb.values())
    ordered = _topological_sort(entries)
    token_of = {n.id: _extract_token(n.knowledge) for n in ordered}
    in_kb = {n.id for n in ordered}

    edges = [
        (token_of[child_id], token_of[n.id])
        for n in ordered
        for child_id in (n.children_knowledge or [])
        if child_id in in_kb
    ]

    lines: list[str] = []
    if edges:
        lines.append("# Dependency edges (prerequisite -> dependent)")
        lines.extend(f"({a}, prerequisite_of, {b})" for a, b in edges)
        lines.append("")

    lines.append("# Definitions (topological order: leaves first)")
    for n in ordered:
        lines.append(_format_line(n, token_of[n.id]))
    return "\n".join(lines)
```

`_format_line` is the same `[TOKEN] name - desc - formula: …` formatting
the old `linearize_subgraph` used, factored out so `format_entry_line`
can call it too.

### `bird_interact_env_tools.py`

`src/conversation2sql/eval_framework/agents/bird_baseline/tools/bird_interact_env_tools.py`:

`get_all_external_knowledge_names`:

```python
@tool
def get_all_external_knowledge_names(runtime):
    return json.dumps(
        get_all_external_knowledge_names_impl(
            masked_agent_kb=runtime.context.masked_agent_kb,
        ),
        indent=2,
    )
```

Drop the branch on `is_kb_linearized` — names are the same in either
mode (`list(masked_agent_kb.keys())`).

`get_knowledge_definition`:

```python
@tool
def get_knowledge_definition(knowledge_name, runtime):
    if runtime.context.is_kb_linearized:
        line = format_entry_line(knowledge_name, runtime.context.masked_agent_kb)
        return json.dumps({"knowledge": line}, indent=2)
    return json.dumps(
        get_knowledge_definition_impl(
            knowledge_name=knowledge_name,
            masked_agent_kb=runtime.context.masked_agent_kb,
        ),
        indent=2,
    )
```

`get_all_knowledge_definitions`:

```python
@tool
def get_all_knowledge_definitions(runtime):
    if runtime.context.is_kb_linearized:
        flat = linearize_kb(runtime.context.masked_agent_kb)
        return json.dumps({"knowledge": flat}, indent=2)
    return json.dumps(
        get_all_knowledge_definitions_impl(
            masked_agent_kb=runtime.context.masked_agent_kb,
        ),
        indent=2,
    )
```

The `*_impl` functions keep their existing signatures.
`get_all_external_knowledge_names_impl` is called in both modes (the
list of names is identical). `get_knowledge_definition_impl` and
`get_all_knowledge_definitions_impl` are only reached when
`is_kb_linearized=False`. All three have their `masked_agent_kb`
parameter type tightened from `dict[str, ExternalKnowledgeEntry | str]`
to `dict[str, ExternalKnowledgeEntry]`.

### `no_tool_baseline/baseline_model.py`

`src/conversation2sql/eval_framework/agents/no_tool_baseline/baseline_model.py`:

```python
from conversation2sql.eval_framework.agents.utils_kb_linearize import linearize_kb
...

kb_for_prompt = (
    linearize_kb(single_task.masked_agent_kb)
    if single_task.is_kb_linearized
    else single_task.masked_agent_kb
)
user_messages = build_omnisql_prompt(
    params={
        "schema": single_task.ddl_database_schema,
        "question": single_task.task_question,
        "kb": kb_for_prompt,
    }
)
```

When the flag is off, behavior is unchanged. When on, the Jinja
template receives a clean linearized string instead of a Pydantic-repr
dict dump. (No prompt template change is needed: the template already
emits `{{ kb }}` as-is — strings and dicts both stringify.)

### `main_pipe_workflow.py`

No change — the run-folder slug logic at `_compute_results_subdir`
already reads `config_reader.is_kb_linearized` and that field stays in
place.

## Tests

### Fixture updates

- `tests/eval_framework/test_main_pipe_workflow.py`: remove the
  `"masked_agent_kb_linearized": {}` entry from the synthetic task
  dict at line ~56.
- `tests/eval_framework/tools/conftest.py`: nothing to remove (it does
  not set `masked_agent_kb_linearized`). Confirm the `TaskData` fixture
  still typechecks against the tightened `masked_agent_kb` type.

### Existing impl tests

`tests/eval_framework/tools/test_bird_interact_env_tools.py` currently
exercises the non-linearized impl path. Those tests stay — they cover
the `is_kb_linearized=False` branch end-to-end.

Add new tests for the linearized branch of the three KB tools. Because
the linearized branch is implemented directly in the `@tool` wrappers
(it calls `linearize_kb` / `format_entry_line` rather than a new
`*_impl`), we test it by invoking the tool wrapper with a synthetic
runtime carrying `is_kb_linearized=True`, mirroring the existing tool-
level tests in the same file.

### New module tests

`tests/eval_framework/agents/test_utils_kb_linearize.py` (new file):

- `linearize_kb({})` → `""`.
- Single leaf entry (`children_knowledge=[]`) → output has the
  `# Definitions` header and one line; no edges block.
- DAG with shared prereq `A` used by both `B` and `C` → `A` appears
  once in the edges block (as `(A, prerequisite_of, B)` and
  `(A, prerequisite_of, C)`) and once in the definitions block.
- LaTeX simplification: an entry with `\frac{X}{Y}` in `definition`
  produces `(X) / (Y)` in the output line.
- `format_entry_line("missing", {...})` → `"Knowledge not found."`.
- `format_entry_line` for a present entry returns the same
  `[TOKEN] name - desc - formula: …` line that appears in
  `linearize_kb`'s output for the same KB (consistency check).

## Docs

Update three CLAUDE.md files:

- `src/conversation2sql/eval_framework/CLAUDE.md`: drop the bullet
  about `masked_agent_kb_linearized`; note that `is_kb_linearized` is
  now an agent-side toggle.
- `src/conversation2sql/eval_framework/dataset_readers/CLAUDE.md`:
  remove the `linearize_kb` paragraph; the reader no longer produces a
  linearized representation.
- `src/conversation2sql/eval_framework/agents/CLAUDE.md`: add a note
  about `utils_kb_linearize.py` as the shared helper used by
  `bird_baseline/tools/bird_interact_env_tools.py` and
  `no_tool_baseline/baseline_model.py`.

## Out of scope

- The `ExternalKnowledgeEntry` Pydantic model is unchanged.
- `full_knowledge_base` and `gt_knowledge_base` are untouched; they
  remain `dict[str, ExternalKnowledgeEntry]`.
- The user-facing tool patience costs are unchanged.
- No changes to `ConfigReader` or CLI flags.
- The `2026-05-16-resume-from-results-design.md` identity-fields list
  already includes `is_kb_linearized`; no change there.
