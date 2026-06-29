# Note
0. it seems that when the model has more budget it is performing better but the actual budget usage is lower...strange...mostly of the model fails are due to budget exhausted probably because the model does not know how to solve it -> we need to change the boxplot between passed and not-passed

2. jsonb is usually a problem in this postgreSQL?

3. why budged death and repeated identical call are higly correlated?


# TODO
7. Add a file MD that represents the flow of the agent history and what are the differences with what is shown in the explorer

1. in budget spent plot, I want a red line that represents the maximum budget available (only when not ambiguous)

## explorer: wire in deep_agent baseline + its ablations
1. `explorer/index.py` → `render_args()`: emit the four `deep_enable_*` reader flags (`deep_enable_todos`, `deep_enable_subagents`, `deep_enable_summarization`, `deep_enable_fs_write`) so deep_agent ablations diff as one token in experiments.csv. Currently they're omitted, so two deep_agent runs with different flags render an identical `args` string. (`enable_psql_console` / `enable_python_udf` are also missing from `render_args` — same pre-existing gap; fold them in too.)
2. `explorer/patterns.py` → `_detect_kb_blind()`: **correctness bug**. It flags KB-blind when `get_knowledge_definition` is never called, but deep_agent has no such tool — it reads the KB via `read_file`/`grep` on `/db/knowledge_base.md`. Result: every KB-needing deep_agent run is falsely flagged. Fix: also accept an FS read whose arg path references `knowledge_base.md` (mirror how `_validates_query` handles `psql_console`). Update `PATTERN_DESCRIPTIONS["kb_blind"]` text accordingly.
3. `explorer/colors.py` → `TOOL_COLORS`: pin the deepagents tool names (`ls`, `read_file`, `glob`, `grep`, `write_file`, `edit_file`, `write_todos`, `task`) so the tool-usage / positional charts get distinct, stable colors instead of hashed (possibly colliding) ones. Widen the "must match keys of `TOOL_COSTS`" comment to also cover `FS_TOOL_COSTS` in `deep_agent/filesystem_seed.py`.
4. Add also System Prompt for deep_agent generation

## Done:
1. when the submit_sql passed, the conversation should end without the "Conversation limit reached. User patience exhausted"

8. change prompt descriptions in one place and use them in prompt and tool descriptions

6. Execute sql must be truncated row by row rather than characters count

2. I want to see the tool budget also in the explorer page whene we drill down into one conversation

5. I want to add a graph where the bucket of the conversaion is correlated with the passed/fail