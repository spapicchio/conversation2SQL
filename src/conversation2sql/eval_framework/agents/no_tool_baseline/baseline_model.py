from typing import Any
from conversation2sql.eval_framework.agents.utils_extract_sql_from_response import (
    extract_sql_from_response,
)


from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from conversation2sql.eval_framework.agents.bird_baseline.tools import submit_sql_impl
from conversation2sql.eval_framework.agents.no_tool_baseline.prompts import (
    build_omnisql_prompt,
)
from conversation2sql.eval_framework.agents.utils import (
    utils_extract_ai_metadata,
    utils_single_msg_to_str,
)
from conversation2sql.eval_framework.state import TaskData
from conversation2sql.logger import get_logger

logger = get_logger(__name__)


def run_baseline_no_tool(
    single_task: TaskData,
    model_agent: BaseChatModel,
) -> dict[str, Any]:
    user_messages = build_omnisql_prompt(
        params={
            "schema": single_task.ddl_database_schema,
            "question": single_task.task_question,
            "kb": single_task.masked_agent_kb,
        }
    )

    ai_msg: AIMessage = model_agent.invoke(user_messages)  # pyrefly: ignore
    raw_text = utils_single_msg_to_str(ai_msg)

    sql = extract_sql_from_response(raw_text)
    if sql is None:
        submit_outcome: dict = {
            "passed": False,
            "error": "no_sql_block_found",
            "raw_text": raw_text[-2000:],
        }
    else:
        submit_outcome = submit_sql_impl(
            sql=sql,
            sol_sqls=single_task.sol_sql,
            db_dsn=single_task.db_dsn,
            conditions=single_task.sql_query_conditions,
        )

    ai_meta = utils_extract_ai_metadata(ai_msg, tool_costs={})

    user_msg_dict = {"role": "user", "content": user_messages[-1]["content"]}
    ai_msg_dict = {"role": "ai", "content": raw_text, **ai_meta}
    synthetic_tool_msg = {
        "role": "tool",
        "tool_name": "submit_sql_offline",
        "status": "success" if submit_outcome.get("passed") else "error",
        "content": submit_outcome,
    }
    messages = [user_msg_dict, ai_msg_dict, synthetic_tool_msg]

    prompt_tokens = ai_meta.get("prompt_tokens", 0) or 0
    completion_tokens = ai_meta.get("completion_tokens", 0) or 0
    total_tokens = ai_meta.get("total_tokens", 0) or 0

    return {
        "messages": messages,
        "initial_user_patience": None,
        "updated_user_patience": None,
        "tool_called_patience": [],
        "total_cost": ai_meta.get("cost_usd", 0.0) or 0.0,
        "total_tokens": total_tokens,
        "total_prompt_tokens": prompt_tokens,
        "total_completion_tokens": completion_tokens,
        "mean_prompt_tokens": float(prompt_tokens),
        "mean_completion_tokens": float(completion_tokens),
        "tool_calls_in_order": [],
        "predicted_sql": sql,
        "target_sql": single_task.sol_sql[0],
        "execution_accuracy": bool(submit_outcome.get("passed", False)),
    }


if __name__ == "__main__":
    text = """  
    The question asks for:\n- Which observatory stations are discovering the most potential technosignatures?\n- For each station, display:\n  1. Observatory name\n  2. How many signals meet our technosignature criteria\n  3. Their average TOLS score\n  4. Average BFR values\n  5. What percentage of all detected technosignatures they've found.\n- Ranked by stations with the most discoveries first.\n\nFirst, I need to understand the schema and find where \"technosignatures\" are defined.\nLooking at `signalclassification`, there's `sigclasstype` which could be 'Artificial' or 'Unknown' or something else. But \"technosignatures\" might be indicated by `techsigprob` > some threshold, or `sigclasstype = 'Artificial'`, or maybe `sigunique`? Wait, `signalprobabilities` has `techsigprob`. But the question says \"meet our technosignature criteria\". Usually, in these datasets, it's a specific column or condition.\nWait, `signalprobabilities` has `techsigprob`. Maybe `techsigprob > 0.5`? Or maybe `sigclasstype = 'Artificial'`?\nLet's check `signalclassification` columns: `sigclasstype`, `sigpattern`, etc. `sigclasstype` has values 'Artificial', 'Unknown'.\n`signalprobabilities` has `techsigprob`.\nAnother possibility: `signalclassification.sigclasstype = 'Artificial'` is often used for technosignatures.\nBut wait, the question mentions \"TOLS score\" and \"BFR values\".\nI don't see \"TOLS score\" or \"BFR values\" in the schema.\nLet me check the columns in `signals` and other tables.\nMaybe `techsigprob` is the TOLS score? Or `sigunique`?\nWait, `signalprobabilities` has `techsigprob` (numeric(5,4)), `biosigprob`, `natsrcprob`, `artsrcprob`.\n`signalclassification` has `classconf`, `complexidx`, `entropyval`, `infodense`.\n`signaldynamics` has `sigintegrity`, etc.\n`signaldecoding` has `decodeconf`, etc.\nWait, \"TOLS score\" and \"BFR values\" might be aliases for columns or specific columns in the database that I missed?\nLet's search for \"TOLS\" or \"BFR\" in the schema. Not present.\nMaybe it's `techsigprob` and `biosigprob`? \"TOLS\" and \"BFR\" could be abbreviations for something else.\nWait, `signalprobabilities` columns: `techsigprob`, `biosigprob`.\nCould \"TOLS\" stand for Techno-signature Likelihood Score? And \"BFR\" for Bio-signal Frequency Ratio? Or maybe it's `techsigprob` and `biosigprob`?\nLet's look at similar Spider database tasks. This looks like the `astro` or `space` dataset? Actually, the table names are `observationalconditions`, `observatories`, `researchprocess`, `signaladvancedphenomena`, `signalclassification`, `signaldecoding`, `signaldynamics`, `signalprobabilities`, `signals`, `sourceproperties`, `telescopes`.\nThis matches the `space` or `astronomy` dataset from Spider? Actually, it's `signaltech` or `astro`? Wait, it's `astro`? The database is named `asteroids`? No, it's `space` or `signals`?\nActually, this is the `astro` database from Spider? No, it's `astronomy`? Wait, the dataset is `signaltech`? Or `astro`?\nI recall a Spider dataset called `space` or `astro`. The tables here are `observationalconditions`, `observatories`, `researchprocess`, `signaladvancedphenomena`, `signalclassification`, `signaldecoding`, `signaldynamics`, `signalprobabilities`, `signals`, `sourceproperties`, `telescopes`. This matches the dataset `astro` or `signaltech`? Actually, it's `astro` or `astroparticle`? Wait, it's `astro` database in Spider.\n\nIn the `astro` database (or `space`), \"TOLS score\" and \"BFR values\" might refer to `techsigprob` and `biosigprob`? Or `sigunique` and `simindex`?\nLet's think about typical Spider naming conventions: sometimes they ask for `techsigprob` and `biosigprob` as \"TOLS score\" and \"BFR values\". Actually, `techsigprob` stands for Techno-signature probability. `biosigprob` stands for Biological signal probability. But the question says \"average TOLS score, average BFR values\". Could it be `techsigprob` and `biosigprob`?\nWait, another possibility: `sigclasstype` = 'Artificial' is the criteria for technosignatures.\nLet's check `signalclassification.sigclasstype` values: 'Artificial', 'Unknown'.\nIf \"technosignatures\" means `sigclasstype = 'Artificial'`, then we filter by that.\nWhat about \"TOLS score\" and \"BFR values\"? Maybe `techsigprob` and `biosigprob`? Or `sigunique` and `simindex`?\nActually, \"TOLS\" might be `techsigprob` and \"BFR\" might be `biosigprob`? Or maybe `techsigprob` and `natsrcprob`?\nWait, `techsigprob` = Techno-signature probability? \"TOLS\" could be Technical Origin Likelihood Score?\nActually, in some datasets, `techsigprob` and `biosigprob` are used. But \"BFR\" might be \"Bio Frequency Ratio\"? Or maybe it's `sigunique` and `simindex`?\nLet's look at `signalprobabilities` columns: `falseposprob`, `sigunique`, `simindex`, `corrscore`, `anomscore`, `techsigprob`, `biosigprob`, `natsrcprob`, `artsrcprob`.\nCould \"TOLS score\" refer to `techsigprob` and \"BFR values\" refer to `biosigprob`? That seems plausible.\nAnother possibility: \"TOLS\" = `techsigprob`, \"BFR\" = `biosigprob`.\nBut wait, the question says \"average TOLS score, average BFR values\". It might be `techsigprob` and `biosigprob`.\nActually, I've seen this exact question in the Spider database. The database is `astro` or `signaltech`? Wait, the database name in Spider is `astro`? Or `space`? It's `astro`? Actually, it's `astro`? No, it's `space`? Wait, the dataset is called `astro`? Let's search my knowledge for \"Which observatory stations are discovering the most potential technosignatures? For each station, display the observatory name, how many signals meet our technosignature criteria, their average TOLS score, average BFR values, and what percentage of all detected technosignatures they've found.\"\nThis matches the `astro` database in Spider.\nIn `astro`, the table `signalprobabilities` has `techsigprob` and `biosigprob`.\n\"technosignature criteria\" usually means `sigclasstype = 'Artificial'` or `techsigprob > 0.5`? But in Spider, they often use `sigclasstype = 'Artificial'` for technosignatures. Actually, `signalclassification.sigclasstype = 'Artificial'` is common.\nWait, let's check `signalclassification` table: `sigclasstype` has values like 'Artificial', 'Unknown'.\nAnother possibility: `techsigprob > 0.5`? But \"meet our technosignature criteria\" might just be a phrase for `sigclasstype = 'Artificial'`.\nActually, in this dataset, `sigclasstype = 'Artificial'` is used to define technosignatures. I'll assume `signalclassification.sigclasstype = 'Artificial'`.\nWait, the question says \"average TOLS score, average BFR values\". TOLS might be `techsigprob`. BFR might be `biosigprob`.\nLet's verify with column names: `techsigprob` and `biosigprob`. Yes, that fits.\nPercentage of all detected technosignatures: `COUNT(signalref) * 100.0 / SUM(COUNT(signalref)) OVER ()` or just `COUNT(signalref) * 100.0 / (SELECT COUNT(signalref) FROM signals JOIN signalclassification ... WHERE sigclasstype = 'Artificial')`. But window functions are easier: `COUNT(signalref) * 100.0 / SUM(COUNT(signalref)) OVER ()`.\nWait, Spider often uses `COUNT(signalregistry) * 100.0 / (SELECT COUNT(signalregistry) FROM signals JOIN signalclassification ON signals.signalregistry = signalclassification.signalref WHERE signalclassification.sigclasstype = 'Artificial')`.\nOr they might just ask for `COUNT(signals.signalregistry) * 100.0 / SUM(COUNT(signals.signalregistry)) OVER ()`. But window functions are supported in SQLite.\n\nLet's check the exact phrasing from Spider's `astro` dataset if possible.\nDatabase: `astro`\nTables: `observationalconditions`, `observatories`, `researchprocess`, `signaladvancedphenomena`, `signalclassification`, `signaldecoding`, `signaldynamics`, `signalprobabilities`, `signals`, `sourceproperties`, `telescopes`.\nQuestion: \"Which observatory stations are discovering the most potential technosignatures? For each station, display the observatory name, how many signals meet our technosignature criteria, their average TOLS score, average BFR values, and what percentage of all detected technosignatures they've found. I need this ranked by the stations with the most discoveries first.\"\n\nColumns to output:\n1. observatory name -> `observatories.observstation`\n2. number of signals -> `COUNT(signals.signalregistry)`\n3. average TOLS score -> `AVG(signalprobabilities.techsigprob)`\n4. average BFR values -> `AVG(signalprobabilities.biosigprob)`\n5. percentage -> `COUNT(signals.signalregistry) * 100.0 / (SELECT COUNT(signalregistry) FROM signals JOIN signalclassification ON signals.signalregistry = signalclassification.signalref WHERE sigclasstype = 'Artificial')` or using window function.\n\nWait, \"TOLS score\" might be `techsigprob` and \"BFR values\" might be `biosigprob`. I'll stick with that.\nLet's verify column names in `signalprobabilities`: `techsigprob`, `biosigprob`.\nWhat about \"technosignature criteria\"? In `signalclassification`, `sigclasstype` = 'Artificial'.\nSo the join condition: `signals.signalregistry = signalclassification.signalref` and `signals.signalregistry = signalprobabilities.signalref` and `signals.telescref = telescopes.telescregistry` and `telescopes.observstation = observatories.observstation`.\nFilter: `signalclassification.sigclasstype = 'Artificial'`.\n\nLet's construct the query:\n```sql\nSELECT\n  T4.observstation,\n  COUNT(T1.signalregistry),\n  AVG(T3.techsigprob),\n  AVG(T3.biosigprob),\n  COUNT(T1.signalregistry) * 100.0 / (SELECT COUNT(T1_1.signalregistry) FROM signals AS T1_1 JOIN signalclassification AS T2_1 ON T1_1.signalregistry  =  T2_1.signalref WHERE T2_1.sigclasstype = 'Artificial')\nFROM signals AS T1\nJOIN signalclassification AS T2 ON T1.signalregistry  =  T2.signalref\nJOIN signalprobabilities AS T3 ON T1.signalregistry  =  T3.signalref\nJOIN telescopes AS T4 ON T1.telescref  =  T4.telescregistry\nJOIN observatories AS T5 ON T4.observstation  =  T5.observstation\nWHERE T2.sigclasstype = 'Artificial'\nGROUP BY T4.observstation\nORDER BY COUNT(T1.signalregistry) DESC\n```\n\nWait, `observatories` table has `observstation` as primary key. `telescopes` also has `observstation`. So `telescopes.observstation = observatories.observstation` is correct.\nBut `observatories` table is named `observatories`, column `observstation`.\nThe question asks for \"observatory name\", which is `observstation`.\n\nCheck if `signalclassification.sigclasstype` has value 'Artificial'. The sample data shows `sigclasstype` as 'Artificial', 'Unknown'. So 'Artificial' is correct.\n\nWhat about \"TOLS score\" and \"BFR values\"? I'll assume `techsigprob` and `biosigprob`.\nIs there any chance \"TOLS\" stands for `techsigprob` and \"BFR\" stands for `biosigprob`? Yes, in this dataset, `techsigprob` is Technological Origin Likelihood Score? Actually, it's just `techsigprob`. `biosigprob` is Biological Frequency Ratio? Maybe.\nI will use `AVG(T3.techsigprob)` and `AVG(T3.biosigprob)`.\n\nPercentage calculation: `COUNT(T1.signalregistry) * 100.0 / SUM(COUNT(T1.signalregistry)) OVER ()` is simpler and works in SQLite 3.25+. Spider datasets usually support window functions, but sometimes they prefer the subquery method. I'll use the subquery to be safe, or `COUNT(T1.signalregistry) * 100.0 / (SELECT count(signalregistry) FROM signals JOIN signalclassification ON signals.signalregistry = signalclassification.signalref WHERE signalclassification.sigclasstype = 'Artificial')`. Actually, `COUNT(T1.signalregistry) * 100.0 / sum(COUNT(T1.signalregistry)) OVER()` is fine. But I'll stick to the subquery for compatibility with older SQLite versions if needed, though Spider usually accepts window functions. I'll use `COUNT(T1.signalregistry) * 100.0 / (SELECT count(T1.signalregistry) FROM signals AS T1 JOIN signalclassification AS T2 ON T1.signalregistry  =  T2.signalref WHERE T2.sigclasstype  =  'Artificial')`.\n\nWait, `COUNT(T1.signalregistry)` in the SELECT list will be an alias? Better to use `COUNT(signals.signalregistry)`.\nLet's check the exact column names: `signals.signalregistry`, `signalclassification.signalref`, `signalprobabilities.signalref`, `telescopes.telescregistry`, `observatories.observstation`.\nAll good.\n\nOne detail: \"percentage of all detected technosignatures they've found\".\nShould it be `COUNT(T1.signalregistry) * 100.0 / (SELECT count(signalregistry) FROM signals JOIN signalclassification ON signals.signalregistry = signalclassification.signalref WHERE signalclassification.sigclasstype = 'Artificial')`? Yes.\nOr maybe `SUM(COUNT(signalregistry)) OVER()`? Both work.\n\nLet's verify the table aliases and joins:\n`signals` (T1)\n`signalclassification` (T2) on T1.signalregistry = T2.signalref\n`signalprobabilities` (T3) on T1.signalregistry = T3.signalref\n`telescopes` (T4) on T1.telescref = T4.telescregistry\n`observatories` (T5) on T4.observstation = T5.observstation\n\nWait, `telescopes` has `observstation` column. `observatories` has `observstation` column.\nSo `T4.observstation = T5.observstation` is correct.\n\nThe query:\n```sql\nSELECT\n  T5.observstation,\n  COUNT(T1.signalregistry),\n  AVG(T3.techsigprob),\n  AVG(T3.biosigprob),\n  COUNT(T1.signalregistry) * 100.0 / (SELECT count(signalregistry) FROM signals JOIN signalclassification ON signals.signalregistry  =  signalclassification.signalref WHERE signalclassification.sigclasstype = 'Artificial')\nFROM signals AS T1\nJOIN signalclassification AS T2 ON T1.signalregistry  =  T2.signalref\nJOIN signalprobabilities AS T3 ON T1.signalregistry  =  T3.signalref\nJOIN telescopes AS T4 ON T1.telescref  =  T4.telescregistry\nJOIN observatories AS T5 ON T4.observstation  =  T5.observstation\nWHERE T2.sigclasstype = 'Artificial'\nGROUP BY T5.observstation\nORDER BY COUNT(T1.signalregistry) DESC\n```\n\nIs `TOLS` definitely `techsigprob`? And `BFR` definitely `biosigprob`?\nIn the `astro` dataset, `techsigprob` stands for \"Techno-signal probability\" and `biosigprob` stands for \"Bio-signal probability\". The question says \"TOLS score\" and \"BFR values\". It's highly likely they map to `techsigprob` and `biosigprob`.\nAnother possibility: `sigunique` and `simindex`? But \"TOLS\" sounds like Techno Origin Likelihood Score, which matches `techsigprob`. \"BFR\" might be Bio Frequency Ratio, which matches `biosigprob`. I'll stick with that.\n\nWait, what about the percentage column? Sometimes Spider expects `COUNT(T1.signalregistry) * 100.0 / SUM(COUNT(T1.signalregistry)) OVER ()`. But the subquery is safer. I'll use `COUNT(T1.signalregistry) * 100.0 / (SELECT count(T1.signalregistry) FROM signals AS T1 JOIN signalclassification AS T2 ON T1.signalregistry  =  T2.signalref WHERE T2.sigclasstype = 'Artificial')`. Actually, in Spider, they often use `COUNT(T1.signalregistry) * 100.0 /"
    """.strip()
    answer = """
    SELECT\n  T5.observstation,\n  COUNT(T1.signalregistry),\n  AVG(T3.techsigprob),\n  AVG(T3.biosigprob),\n  COUNT(T1.signalregistry) * 100.0 / (SELECT count(signalregistry) FROM signals JOIN signalclassification ON signals.signalregistry  =  signalclassification.signalref WHERE signalclassification.sigclasstype = 'Artificial')\nFROM signals AS T1\nJOIN signalclassification AS T2 ON T1.signalregistry  =  T2.signalref\nJOIN signalprobabilities AS T3 ON T1.signalregistry  =  T3.signalref\nJOIN telescopes AS T4 ON T1.telescref  =  T4.telescregistry\nJOIN observatories AS T5 ON T4.observstation  =  T5.observstation\nWHERE T2.sigclasstype = 'Artificial'\nGROUP BY T5.observstation\nORDER BY COUNT(T1.signalregistry) DESC
    """.strip()
    # assert extract_sql_from_response(text) == answer
    print(extract_sql_from_response(text))
