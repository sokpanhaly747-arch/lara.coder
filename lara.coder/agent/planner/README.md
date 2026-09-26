# agent/planner/

Agent-side planning glue. Today planning happens in the top-level planning/ package; this folder is the agent's integration point for invoking it (task selection, replanning triggers) as that logic grows beyond orchestrator.py.
