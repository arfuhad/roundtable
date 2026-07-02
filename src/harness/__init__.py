"""llm-harness: a multi-LLM planning and orchestration harness.

Flow: Planner LLM -> approved Plan (phases -> tasks) -> Main
Orchestrator drives phases; each Phase Orchestrator (fresh context) dispatches
Task Agents, summarizes, and reports to Main, which maintains the docs.

Every role's model is choosable via config or the plan manifest.
"""

__version__ = "0.3.0"
