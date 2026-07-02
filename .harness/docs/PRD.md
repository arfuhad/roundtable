- **Task Status:** A background task has been started to locate the codebase files (`harness.config.yaml`) and check the environment. I will wait for it to complete.
I have reverse-engineered the Product Requirements Document (PRD) for the **llm-harness** project based on a deep analysis of its codebase files (located in `/Users/arfuhad/Development/personal/harness`). 

The complete, detailed PRD has been written as an artifact in your workspace:
*   [reverse_engineered_prd.md](file:///Users/arfuhad/.gemini/antigravity-cli/brain/9a4050e4-4e03-4ec1-a17c-dd4ef36b52b0/reverse_engineered_prd.md)

### Key PRD Structural Highlights

1.  **Summary:** Defines `llm-harness` as a local-first multi-LLM orchestration framework that coordinates external CLI LLM tools (Claude Code, aider, Ollama, Codex, Gemini CLI) or APIs to execute structured engineering workflows.
2.  **Problem Statement & Goals:** Outlines issues with monolithic agent context bloat, lack of model diversity, and unsafe agent control loops, and details the "context-cleaning invariant" design solution.
3.  **Current Features & Capabilities:** Maps each system feature to its concrete codebase files, including:
    *   Workspace Scaffolding & Agent Discovery ([store.py](file:///Users/arfuhad/Development/personal/harness/src/harness/store.py) / [discovery.py](file:///Users/arfuhad/Development/personal/harness/src/harness/discovery.py))
    *   Codebase Mapping ([scan.py](file:///Users/arfuhad/Development/personal/harness/src/harness/scan.py) / [agents.py](file:///Users/arfuhad/Development/personal/harness/src/harness/agents.py))
    *   DAG-based planning ([models.py](file:///Users/arfuhad/Development/personal/harness/src/harness/models.py))
    *   Human Approval Gate & Sequential Engine Loop ([engine.py](file:///Users/arfuhad/Development/personal/harness/src/harness/engine.py))
    *   Zero-Dependency Dashboard Telemetry ([dashboard.py](file:///Users/arfuhad/Development/personal/harness/src/harness/dashboard.py) / [insights.py](file:///Users/arfuhad/Development/personal/harness/src/harness/insights.py))
4.  **Non-functional Constraints:** Includes constraints on Python 3.11+, zero external frontend dependencies, local state containment in `.harness/`, and subprocess argv lists for command-injection safety.
5.  **Out of Scope:** Declares subprocess isolation/sandboxing, VCS commits/rollbacks, and inter-phase dependencies as out of scope.
6.  **Open Questions & Assumptions:** Flags critical implementation details that need review, such as the engine's behavior of marking failed/error-returning tasks as done and the lack of orphaned process cleanup when a run is canceled.
