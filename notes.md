# Notes: LangChain DeepAgents vs rig-deepagents (Full Audit)

## Sources

### Source 1: Python DeepAgents (LangChain)
- Path: `deepagents_sourcecode/libs/deepagents/deepagents/graph.py`
- Key points:
  - `create_deep_agent` wires default middleware: `TodoListMiddleware`, `FilesystemMiddleware`, `SubAgentMiddleware`, `SummarizationMiddleware`, `AnthropicPromptCachingMiddleware`, `PatchToolCallsMiddleware`; adds `HumanInTheLoopMiddleware` when `interrupt_on` is set.
  - Filesystem toolset includes `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `execute`.
  - Subagents use default middleware/tools and can attach `interrupt_on`; includes a general-purpose subagent.
  - `create_agent(...).with_config({"recursion_limit": 1000})`.

### Source 2: Python DeepAgents Filesystem Middleware
- Path: `deepagents_sourcecode/libs/deepagents/deepagents/middleware/filesystem.py`
- Key points:
  - Tools update state via `Command(update={...})` when backend returns `files_update` (write/edit).
  - `execute` tool added only if backend supports `SandboxBackendProtocol`.
  - Path validation normalizes and rejects traversal; Windows absolute paths are rejected.
  - Large tool result eviction into filesystem via `tool_token_limit_before_evict`.
  - Model call wrapping removes unsupported execute tool and injects dynamic system prompt.

### Source 3: Python DeepAgents Subagents
- Path: `deepagents_sourcecode/libs/deepagents/deepagents/middleware/subagents.py`
- Key points:
  - `SubAgent` schema supports `interrupt_on` and `default_middleware`/`default_interrupt_on`.
  - `task` tool returns `Command` with state updates (excludes messages/todos/structured_response).
  - Requires `ToolRuntime.tool_call_id` for returning updates.
  - Parallel subagent calls are supported via multiple tool_calls in a single message.

### Source 4: Python DeepAgents Backends
- Paths: `deepagents_sourcecode/libs/deepagents/deepagents/backends/*`
- Key points:
  - `StoreBackend` supports LangGraph `BaseStore` (long-term memory across threads).
  - `SandboxBackendProtocol` provides `execute` capability for shell commands; `BaseSandbox` helper exists.
  - `BackendFactory` allows runtime-based backend selection.
  - `grep` uses regex (`re.compile`) in utils.
  - `StateBackend` returns `files_update` and relies on LangGraph `Command` updates (state is not mutated directly).

### Source 5: Python DeepAgents Tests (Behavioral Spec)
- Paths: `deepagents_sourcecode/libs/deepagents/tests/*`
- Key points:
  - `create_deep_agent` supports `response_format` (ToolStrategy) and `context_schema`.
  - `FilesystemMiddleware` exposes 7 tools incl. execute, and evicts large tool results into files.
  - `StateBackend` supports regex grep errors; `CompositeBackend` routing tested.
  - `SubAgent` returns ToolMessage with final response and merges state updates.
  - Checkpointer usage (`InMemorySaver`) used with `create_deep_agent` invocations.

### Source 6: deepagents-cli (Adjacent Capabilities)
- Paths: `deepagents_sourcecode/libs/deepagents-cli/*`
- Key points:
  - Adds `shell`, `web_search`, `fetch_url` tools, memory middleware, skills middleware, auto-approve HITL.
  - Persistent memory stored via `.deepagents/agent.md` and project `.deepagents/` files.
  - Skills system supports project/user-level skills with progressive disclosure.

### Source 7: Rust rig-deepagents
- Paths: `rust-research-agent/crates/rig-deepagents/src/*`
- Key points:
  - No `create_deep_agent` factory; uses `AgentExecutor` + `MiddlewareStack` + explicit tool injection.
  - Tools available: `read_file`, `write_file`, `edit_file`, `ls`, `glob`, `grep`, `write_todos`, `task`, `think`, `tavily_search`. No `execute` tool.
  - `Backend` trait lacks `execute`; includes `exists`/`delete` (extra vs Python).
  - Tools return `String` only; no mechanism to return `Command`-like state updates.
  - `ToolRuntime` has `tool_call_id` but executor does not set it per tool call.
  - Subagent config includes `default_middleware` but is not applied in `SubAgentMiddleware::new`.
  - `SubAgentSpec` lacks `interrupt_on` and `default_interrupt_on` handling.
  - `SummarizationMiddleware`, `PatchToolCallsMiddleware`, `HumanInTheLoopMiddleware` exist but are not wired as defaults.
  - `grep` is literal (non-regex) by design; path validation does not reject Windows absolute paths.
  - Rust includes extra systems absent in Python: Pregel runtime/checkpointer, workflow graph, skills middleware, research workflows.

### Source 8: Rust Pregel Workflow System
- Paths: `rust-research-agent/crates/rig-deepagents/src/pregel/*`, `src/workflow/*`
- Key points:
  - Pregel runtime is custom; `WorkflowState` uses a single update type (no per-key reducers).
  - Agent execution in Pregel uses `AgentVertex` with tool registry; tools are executed via `MemoryBackend` and do not mutate shared `AgentState`.
  - Workflow vertices do not integrate middleware stack or file/todo updates.
  - SubAgentVertex uses empty isolated state and `superstep` as recursion proxy.
  - AgentExecutor (non-Pregel) is the primary DeepAgents path today.

## Synthesized Findings

### Parity Gaps (Python has, Rust missing or incomplete)
- `create_deep_agent` high-level factory with default middleware + base prompt + recursion_limit.
- Filesystem middleware behavior: toolset auto-injection + state updates on write/edit + tool-result eviction.
- Planning middleware (`TodoListMiddleware`) behavior: `write_todos` updates state.
- `execute` tool + sandbox backend protocol.
- Subagent default middleware/interrupt handling and state update propagation back to parent.
- `StoreBackend` (LangGraph store) and backend factory pattern.
- `ToolRuntime.tool_call_id` propagation to tools.
- Regex-capable grep (Rust is literal only).
- Structured output wiring (`response_format`, `context_schema`) and cache hooks in `create_deep_agent`.
- Streaming + checkpointer integration in main agent path.
- Parallel tool-call execution semantics (Python allows parallel; Rust executor is sequential).

### Pregel-Model Differences (Execution Layer)
- LangChain DeepAgents runs on LangGraph (Pregel-style) with Command state updates per tool.
- Rust pregel exists but deepagent logic uses a custom `AgentExecutor`, not Pregel.
- Rust Pregel workflow graph does not use the same middleware stack or filesystem/todo state reducers.
 - LangGraph uses per-key reducers (e.g., files reducer supports deletion via `None`); Rust `WorkflowState` lacks reducer-per-field semantics.
 - LangGraph tool execution can update state via `Command(update=...)`; Rust tool execution returns `String` only.
 - LangGraph checkpointer and stream channels are part of the main agent path; Rust Pregel checkpointer exists but is not used by `AgentExecutor`.

### Divergences/Extras (Rust-only)
- Pregel workflow runtime + checkpointing backends (SQLite/Redis/Postgres).
- Skills middleware (progressive disclosure) included in Rust core.
- Research workflow builders (planner/search/synthesizer) in Rust.
- Backend supports `exists`/`delete` (not exposed as tools).

### Divergences/Extras (Python CLI-only)
- Shell tool, web search, fetch_url.
- Agent memory middleware (user/project memory).
- Skills system at user/project level in CLI distribution.
