# Rig DeepAgents Parity Plan (LangChain DeepAgents vs rig-deepagents)

## Differences Summary (Current Gaps)

1. **No high-level `create_deep_agent` factory in Rust**
   - Python wires default middleware, base prompt, and `recursion_limit=1000`.

2. **Filesystem + planning state updates missing**
   - Python tools (`write_file`, `edit_file`, `write_todos`, `task`) can return `Command` with state updates.
   - Rust tools return `String` only; `AgentState.todos` / `AgentState.files` are not updated.

3. **No `execute` tool + sandbox backend**
   - Python exposes `execute` when backend supports `SandboxBackendProtocol`.
   - Rust `Backend` has no `execute` equivalent.

4. **Subagent parity gaps**
   - Default middleware is not applied in `SubAgentMiddleware::new`.
   - `interrupt_on` and `default_interrupt_on` are not supported.
   - Subagent file updates are not merged back into parent state.

5. **Backend factory + Store backend missing**
   - Python supports `BackendFactory` and `StoreBackend` (LangGraph BaseStore).
   - Rust only has static backends (memory/filesystem/composite).

6. **Tool runtime metadata not propagated**
   - `ToolRuntime.tool_call_id` exists but is not set per tool call.

7. **Grep semantics differ**
   - Python uses regex; Rust uses literal search only.

8. **Structured output + cache hooks not wired**
   - Python supports `response_format`, `context_schema`, `cache` in `create_deep_agent`.

9. **Tool-result eviction / streaming / checkpointer integration missing**
   - Python evicts large tool outputs to filesystem and supports `checkpointer` + stream channels.

10. **Parallel tool-call execution semantics differ**
   - Python can execute multiple tool_calls in a single step; Rust executor is sequential.

11. **Pregel execution integration mismatch**
   - Python DeepAgents runs on LangGraph (Pregel-style) with Command state updates;
     Rust Pregel exists but DeepAgents logic uses `AgentExecutor` and does not share middleware/state reducers.

## Scope Decision (Required vs Optional)

### Required (Core Parity)
- **P0** ToolResult + state update pipeline (Phase 1).
- **P0** Filesystem + planning middleware parity (Phase 2).
- **P1** Subagent parity (Phase 3).
- **P1** High-level factory API (Phase 5).

### Optional (Nice-to-have / Riskier)
- **P2** Sandbox execution + `execute` tool (Phase 4).
- **P2** Pregel alignment bridge (Phase 7).
- **P3** Backend factory + Store backend (Gap 5).
- **P3** Regex grep or dual-mode grep (Gap 7).
- **P3** Structured output + cache hooks (Gap 8).

## Pregel-First Implementation Principle (Now)

- Implement P0/P1 features so they can be shared by **both** the current `AgentExecutor`
  and the Pregel workflow vertices, even before Phase 7.
- Favor **immutable state updates** (`StateUpdate` / `ToolResult` → reducer-style merge) over in-place mutation.
- Keep tool execution **Command-like** (message + state updates) to preserve LangGraph parity.

## Implementation Plan (Rig-based)

### Phase 1 (P0, Required): Add ToolResult + State Update Pipeline

- **Goal:** Allow tools to return both messages and state updates.
- **Changes:**
  - Add a `ToolResult` struct (e.g., `{ message: String, updates: Vec<StateUpdate> }`).
  - Update `Tool` trait to return `ToolResult` (or `Result<ToolResult, MiddlewareError>`).
  - Update `AgentExecutor::execute_tool_call` to:
    - set `ToolRuntime` with `tool_call_id`,
    - apply `StateUpdate`s to `AgentState`,
    - add `Message::tool` with the returned message.
  - Define `ToolResult` so it can be consumed by Pregel vertices later (e.g., a helper to convert updates into `WorkflowMessage` or workflow `StateUpdate`).
- **Targets:**
  - `rust-research-agent/crates/rig-deepagents/src/middleware/traits.rs`
  - `rust-research-agent/crates/rig-deepagents/src/executor.rs`
  - `rust-research-agent/crates/rig-deepagents/src/runtime.rs`
- **Tests:**
  - Tool result update flow updates `AgentState.todos` and `AgentState.files`.

### Phase 2 (P0, Required): Filesystem + Planning Middleware Parity

- **Goal:** Mirror Python `TodoListMiddleware` + `FilesystemMiddleware` behaviors.
- **Changes:**
  - Add a `TodoListMiddleware` (or equivalent) that injects `write_todos` tool + system prompt guidance.
  - Update `WriteTodosTool` to return `StateUpdate::SetTodos` via `ToolResult`.
  - Add a `FilesystemMiddleware` that injects file tools and prompt guidance.
  - Update `WriteFileTool` / `EditFileTool` to return `StateUpdate::UpdateFiles` when backend returns `files_update`.
  - Ensure file updates support reducer-style merges (future Pregel alignment).
- **Targets:**
  - `rust-research-agent/crates/rig-deepagents/src/middleware/`
  - `rust-research-agent/crates/rig-deepagents/src/tools/write_todos.rs`
  - `rust-research-agent/crates/rig-deepagents/src/tools/write_file.rs`
  - `rust-research-agent/crates/rig-deepagents/src/tools/edit_file.rs`
- **Tests:**
  - `write_todos` updates state.
  - `write_file` / `edit_file` updates state with `MemoryBackend`.

### Phase 3 (P1, Required): Subagent Parity (Middleware + Updates)

- **Goal:** Align subagent behavior with LangChain DeepAgents.
- **Changes:**
  - Apply `default_middleware` in `SubAgentMiddleware::new` (pass into `SubAgentExecutorConfig`).
  - Add `interrupt_on` to `SubAgentSpec` and plumb through execution.
  - Extend `SubAgentMiddlewareConfig` with `default_interrupt_on` and apply `HumanInTheLoopMiddleware` to subagents.
  - Update `TaskTool` to return state updates (merge subagent files into parent using `IsolatedState::merge_files`).
- **Targets:**
  - `rust-research-agent/crates/rig-deepagents/src/middleware/subagent/*`
- **Tests:**
  - Subagent file updates propagate to parent.
  - `interrupt_on` triggers `HumanInTheLoopMiddleware` for subagent tools.

### Phase 4 (P2, Optional): Sandbox Execution + Execute Tool

- **Goal:** Provide `execute` tool parity and safe shell execution.
- **Changes:**
  - Add a new trait (e.g., `ExecutableBackend`) or extend `Backend` with `execute`.
  - Implement a `SandboxBackend` (local command runner with allowlist/root restriction).
  - Implement `ExecuteTool` and include it in default toolset when backend supports execution.
- **Targets:**
  - `rust-research-agent/crates/rig-deepagents/src/backends/`
  - `rust-research-agent/crates/rig-deepagents/src/tools/execute.rs`
  - `rust-research-agent/crates/rig-deepagents/src/tools/mod.rs`
- **Tests:**
  - `execute` tool only available when backend supports execution.

### Phase 5 (P1, Required): High-Level Factory API

- **Goal:** Provide `create_deep_agent` style factory using Rig.
- **Changes:**
  - Add `create_deep_agent` or `DeepAgentBuilder` that wires:
    - default middleware stack (TodoList, Filesystem, SubAgent, Summarization, PatchToolCalls, optional HumanInTheLoop),
    - base system prompt (DeepAgents-style),
    - recursion limit (match 1000),
    - optional tools/subagents/interrupt_on.
  - Use `RigAgentAdapter` for LLM and `RigToolAdapter` for Rig tools.
- **Targets:**
  - `rust-research-agent/crates/rig-deepagents/src/lib.rs`
  - `rust-research-agent/crates/rig-deepagents/src/config.rs` or new builder module.
- **Tests:**
  - Factory creates an executor with expected default tools and prompts.

### Phase 6 (P3, Optional): Parity Extensions

- **Regex grep:** add `grep_regex` tool or a config flag to switch regex vs literal.
- **Store backend:** implement a `StoreBackend` using a persistent KV store (e.g., sqlite) and allow backend factory hooks.
- **Structured output:** map to Rig structured outputs if available, or add middleware to validate JSON responses.
- **Prompt caching:** optional middleware if Rig provider exposes caching semantics.
- **Tool-result eviction:** add large tool result interception and file-backed storage.
- **Streaming/checkpointer:** add stream channels and checkpointer integration for agent execution.
- **Parallel tool calls:** enable concurrent tool execution when multiple tool_calls are returned.

#### ToolResult-based large tool result eviction (design draft)

- **Goal:** prevent oversized `ToolResult.message` strings from bloating context by evicting to `/large_tool_results/{tool_call_id}`.
- **Trigger:** if `result.message.len() > 4 * token_limit` (default 20k token ≈ 80k chars); skip for filesystem tools to avoid recursion.
- **Flow:** after tool execution, if oversized → `backend.write(evict_path, result.message)` → replace message with a short notice + first 10 lines sample; attach `StateUpdate::UpdateFiles` when backend returns `files_update`.
- **Integration point:** add a `ToolResultInterceptor` hook in `AgentExecutor` (or new middleware hook) that transforms `ToolResult` before `Message::tool` is added.
- **Open questions:** path sanitization for `tool_call_id`, token estimation source, and whether to expose config on `FilesystemMiddleware` vs executor.

### Phase 7 (P2, Optional): Pregel Alignment

- **Goal:** Align DeepAgents execution with the Pregel workflow runtime.
- **Changes:**
  - Add a `create_deep_agent_graph` or bridge that builds a `WorkflowGraph` with Agent/Tool/SubAgent vertices.
  - Port middleware behaviors (filesystem/todo/task updates) into Pregel vertices or a shared ToolResult pipeline.
  - Ensure state updates follow LangGraph-style Command semantics.
- **Targets:**
  - `rust-research-agent/crates/rig-deepagents/src/workflow/*`
  - `rust-research-agent/crates/rig-deepagents/src/pregel/*`

## Suggested Sequencing

1. Phase 1 (P0) - unlocks state update pipeline (foundation).
2. Phase 2 (P0) - filesystem + planning parity on top of state updates.
3. Phase 3 (P1) - subagent parity with update propagation.
4. Phase 5 (P1) - high-level factory API for ergonomics.
5. Phase 4 (P2) - sandbox execute tool (optional).
6. Phase 6 (P3) - optional parity extensions.
