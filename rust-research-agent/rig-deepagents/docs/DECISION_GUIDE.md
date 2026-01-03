# rig-deepagents 개발 의사결정 가이드

> **TL;DR**: Rig가 이미 제공하는 기능은 사용하고, rig-deepagents 고유 기능에 집중합니다.

---

## Quick Reference: 무엇을 사용할까?

### ✅ Rig 사용 (중복 개발 금지)

| 기능 | Rig 코드 | 사용 방법 |
|------|----------|-----------|
| **LLM 호출** | `rig::providers::*` | `Client::from_env()` |
| **ThinkTool** | `rig::tools::think::ThinkTool` | `.tool(ThinkTool)` |
| **Streaming** | `rig::streaming` | `.stream_prompt()` |
| **Vector Store** | `rig-lancedb`, `rig-qdrant` 등 | 별도 crate |
| **멀티턴 대화** | `PromptRequest` | `.multi_turn(5)` |
| **도구 콜백** | `PromptHook` | `.with_hook()` |

### ✅ rig-deepagents 사용 (고유 기능)

| 기능 | 위치 | 용도 |
|------|------|------|
| **AgentMiddleware** | `middleware/` | 도구 주입, 프롬프트 수정 |
| **Pregel Runtime** | `pregel/` | 그래프 워크플로우 실행 |
| **Checkpointing** | `pregel/checkpoint/` | SQLite/Redis/Postgres 상태 저장 |
| **Backend** | `backends/` | 파일시스템/메모리 추상화 |
| **SubAgent** | `middleware/subagent.rs` | 태스크 위임 |
| **Human-in-the-Loop** | `middleware/human_in_loop.rs` | 승인 흐름 |
| **SummarizationMiddleware** | `middleware/summarization.rs` | 토큰 예산 관리 |
| **SkillsMiddleware** | `skills/` | 스킬 점진적 공개 |
| **TavilySearchTool** | `tools/tavily.rs` | 웹 검색 (Rig에 없음) |

---

## 체크리스트: 새 기능 개발 전

```
□ Rig docs.rig.rs 에서 해당 기능 검색했는가?
□ Rig GitHub examples 에서 유사 패턴 확인했는가?
□ rig-core 소스에서 trait/impl 확인했는가?
```

**위 체크리스트를 모두 확인 후에도 Rig에 없다면** → rig-deepagents에 구현

---

## 올바른 LLM 사용법

### ✅ Rig Agent를 RigAgentAdapter로 래핑

```rust
use rig::client::{CompletionClient, ProviderClient};
use rig_deepagents::RigAgentAdapter;

// Rig Client 생성
let client = rig::providers::openai::Client::from_env();
let agent = client.agent("gpt-4").build();

// RigAgentAdapter로 래핑하여 LLMProvider로 사용
let provider = Arc::new(RigAgentAdapter::new(agent));
```

> **Note**: 레거시 `OpenAIProvider`와 `AnthropicProvider`는 제거되었습니다.
> 대신 `RigAgentAdapter`를 사용하여 Rig의 네이티브 프로바이더를 래핑하세요.

---

### ❌ 잘못된 예

```rust
// rig-deepagents의 ThinkTool 사용
use rig_deepagents::tools::ThinkTool;
```

### ✅ 올바른 예

```rust
// Rig 내장 ThinkTool 사용
use rig::tools::think::ThinkTool;
```

---

## Tool 개발 기준

### Rig Tool 사용 (단순 도구)

- 상태(ToolRuntime) 접근 불필요
- 단순 입력 → 출력 변환
- 타입 안전성 중요

### rig-deepagents Tool 사용 (상태 필요)

- 파일시스템 접근 (Backend)
- 에이전트 상태 읽기/쓰기
- 다른 도구와 상호작용
- 동적 스키마 필요

---

## 참조 문서

- [RIG_FRAMEWORK_REFERENCE.md](./RIG_FRAMEWORK_REFERENCE.md) - 상세 기능 비교
- [Rig Docs](https://docs.rig.rs) - 공식 문서
- [Rig Examples](https://github.com/0xPlaygrounds/rig/tree/main/rig-core/examples)
