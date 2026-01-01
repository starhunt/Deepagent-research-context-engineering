# PROJECT.md - Rust Research Agent

이 문서는 rust-research-agent 프로젝트의 구조와 아키텍처에 대한 종합적인 가이드입니다.

## 프로젝트 개요

**rust-research-agent**는 **Rig 프레임워크**를 사용한 Rust AI 연구 에이전트입니다.  
Ollama 로컬 LLM과 DuckDuckGo 검색을 결합하여 웹 연구를 수행하고 결과를 합성합니다.

### 핵심 특징

- **완전 로컬 실행**: API 키 없이 Ollama + DuckDuckGo로 동작
- **프라이버시 중심**: 모든 처리가 로컬에서 수행
- **교육용 설계**: Rust 초보자를 위한 상세한 주석 포함
- **프로덕션 수준**: 적절한 에러 처리, 로깅, 검증 포함

---

## 프로젝트 구조

```
rust-research-agent/
├── Cargo.toml          # 프로젝트 메타데이터 및 의존성
├── README.md           # 프로젝트 문서
├── LICENSE             # 라이선스 파일
├── .env.example        # 환경 변수 템플릿
├── PROJECT.md          # 이 문서
└── src/
    ├── main.rs         # CLI 진입점, 인자 파싱, 오케스트레이션
    ├── config.rs       # 설정 관리 및 환경 변수 로딩
    ├── agent.rs        # 연구 에이전트 구현 (Rig 통합)
    └── tools.rs        # 웹 검색 도구 (DuckDuckGo 통합)
```

### 모듈 의존성

```
main.rs ─┬─> config.rs
         ├─> agent.rs ──> config.rs, tools.rs
         └─> tools.rs (standalone)
```

---

## 핵심 의존성

| 패키지 | 버전 | 역할 | 기능 플래그 |
|--------|------|------|-------------|
| **rig-core** | 0.27 | LLM 에이전트 프레임워크 핵심 | `derive` |
| **tokio** | 1.x | 비동기 런타임 | `full` |
| **serde/serde_json** | 1.x | JSON 직렬화/역직렬화 | `derive` |
| **reqwest** | 0.12 | HTTP 클라이언트 | `json` |
| **duckduckgo_search** | 0.1 | DuckDuckGo 검색 통합 | - |
| **clap** | 4.x | CLI 인자 파싱 | `derive`, `env` |
| **anyhow** | 1.x | 애플리케이션 레벨 에러 처리 | - |
| **thiserror** | 2.x | 커스텀 에러 타입 | - |
| **tracing** | 0.1 | 구조화된 로깅 | - |
| **tracing-subscriber** | 0.3 | 로깅 구독자 | `env-filter` |
| **dotenvy** | 0.15 | .env 파일 로딩 | - |
| **async-trait** | 0.1 | 트레이트의 async 메서드 | - |
| **futures** | 0.3 | 추가 async 유틸리티 | - |
| **urlencoding** | 2.1 | URL 인코딩 | - |

---

## 모듈별 상세 구조

### 1. config.rs - 설정 관리

```rust
pub struct Config {
    pub model: String,              // Ollama 모델 (예: "llama3.2")
    pub ollama_host: String,        // Ollama 서버 URL
    pub temperature: f32,           // LLM 창의성 (0.0-2.0)
    pub max_search_results: usize,  // 분석할 검색 결과 수
    pub log_level: String,          // 로깅 레벨
}
```

**주요 메서드:**
- `Config::from_env()` - .env 파일 + 환경 변수에서 설정 로드
- `Config::validate()` - 설정 값 범위 검증 (fail-fast 패턴)
- `Config::default()` - 합리적인 기본값 제공

**설정 로딩 우선순위:**
1. `Config::default()` 기본값
2. `.env` 파일 (dotenvy)
3. 환경 변수
4. CLI 인자 (`--model` 플래그)

---

### 2. agent.rs - 연구 에이전트

```rust
pub struct ResearchAgent {
    config: Config,           // 설정 (소유권)
    search_tool: WebSearchTool,
}

impl ResearchAgent {
    pub fn new(config: Config) -> Self
    pub async fn research(&self, query: &str) -> Result<String>     // 검색 + LLM 합성
    pub async fn quick_search(&self, query: &str) -> Result<String> // 검색만 수행
}
```

**research() 워크플로우:**
1. Ollama 클라이언트 생성
2. Rig 에이전트 빌더로 도구 등록
3. 시스템 프롬프트 설정 (`RESEARCH_SYSTEM_PROMPT`)
4. 멀티턴 실행 (최대 5회 반복)
5. LLM이 `web_search` 도구 호출 → 결과 합성

**시스템 프롬프트 전략:**
- 쿼리당 단일 검색 강제 (무한 루프 방지)
- 결과 수신 후 즉시 합성
- 구조화된 출력 형식: Overview → Key Sources → Summary → Next Steps

---

### 3. tools.rs - 웹 검색 도구

```rust
pub struct WebSearchTool {
    max_results: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

#[derive(Error, Debug)]
pub enum SearchError {
    SearchFailed(String),
    RateLimited,
    NoResults(String),
    NetworkError(#[from] reqwest::Error),
}
```

**Rig Tool 트레이트 구현:**
```rust
impl Tool for WebSearchTool {
    const NAME: &'static str = "web_search";
    type Args = SearchArgs;
    type Output = String;
    type Error = SearchError;

    async fn definition(&self, _prompt: String) -> ToolDefinition;
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error>;
}
```

**HTML 파싱 전략 (3중 폴백):**
1. **uddg 파라미터**: `uddg=` 리다이렉트에서 URL 추출 → URL 디코딩 → 검증
2. **result__url 클래스**: `result__url` HTML 클래스에서 href 추출
3. **직접 HTTPS 추출**: `https://` URL 직접 추출 + 필터링 (DuckDuckGo 내부, CDN, 리소스 파일 제외)

**Rate Limiting:** 요청 전 500ms 지연

---

### 4. main.rs - CLI 진입점

```rust
#[derive(Parser)]
#[command(name = "ai-research-agent")]
#[command(about = "AI-powered research agent using local LLMs")]
struct Args {
    /// The research query or topic
    query: String,

    /// Override the default Ollama model
    #[arg(short, long)]
    model: Option<String>,

    /// Quick search mode (no AI synthesis)
    #[arg(short, long)]
    quick: bool,

    /// Enable verbose/debug logging
    #[arg(short, long)]
    verbose: bool,
}
```

**실행 흐름:**
```
인자 파싱 → 로깅 초기화 → 설정 로드 → 설정 검증
    → ResearchAgent 생성 → research()/quick_search() 실행
    → 결과 포맷팅 → 출력/에러 처리
```

**에러 처리:**
- 컨텍스트 인식 힌트 제공 (예: "Ollama가 실행 중인지 확인하세요")
- `anyhow::Result`로 애플리케이션 레벨 에러 전파
- 구조화된 로깅으로 디버깅 지원

---

## 실행 워크플로우

### Full Research 모드

```
┌─────────────────────────────────────────────────────────────┐
│  1. Ollama 클라이언트 생성                                   │
│       ↓                                                      │
│  2. Rig 에이전트 빌드                                        │
│      - 모델 선택                                             │
│      - 시스템 프롬프트 설정                                   │
│      - WebSearchTool 등록                                    │
│       ↓                                                      │
│  3. 멀티턴 프롬프트 실행                                      │
│      - "Research the following topic thoroughly..."          │
│      - 최대 5회 반복 (도구 사용)                              │
│       ↓                                                      │
│  4. LLM이 web_search 도구 호출                               │
│      - 정보 검색                                             │
│      - 결과 합성                                             │
│      - 출처 포함 요약 제공                                    │
└─────────────────────────────────────────────────────────────┘
```

### Quick Search 모드

```
search_tool 직접 호출 → 결과 포맷팅 (title, snippet, URL) → 반환 (LLM 없음)
```

---

## 테스트 구조

각 모듈에 단위 테스트가 포함되어 있습니다:

| 모듈 | 테스트 항목 |
|------|-------------|
| **config.rs** | `test_default_config`, `test_config_validation_valid`, `test_config_validation_invalid_temperature`, `test_config_validation_invalid_search_results` |
| **tools.rs** | `test_web_search_tool_creation`, `test_extract_domain`, `test_search_result_serialization` |
| **agent.rs** | `test_agent_creation`, `test_system_prompt_not_empty` |
| **main.rs** | `test_args_parsing`, `test_args_with_flags` |

**테스트 실행:**
```bash
cargo test
cargo test -- --nocapture  # 출력 표시
```

---

## Rust 학습 패턴

이 프로젝트에서 배울 수 있는 Rust 패턴들:

| 패턴 | 위치 | 설명 |
|------|------|------|
| **Struct & Fields** | config.rs | 커스텀 데이터 타입 |
| **Trait Implementation** | tools.rs | 인터페이스 패턴 (Tool 트레이트) |
| **Ownership & Borrowing** | agent.rs | 메모리 안전성 |
| **Async/Await** | agent.rs, tools.rs | 논블로킹 I/O |
| **Error Handling** | 전체 | Result 타입, `?` 연산자, anyhow |
| **Derive Macros** | 전체 | Debug, Clone, Serialize, Deserialize |
| **Option Type** | main.rs | 패턴 매칭, if let |
| **Builder Pattern** | agent.rs | Fluent 설정 |
| **Custom Error Types** | tools.rs | thiserror derive |
| **Unit Tests** | 전체 | `#[test]`, `#[cfg(test)]` |
| **Environment Loading** | config.rs | dotenvy, env::var |
| **CLI with Derive** | main.rs | clap derive 매크로 |
| **Structured Logging** | main.rs | tracing subscriber |

---

## 환경 변수

`.env.example`에서 복사하여 `.env` 생성:

```bash
# Ollama 설정
OLLAMA_MODEL=llama3.2              # 사용할 모델 (설치 필요)
OLLAMA_HOST=http://localhost:11434 # Ollama 서버 URL
OLLAMA_API_BASE_URL=...            # 대체 API 기본 URL

# 에이전트 설정
TEMPERATURE=0.7                    # 응답 창의성 (일반적으로 0.0-1.0)
MAX_SEARCH_RESULTS=5               # 분석할 검색 결과 수

# 로깅
RUST_LOG=info                      # 로깅 레벨 (info, debug, error)
```

---

## 빌드 및 실행

### 빌드

```bash
cargo build --release
```

### 실행

```bash
# 전체 연구 (검색 + LLM 합성)
cargo run -- "Rust async 최신 동향은?"

# 빠른 검색 (LLM 없이)
cargo run -- --quick "Rust 웹 프레임워크 2024"

# 모델 및 상세 로깅 지정
cargo run -- --model deepseek-v3.2 --verbose "주제"
```

### 필수 요건

1. **Ollama 설치 및 실행**: `ollama serve`
2. **모델 다운로드**: `ollama pull llama3.2`
3. **환경 설정**: `.env.example` → `.env` 복사 및 수정

---

## 아키텍처 강점

1. **교육적 초점** - Rust 패턴을 설명하는 상세한 인라인 주석
2. **프로덕션 준비** - 적절한 에러 처리, 로깅, 검증
3. **모듈성** - 명확한 관심사 분리 (CLI, 에이전트, 도구, 설정)
4. **확장성** - 새 도구 추가 용이 (Tool 트레이트 패턴)
5. **프라이버시 중심** - 로컬 LLM, DuckDuckGo (OpenAI/API 키 불필요)
6. **Async 네이티브** - Tokio 기반 논블로킹 I/O
7. **타입 안전** - Rust의 강력한 타이핑으로 버그 방지
8. **테스트 완비** - 각 모듈에 단위 테스트 포함

---

## Python research_agent와의 비교

| 항목 | Rust (rust-research-agent) | Python (research_agent) |
|------|---------------------------|-------------------------|
| **프레임워크** | Rig | LangChain/DeepAgents |
| **LLM** | Ollama (로컬) | OpenAI GPT-4.1 |
| **검색** | DuckDuckGo (API 키 불필요) | Tavily (API 키 필요) |
| **도구 정의** | `Tool` 트레이트 구현 | `@tool` 데코레이터 |
| **에이전트 구조** | 단일 에이전트 | 멀티 서브에이전트 |
| **실행 환경** | 완전 로컬 | 클라우드 API 의존 |

---

## 관련 문서

- [README.md](./README.md) - 빠른 시작 가이드
- [Cargo.toml](./Cargo.toml) - 의존성 상세
- [Rig 공식 문서](https://docs.rs/rig-core)
- [Ollama 공식 사이트](https://ollama.ai)
