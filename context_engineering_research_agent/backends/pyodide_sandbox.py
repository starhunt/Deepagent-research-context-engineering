"""Pyodide 기반 WASM 샌드박스 백엔드.

WebAssembly 환경에서 Python 코드를 안전하게 실행하기 위한 백엔드입니다.

## Pyodide란?

Pyodide는 CPython을 WebAssembly로 컴파일한 프로젝트입니다.
브라우저나 Node.js 환경에서 Python 코드를 실행할 수 있습니다.

## 보안 모델

WASM 샌드박스는 다음과 같은 격리를 제공합니다:
- 호스트 파일시스템 접근 불가
- 네트워크 접근 제한 (JavaScript API 통해서만)
- 메모리 격리

## 한계

1. 네이티브 C 확장 라이브러리 제한적 지원
2. 성능 오버헤드 (네이티브 대비 ~3-10x 느림)
3. 초기 로딩 시간 (Pyodide 런타임 + 패키지)

## 권장 사용 사례

- 신뢰할 수 없는 사용자 코드 실행
- 브라우저 기반 Python 환경
- 격리된 데이터 분석 작업

## 사용 예시 (JavaScript 환경)

```javascript
const pyodide = await loadPyodide();
await pyodide.loadPackagesFromImports(pythonCode);
const result = await pyodide.runPythonAsync(pythonCode);
```
"""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@runtime_checkable
class PyodideRuntime(Protocol):
    def runPython(self, code: str) -> str: ...
    async def runPythonAsync(self, code: str) -> str: ...
    def loadPackagesFromImports(self, code: str) -> None: ...


@dataclass
class PyodideConfig:
    timeout_seconds: int = 30
    max_memory_mb: int = 512
    allowed_packages: list[str] = field(
        default_factory=lambda: [
            "numpy",
            "pandas",
            "scipy",
            "matplotlib",
            "scikit-learn",
        ]
    )
    enable_network: bool = False


@dataclass
class ExecuteResponse:
    output: str
    exit_code: int | None = None
    truncated: bool = False
    error: str | None = None


class PyodideSandboxBackend:
    """Pyodide WASM 샌드박스 백엔드.

    Note: 이 클래스는 설계 문서입니다.
    실제 구현은 JavaScript/TypeScript 환경에서 이루어집니다.

    Python에서 직접 Pyodide를 실행하려면 별도의 subprocess나
    JS 런타임(Node.js) 연동이 필요합니다.
    """

    def __init__(self, config: PyodideConfig | None = None) -> None:
        self.config = config or PyodideConfig()
        self._runtime: PyodideRuntime | None = None

    def execute(self, code: str) -> ExecuteResponse:
        """Python 코드를 WASM 샌드박스에서 실행합니다.

        실제 구현에서는 Node.js subprocess나
        WebWorker를 통해 Pyodide를 실행합니다.
        """
        return ExecuteResponse(
            output="Pyodide 실행은 JavaScript 환경에서만 지원됩니다.",
            exit_code=1,
            error="NotImplemented: Python에서 직접 Pyodide 실행 불가",
        )

    async def aexecute(self, code: str) -> ExecuteResponse:
        return self.execute(code)

    def get_pyodide_js_code(self, python_code: str) -> str:
        """주어진 Python 코드를 실행하는 JavaScript 코드를 생성합니다.

        이 JavaScript 코드를 브라우저나 Node.js에서 실행하면
        Pyodide 환경에서 Python 코드가 실행됩니다.
        """
        escaped_code = python_code.replace("`", "\\`").replace("$", "\\$")

        return f"""
import {{ loadPyodide }} from "pyodide";

async function runPythonInSandbox() {{
    const pyodide = await loadPyodide();
    
    const pythonCode = `{escaped_code}`;
    
    await pyodide.loadPackagesFromImports(pythonCode);
    
    try {{
        const result = await pyodide.runPythonAsync(pythonCode);
        return {{ success: true, result }};
    }} catch (error) {{
        return {{ success: false, error: error.message }};
    }}
}}

runPythonInSandbox().then(console.log);
"""


PYODIDE_DESIGN_DOC = """
# Pyodide 기반 안전한 코드 실행 설계

## 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                      Python Agent                           │
│  (LangGraph/DeepAgents)                                     │
└─────────────────┬───────────────────────────────────────────┘
                  │ execute() 호출
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                PyodideSandboxBackend                        │
│  - 코드 전처리                                               │
│  - 보안 검증                                                 │
│  - JS 코드 생성                                              │
└─────────────────┬───────────────────────────────────────────┘
                  │ subprocess 또는 IPC
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                   Node.js Runtime                           │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                  WebWorker                            │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │              Pyodide (WASM)                     │  │  │
│  │  │  - Python 인터프리터                            │  │  │
│  │  │  - 제한된 패키지                                │  │  │
│  │  │  - 격리된 메모리                                │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Docker vs Pyodide 비교

| 측면 | Docker | Pyodide (WASM) |
|------|--------|----------------|
| 격리 수준 | 컨테이너 (OS 레벨) | WASM 샌드박스 |
| 시작 시간 | ~1-2초 | ~3-5초 (최초), 이후 빠름 |
| 메모리 | 높음 | 낮음 |
| 패키지 지원 | 완전 | 제한적 |
| 보안 | 높음 | 매우 높음 |
| 호스트 접근 | 마운트 통해 가능 | 불가 |

## 권장 사용 시나리오

### Docker 사용
- 복잡한 라이브러리 필요
- 파일 I/O 필요
- 장시간 실행 작업

### Pyodide 사용
- 간단한 계산/분석
- 신뢰할 수 없는 코드
- 브라우저 환경
- 빠른 피드백 필요
"""
