// src/middleware/stack.rs
//! 미들웨어 스택
//!
//! 여러 미들웨어를 조합하여 순차적으로 실행합니다.

use std::sync::Arc;
use crate::state::AgentState;
use crate::error::MiddlewareError;
use crate::runtime::ToolRuntime;
use super::traits::{AgentMiddleware, DynTool, StateUpdate, ModelRequest, ModelResponse, ModelControl};

/// 미들웨어 스택
pub struct MiddlewareStack {
    middlewares: Vec<Arc<dyn AgentMiddleware>>,
}

impl MiddlewareStack {
    pub fn new() -> Self {
        Self { middlewares: vec![] }
    }

    /// 미들웨어 추가 (빌더 패턴)
    pub fn with_middleware<M: AgentMiddleware + 'static>(mut self, middleware: M) -> Self {
        self.middlewares.push(Arc::new(middleware));
        self
    }

    /// Arc로 래핑된 미들웨어 추가
    pub fn with_middleware_arc(mut self, middleware: Arc<dyn AgentMiddleware>) -> Self {
        self.middlewares.push(middleware);
        self
    }

    /// 미들웨어 개수
    pub fn len(&self) -> usize {
        self.middlewares.len()
    }

    pub fn is_empty(&self) -> bool {
        self.middlewares.is_empty()
    }

    /// 모든 미들웨어의 도구 수집
    pub fn collect_tools(&self) -> Vec<DynTool> {
        self.middlewares
            .iter()
            .flat_map(|m| m.tools())
            .collect()
    }

    /// 시스템 프롬프트 빌드 (체이닝)
    pub fn build_system_prompt(&self, base: &str) -> String {
        self.middlewares.iter().fold(
            base.to_string(),
            |acc, m| m.modify_system_prompt(acc)
        )
    }

    /// before_agent 훅 실행 (순차)
    pub async fn before_agent(
        &self,
        state: &mut AgentState,
        runtime: &ToolRuntime,
    ) -> Result<Vec<StateUpdate>, MiddlewareError> {
        let mut updates = vec![];

        for middleware in &self.middlewares {
            if let Some(update) = middleware.before_agent(state, runtime).await? {
                Self::apply_update(state, &update);
                updates.push(update);
            }
        }

        Ok(updates)
    }

    /// after_agent 훅 실행 (역순)
    pub async fn after_agent(
        &self,
        state: &mut AgentState,
        runtime: &ToolRuntime,
    ) -> Result<Vec<StateUpdate>, MiddlewareError> {
        let mut updates = vec![];

        for middleware in self.middlewares.iter().rev() {
            if let Some(update) = middleware.after_agent(state, runtime).await? {
                Self::apply_update(state, &update);
                updates.push(update);
            }
        }

        Ok(updates)
    }

    // =========================================================================
    // Model Call Hooks
    // =========================================================================

    /// before_model 훅 실행 (순차, 앞에서 뒤로)
    ///
    /// 각 미들웨어의 `before_model` 훅을 순차적으로 호출합니다.
    /// 미들웨어는 요청을 수정하거나, LLM 호출을 건너뛰거나, 인터럽트를 발생시킬 수 있습니다.
    ///
    /// # Returns
    ///
    /// - `ModelControl::Continue` - 모든 미들웨어가 Continue 반환
    /// - `ModelControl::ModifyRequest` - 마지막 수정된 요청 (request가 이미 수정됨)
    /// - `ModelControl::Skip(resp)` - LLM 호출 건너뛰기
    /// - `ModelControl::Interrupt(req)` - 실행 인터럽트
    pub async fn before_model(
        &self,
        request: &mut ModelRequest,
        state: &AgentState,
        runtime: &ToolRuntime,
    ) -> Result<ModelControl, MiddlewareError> {
        for middleware in &self.middlewares {
            match middleware.before_model(request, state, runtime).await? {
                ModelControl::Continue => continue,
                ModelControl::ModifyRequest(new_req) => {
                    // 요청 수정 후 계속 진행
                    *request = new_req;
                }
                control @ ModelControl::Skip(_) => {
                    // LLM 호출 건너뛰기 - 즉시 반환
                    tracing::debug!(
                        middleware = middleware.name(),
                        "Middleware skipping model call"
                    );
                    return Ok(control);
                }
                control @ ModelControl::Interrupt(_) => {
                    // 인터럽트 - 즉시 반환
                    tracing::info!(
                        middleware = middleware.name(),
                        "Middleware triggering interrupt in before_model"
                    );
                    return Ok(control);
                }
            }
        }
        Ok(ModelControl::Continue)
    }

    /// after_model 훅 실행 (역순, 뒤에서 앞으로)
    ///
    /// 각 미들웨어의 `after_model` 훅을 역순으로 호출합니다.
    /// 주로 HumanInTheLoop 인터럽트를 발생시키는 데 사용됩니다.
    ///
    /// # Returns
    ///
    /// - `ModelControl::Continue` - 모든 미들웨어가 Continue 반환
    /// - `ModelControl::Interrupt(req)` - 인간 승인 대기
    pub async fn after_model(
        &self,
        response: &ModelResponse,
        state: &AgentState,
        runtime: &ToolRuntime,
    ) -> Result<ModelControl, MiddlewareError> {
        for middleware in self.middlewares.iter().rev() {
            match middleware.after_model(response, state, runtime).await? {
                ModelControl::Continue => continue,
                control @ ModelControl::Interrupt(_) => {
                    // 인터럽트 - 즉시 반환
                    tracing::info!(
                        middleware = middleware.name(),
                        "Middleware triggering interrupt in after_model"
                    );
                    return Ok(control);
                }
                // Skip과 ModifyRequest는 after_model에서 의미 없음 - 무시
                ModelControl::Skip(_) | ModelControl::ModifyRequest(_) => {
                    tracing::warn!(
                        middleware = middleware.name(),
                        "Skip/ModifyRequest ignored in after_model (only valid in before_model)"
                    );
                    continue;
                }
            }
        }
        Ok(ModelControl::Continue)
    }

    /// 상태 업데이트 적용
    fn apply_update(state: &mut AgentState, update: &StateUpdate) {
        match update {
            StateUpdate::AddMessages(msgs) => {
                state.messages.extend(msgs.clone());
            }
            StateUpdate::SetMessages(msgs) => {
                // Replace entire message history (used by SummarizationMiddleware)
                state.messages = msgs.clone();
            }
            StateUpdate::SetTodos(todos) => {
                state.todos = todos.clone();
            }
            StateUpdate::UpdateFiles(files) => {
                for (path, data) in files {
                    if let Some(d) = data {
                        state.files.insert(path.clone(), d.clone());
                    } else {
                        state.files.remove(path);
                    }
                }
            }
            StateUpdate::Batch(updates) => {
                for u in updates {
                    Self::apply_update(state, u);
                }
            }
        }
    }
}

impl Default for MiddlewareStack {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::MemoryBackend;
    use async_trait::async_trait;

    struct TestMiddleware {
        name: String,
        prompt_addition: String,
    }

    #[async_trait]
    impl AgentMiddleware for TestMiddleware {
        fn name(&self) -> &str {
            &self.name
        }

        fn modify_system_prompt(&self, prompt: String) -> String {
            format!("{}\n{}", prompt, self.prompt_addition)
        }
    }

    #[test]
    fn test_middleware_stack_prompt_chaining() {
        let stack = MiddlewareStack::new()
            .with_middleware(TestMiddleware {
                name: "First".to_string(),
                prompt_addition: "First addition".to_string()
            })
            .with_middleware(TestMiddleware {
                name: "Second".to_string(),
                prompt_addition: "Second addition".to_string()
            });

        let result = stack.build_system_prompt("Base prompt");
        assert!(result.contains("Base prompt"));
        assert!(result.contains("First addition"));
        assert!(result.contains("Second addition"));
    }

    #[tokio::test]
    async fn test_middleware_stack_hooks() {
        let stack = MiddlewareStack::new()
            .with_middleware(TestMiddleware {
                name: "Test".to_string(),
                prompt_addition: "Test".to_string()
            });

        let mut state = AgentState::new();
        let backend = Arc::new(MemoryBackend::new());
        let runtime = ToolRuntime::new(state.clone(), backend);

        let updates = stack.before_agent(&mut state, &runtime).await.unwrap();
        assert!(updates.is_empty()); // 기본 미들웨어는 None 반환
    }

    #[test]
    fn test_middleware_stack_len() {
        let stack = MiddlewareStack::new()
            .with_middleware(TestMiddleware {
                name: "First".to_string(),
                prompt_addition: "First".to_string()
            })
            .with_middleware(TestMiddleware {
                name: "Second".to_string(),
                prompt_addition: "Second".to_string()
            });

        assert_eq!(stack.len(), 2);
        assert!(!stack.is_empty());
    }
}
