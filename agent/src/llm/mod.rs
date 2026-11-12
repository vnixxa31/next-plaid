//! Text-generation engines.
//!
//! Two kinds of model can drive a session:
//!
//! - **Prompt engines** ([`Generator`], wrapped by [`prompt::PromptModel`]) continue a
//!   raw prompt the agent rendered from the model's chat template. The agent parses tool
//!   calls out of the generated text (the trained MiniCPM5 contract). Engines are
//!   expected to reuse their KV cache for the longest common prefix with the previous
//!   call: each turn re-sends the whole conversation, and only the new tokens should
//!   cost compute.
//! - **Chat engines** ([`chat::OpenAiChat`]) receive structured messages and tool
//!   definitions and answer with structured tool calls. The server owns the model's
//!   chat template and native tool-call syntax.
//!
//! [`TurnModel`] is the seam: the session loop only knows how to ask for the next
//! assistant turn, execute the calls it contains, and append the results to the
//! history.

use serde_json::Value;

use crate::protocol::{Message, ToolCall};

pub mod chat;
#[cfg(builtin_llama)]
pub mod llama;
pub mod openai;
pub mod prompt;
#[cfg(feature = "local")]
pub mod server;

/// Sampling and length settings for one generation.
#[derive(Debug, Clone, PartialEq)]
pub struct GenParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    /// 0 disables top-k.
    pub top_k: usize,
    pub seed: u64,
}

impl Default for GenParams {
    /// The harness's evaluation sampling: temperature 0.6, top-p 0.95, top-k 20,
    /// 2048 completion tokens per turn.
    fn default() -> Self {
        Self {
            max_tokens: 2048,
            temperature: 0.6,
            top_p: 0.95,
            top_k: 20,
            seed: 0,
        }
    }
}

/// What one generation produced.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Generation {
    /// Generated text, stop token excluded.
    pub text: String,
    pub prompt_tokens: usize,
    /// Prompt tokens served from the KV cache instead of recomputed.
    pub cached_tokens: usize,
    pub completion_tokens: usize,
    /// Prompt ingestion, up to the first sampled token (includes waiting on the GPU).
    pub prompt_ms: f64,
    pub generation_ms: f64,
    /// Turning the prompt text into tokens.
    pub tokenize_ms: f64,
}

/// What one assistant turn yielded.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Turn {
    /// Assistant text with the tool calls removed (empty when it only called tools).
    pub content: String,
    /// The assistant message as it should be kept in history when no call was produced
    /// (the prompt backend keeps raw markup; the chat backend keeps the server's text).
    pub assistant_text: String,
    /// Calls in the order the model produced them. The session replaces the ids with
    /// unique `call_N` values before the turn enters the history.
    pub tool_calls: Vec<ToolCall>,
    /// Per-call validation error, parallel to `tool_calls`. A call with an error is put
    /// in the history but must not be executed.
    pub call_errors: Vec<Option<String>>,
    /// The generation stopped at the token limit instead of a natural end.
    pub truncated: bool,
    pub generation: Generation,
    /// Time spent parsing the reply into calls.
    pub parse_ms: f64,
}

/// What the session asks for on one turn.
#[derive(Debug, Clone, PartialEq)]
pub struct TurnOptions {
    /// Only `finish` is served: force it where the backend supports it.
    pub final_turn: bool,
    /// `Some(false)` disables the model's reasoning block.
    pub enable_thinking: Option<bool>,
    pub sampling: GenParams,
}

/// A model that can produce the next assistant turn of a tool-using conversation.
pub trait TurnModel {
    /// Ask for the next assistant turn.
    ///
    /// Implementations must not execute anything: they only turn the conversation into
    /// a model request and parse the reply into [`Turn`]. Generated text may be passed
    /// to `on_text` as it becomes available (the chat backend is non-streaming and
    /// passes the final text once).
    fn next_turn(
        &mut self,
        messages: &[Message],
        tools: &[Value],
        options: &TurnOptions,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Turn, LlmError>;

    /// Prepare a cache for the part of every conversation that never changes (system
    /// prompt and tool definitions). Backends without such a cache do nothing.
    fn warm_prefix(
        &mut self,
        _system_prompt: &str,
        _tools: &[Value],
        _enable_thinking: Option<bool>,
    ) -> Result<(), LlmError> {
        Ok(())
    }
}

/// `(prompt_tokens, cached_prompt_tokens, completion_tokens)` from an OpenAI-style
/// `usage` object, or zeros when the server did not report them.
pub(crate) fn token_counts(usage: &Value) -> (usize, usize, usize) {
    let count = |p: &str| usage.pointer(p).and_then(Value::as_u64).unwrap_or(0) as usize;
    (
        count("/prompt_tokens"),
        count("/prompt_tokens_details/cached_tokens"),
        count("/completion_tokens"),
    )
}

#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    /// `prompt` is the tokens the request needs (the prompt, plus the room for the answer
    /// when the server counts it); both are 0 when the engine did not report them.
    #[error("{} (raise the agent context size)", overflow_message(*.prompt, *.ctx))]
    ContextOverflow { prompt: usize, ctx: usize },
    #[error("{0}")]
    Engine(String),
}

fn overflow_message(prompt: usize, ctx: usize) -> String {
    if prompt > 0 && ctx > 0 {
        format!("request of {prompt} tokens does not fit the {ctx}-token context")
    } else {
        "the request does not fit the context".to_string()
    }
}

/// An engine that continues raw prompts.
pub trait Generator {
    /// Continue `prompt`, passing each piece of generated text to `on_text` as soon as
    /// it is decoded.
    fn generate_streaming(
        &mut self,
        prompt: &str,
        params: &GenParams,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Generation, LlmError>;

    fn generate(&mut self, prompt: &str, params: &GenParams) -> Result<Generation, LlmError> {
        self.generate_streaming(prompt, params, &mut |_| {})
    }

    /// Prepare the KV cache for `prefix`, the part of every prompt that never changes
    /// (system prompt + tool definitions). Engines may persist it across runs; the
    /// default does nothing and the first `generate` evaluates it as usual.
    fn warm_prefix(&mut self, _prefix: &str) -> Result<(), LlmError> {
        Ok(())
    }
}
