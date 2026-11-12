//! Prompt-rendered engine adapter: the agent renders the model's chat template and parses
//! tool calls out of the raw generated text.
//!
//! This is the trained MiniCPM5 contract and the existing backend: byte-identical prompt,
//! `<function name="…">` markup, and `tool_choice=finish` reproduced as a prefill on the
//! last turn.

use std::time::Instant;

use serde_json::Value;

use crate::llm::{Generator, LlmError, Turn, TurnModel, TurnOptions};
use crate::protocol::Message;
use crate::template::ChatTemplate;
use crate::toolcall;

/// Prefill that forces the first call of the last turn to be `finish`.
const FINAL_TURN_PREFILL: &str = "<function name=\"finish\">";

/// A [`Generator`] plus the chat template that turns messages into its prompts.
pub struct PromptModel {
    template: ChatTemplate,
    generator: Box<dyn Generator>,
    /// Keeps tool-call ids unique across the turns of one session.
    next_id: usize,
}

impl PromptModel {
    pub fn new(template: ChatTemplate, generator: Box<dyn Generator>) -> Self {
        Self {
            template,
            generator,
            next_id: 0,
        }
    }
}

impl TurnModel for PromptModel {
    fn next_turn(
        &mut self,
        messages: &[Message],
        tools: &[Value],
        options: &TurnOptions,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Turn, LlmError> {
        let prompt = self
            .template
            .render(messages, tools, options.enable_thinking)
            .map_err(|e| LlmError::Engine(format!("chat template: {e}")))?;
        let prefill = if options.final_turn {
            FINAL_TURN_PREFILL
        } else {
            ""
        };
        if !prefill.is_empty() {
            on_text(prefill);
        }
        let full = format!("{prompt}{prefill}");
        let generation = self
            .generator
            .generate_streaming(&full, &options.sampling, on_text)?;

        let raw = format!("{prefill}{}", generation.text);
        let t = Instant::now();
        let reply = toolcall::parse_reply(&raw, tools, &mut self.next_id);
        let parse_ms = t.elapsed().as_secs_f64() * 1000.0;
        let call_errors = vec![None; reply.tool_calls.len()];
        Ok(Turn {
            content: reply.content,
            assistant_text: raw.trim().to_string(),
            tool_calls: reply.tool_calls,
            call_errors,
            truncated: false,
            generation,
            parse_ms,
        })
    }

    fn warm_prefix(
        &mut self,
        system_prompt: &str,
        tools: &[Value],
        enable_thinking: Option<bool>,
    ) -> Result<(), LlmError> {
        match static_prefix(&self.template, system_prompt, tools, enable_thinking) {
            Some(prefix) => self.generator.warm_prefix(&prefix),
            None => Ok(()),
        }
    }
}

/// The rendered text every session starts with: what two renders differing only in the
/// task have in common (system prompt and tool definitions, for chat templates).
fn static_prefix(
    template: &ChatTemplate,
    system_prompt: &str,
    tools: &[Value],
    enable_thinking: Option<bool>,
) -> Option<String> {
    let render = |task: &str| {
        template
            .render(
                &[
                    Message::System(system_prompt.to_string()),
                    Message::User(task.to_string()),
                ],
                tools,
                enable_thinking,
            )
            .ok()
    };
    let (a, b) = (render("\u{1}")?, render("\u{2}")?);
    let mut end = a
        .char_indices()
        .zip(b.chars())
        .take_while(|((_, x), y)| x == y)
        .last()
        .map(|((i, c), _)| i + c.len_utf8())?;
    // Stop at a line boundary so the cut never splits a token.
    end = a[..end].rfind('\n').map(|i| i + 1)?;
    Some(a[..end].to_string())
}
