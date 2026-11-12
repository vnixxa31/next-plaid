//! Generation through an OpenAI-compatible `/chat/completions` endpoint.
//!
//! The server owns the model's chat template and native tool-call syntax: colgrep sends
//! structured messages and tool definitions and receives structured tool calls, so no
//! model-side parser (and no special-token preservation) is needed. Responses are not
//! streamed in this first version: one request per turn.
//!
//! Server responses are untrusted input. Calls are only accepted when their name is one
//! of the served tools, their arguments are a JSON object, and the required arguments
//! are present; anything else becomes a tool-visible error and is not executed.

use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};

use super::openai::context_overflow;
use super::{token_counts, Generation, LlmError, Turn, TurnModel, TurnOptions};
use crate::protocol::{find_tool, Message, ToolCall};

/// An OpenAI-compatible chat-completions client.
pub struct OpenAiChat {
    base_url: String,
    model: String,
    api_key: Option<String>,
    agent: ureq::Agent,
}

impl OpenAiChat {
    pub fn new(base_url: &str, model: &str, api_key: Option<String>, timeout: Duration) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            api_key,
            agent: ureq::AgentBuilder::new().timeout(timeout).build(),
        }
    }

    fn request_body(&self, messages: &[Message], tools: &[Value], options: &TurnOptions) -> Value {
        let params = &options.sampling;
        let mut body = json!({
            "model": self.model,
            "messages": messages.iter().map(message_value).collect::<Vec<_>>(),
            "tools": tools,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "seed": params.seed % (i32::MAX as u64),
            "stream": false,
        });
        if params.top_k > 0 {
            body["top_k"] = params.top_k.into();
        }
        // llama.cpp and vLLM pass these to the chat template; the Gemma template reads
        // `enable_thinking` to decide whether to open a reasoning channel.
        if let Some(thinking) = options.enable_thinking {
            body["chat_template_kwargs"] = json!({ "enable_thinking": thinking });
        }
        if options.final_turn && find_tool(tools, "finish").is_some() {
            // Only `finish` is served on the last turn. `required` is enforced by
            // llama.cpp and OpenAI-compatible servers; a named function choice is
            // silently ignored by some of them.
            body["tool_choice"] = json!("required");
        }
        body
    }
}

impl TurnModel for OpenAiChat {
    fn next_turn(
        &mut self,
        messages: &[Message],
        tools: &[Value],
        options: &TurnOptions,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Turn, LlmError> {
        let url = format!("{}/chat/completions", self.base_url);
        let body = self.request_body(messages, tools, options);
        let mut req = self.agent.post(&url);
        if let Some(key) = &self.api_key {
            req = req.set("Authorization", &format!("Bearer {key}"));
        }
        let started = Instant::now();
        let response = match req.send_json(body) {
            Ok(r) => r,
            Err(ureq::Error::Status(code, r)) => {
                let text = r.into_string().unwrap_or_default();
                if let Some(overflow) = context_overflow(&text) {
                    return Err(overflow);
                }
                return Err(status_error(&url, code, &text));
            }
            // ureq's transport errors already name the URL.
            Err(e) => return Err(LlmError::Engine(e.to_string())),
        };
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        let raw = response
            .into_string()
            .map_err(|e| LlmError::Engine(format!("{url}: cannot read the response: {e}")))?;
        let value: Value = serde_json::from_str(&raw).map_err(|e| {
            LlmError::Engine(format!(
                "{url}: the server returned a non-JSON response ({e}): {}",
                snippet(&raw)
            ))
        })?;
        let t_parse = Instant::now();
        let mut turn = parse_response(&url, &value, tools)?;
        turn.parse_ms = t_parse.elapsed().as_secs_f64() * 1000.0;
        turn.generation.prompt_ms = timings_ms(&value, "prompt_ms");
        turn.generation.generation_ms = if timings_ms(&value, "predicted_ms") > 0.0 {
            timings_ms(&value, "predicted_ms")
        } else {
            elapsed_ms - turn.generation.prompt_ms
        };
        turn.generation.tokenize_ms = 0.0;
        if !turn.content.is_empty() {
            on_text(&turn.content);
        }
        Ok(turn)
    }
}

/// One message in the OpenAI chat-completions shape.
///
/// Tool-call arguments are a JSON *string* (the API's contract); llama.cpp and vLLM
/// deserialize them before rendering the model's template.
fn message_value(message: &Message) -> Value {
    match message {
        Message::System(c) => json!({ "role": "system", "content": c }),
        Message::User(c) => json!({ "role": "user", "content": c }),
        Message::Assistant {
            content,
            tool_calls,
        } => {
            let mut m = json!({ "role": "assistant", "content": content });
            if !tool_calls.is_empty() {
                let calls: Vec<Value> = tool_calls
                    .iter()
                    .map(|c| {
                        let arguments = serde_json::to_string(&c.arguments)
                            .unwrap_or_else(|_| "{}".to_string());
                        json!({
                            "id": c.id,
                            "type": "function",
                            "function": { "name": c.name, "arguments": arguments }
                        })
                    })
                    .collect();
                m["tool_calls"] = Value::Array(calls);
            }
            m
        }
        Message::Tool { call_id, content } => {
            json!({ "role": "tool", "tool_call_id": call_id, "content": content })
        }
    }
}

/// Parse one chat completion into a [`Turn`], validating the calls against `tools`.
fn parse_response(url: &str, value: &Value, tools: &[Value]) -> Result<Turn, LlmError> {
    if let Some(err) = value.get("error").filter(|e| !e.is_null()) {
        return Err(LlmError::Engine(format!("{url}: {err}")));
    }
    let choice = value.pointer("/choices/0").ok_or_else(|| {
        LlmError::Engine(format!(
            "{url}: response has no choices: {}",
            snippet(&value.to_string())
        ))
    })?;
    let message = choice
        .get("message")
        .ok_or_else(|| LlmError::Engine(format!("{url}: response choice has no message")))?;
    let content = content_text(message.get("content"));
    let truncated = choice.get("finish_reason").and_then(Value::as_str) == Some("length");

    let names: Vec<&str> = tools
        .iter()
        .filter_map(|t| t.pointer("/function/name").and_then(Value::as_str))
        .collect();
    let mut calls = Vec::new();
    let mut call_errors = Vec::new();
    for raw in message
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
    {
        let id = raw
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let name = raw
            .pointer("/function/name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let (arguments, mut error) = parse_arguments(raw.pointer("/function/arguments"));
        if error.is_none() && !names.contains(&name.as_str()) {
            error = Some(if names.is_empty() {
                format!("the server returned a call to '{name}' but no tools were served")
            } else {
                format!(
                    "unknown tool '{name}'; available tools: {}",
                    names.join(", ")
                )
            });
        }
        if error.is_none() {
            if let Some(missing) = missing_required(tools, &name, &arguments) {
                error = Some(format!("missing required argument '{missing}'"));
            }
        }
        calls.push(ToolCall {
            id,
            name,
            arguments,
        });
        call_errors.push(error);
    }

    let (prompt_tokens, cached_tokens, completion_tokens) =
        token_counts(value.get("usage").unwrap_or(&Value::Null));
    Ok(Turn {
        content: content.clone(),
        assistant_text: content.clone(),
        tool_calls: calls,
        call_errors,
        truncated,
        generation: Generation {
            text: String::new(),
            prompt_tokens,
            cached_tokens,
            completion_tokens,
            ..Default::default()
        },
        parse_ms: 0.0,
    })
}

/// Arguments from a `function.arguments` field: a JSON object, or its JSON string form.
/// Anything else is an error the model sees instead of execution.
fn parse_arguments(value: Option<&Value>) -> (Map<String, Value>, Option<String>) {
    match value {
        None | Some(Value::Null) => (Map::new(), None),
        Some(Value::Object(m)) => (m.clone(), None),
        Some(Value::String(s)) if s.trim().is_empty() => (Map::new(), None),
        Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
            Ok(Value::Object(m)) => (m, None),
            Ok(other) => (
                Map::new(),
                Some(format!("arguments must be a JSON object, got {other}")),
            ),
            Err(e) => (
                Map::new(),
                Some(format!("arguments are not valid JSON: {e}")),
            ),
        },
        Some(other) => (
            Map::new(),
            Some(format!("arguments must be a JSON object, got {other}")),
        ),
    }
}

/// The first required property the call did not supply, if any.
fn missing_required(tools: &[Value], name: &str, arguments: &Map<String, Value>) -> Option<String> {
    tools
        .iter()
        .find(|t| t.pointer("/function/name").and_then(Value::as_str) == Some(name))
        .and_then(|t| t.pointer("/function/parameters/required"))
        .and_then(Value::as_array)
        .and_then(|required| {
            required
                .iter()
                .filter_map(Value::as_str)
                .find(|key| !arguments.contains_key(*key))
                .map(str::to_string)
        })
}

/// Text of a chat `content` field: a string, or an array of text parts.
fn content_text(content: Option<&Value>) -> String {
    match content {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str).or_else(|| p.as_str()))
            .collect::<Vec<_>>()
            .join(""),
        Some(other) => other.to_string(),
    }
}

fn timings_ms(value: &Value, key: &str) -> f64 {
    value
        .pointer(&format!("/timings/{key}"))
        .and_then(Value::as_f64)
        .unwrap_or(0.0)
}

/// A short one-line excerpt of a server body, for diagnostics.
fn snippet(s: &str) -> String {
    let line = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if line.chars().count() > 300 {
        format!("{}…", line.chars().take(300).collect::<String>())
    } else {
        line
    }
}

/// An HTTP error with actionable text for the common failure modes.
fn status_error(url: &str, status: u16, body: &str) -> LlmError {
    let body = snippet(body);
    match status {
        401 | 403 => LlmError::Engine(format!(
            "{url}: HTTP {status} (authentication failed; set COLGREP_AGENT_API_KEY or \
             OPENAI_API_KEY): {body}"
        )),
        404 => LlmError::Engine(format!(
            "{url}: HTTP 404 (does the endpoint include the API prefix, e.g. /v1?): {body}"
        )),
        _ => LlmError::Engine(format!("{url}: HTTP {status}: {body}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{parse_tools, DEFAULT_TOOLS_JSON};
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;

    /// A one-shot HTTP server: answers the first request with `response` and reports the
    /// request body on the returned channel.
    fn serve_once(status: u16, response: &'static str) -> (String, mpsc::Receiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = Vec::new();
            let mut chunk = [0u8; 4096];
            let header_end = loop {
                let n = stream.read(&mut chunk).unwrap();
                buf.extend_from_slice(&chunk[..n]);
                if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                    break i + 4;
                }
            };
            let headers = String::from_utf8_lossy(&buf[..header_end]).to_lowercase();
            let len: usize = headers
                .lines()
                .find_map(|l| l.strip_prefix("content-length:"))
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(0);
            while buf.len() < header_end + len {
                let n = stream.read(&mut chunk).unwrap();
                buf.extend_from_slice(&chunk[..n]);
            }
            tx.send(String::from_utf8_lossy(&buf[header_end..]).into_owned())
                .unwrap();
            let reason = if status == 200 { "OK" } else { "Error" };
            write!(
                stream,
                "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n{response}",
                response.len()
            )
            .unwrap();
        });
        (format!("{base}/v1"), rx)
    }

    fn tools() -> Vec<Value> {
        parse_tools(DEFAULT_TOOLS_JSON).unwrap()
    }

    fn options(final_turn: bool) -> TurnOptions {
        TurnOptions {
            final_turn,
            enable_thinking: Some(false),
            sampling: crate::llm::GenParams {
                max_tokens: 128,
                temperature: 0.1,
                top_p: 0.9,
                top_k: 20,
                seed: 7,
            },
        }
    }

    fn conversation() -> Vec<Message> {
        vec![
            Message::System("SYS".into()),
            Message::User("find it".into()),
            Message::Assistant {
                content: String::new(),
                tool_calls: vec![ToolCall {
                    id: "srv_1".into(),
                    name: "colgrep".into(),
                    arguments: serde_json::from_str(r#"{"query":"loss","k":5}"#).unwrap(),
                }],
            },
            Message::Tool {
                call_id: "srv_1".into(),
                content: "$ colgrep 'loss' -k 5\n[1] src/a.py:1-2".into(),
            },
        ]
    }

    #[test]
    fn request_carries_messages_tools_and_sampling() {
        let (base, rx) = serve_once(
            200,
            r#"{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"done"}}],"usage":{"prompt_tokens":11,"completion_tokens":2}}"#,
        );
        let mut client = OpenAiChat::new(&base, "gemma", Some("k".into()), Duration::from_secs(5));
        let mut text = String::new();
        let turn = client
            .next_turn(&conversation(), &tools(), &options(false), &mut |t| {
                text.push_str(t)
            })
            .unwrap();
        assert_eq!(turn.content, "done");
        assert_eq!(text, "done");
        assert_eq!(
            (
                turn.generation.prompt_tokens,
                turn.generation.completion_tokens
            ),
            (11, 2)
        );
        let body: Value = serde_json::from_str(&rx.recv().unwrap()).unwrap();
        assert_eq!(body["model"], "gemma");
        assert_eq!(body["stream"], false);
        assert_eq!(body["max_tokens"], 128);
        assert_eq!(body["seed"], 7);
        assert_eq!(body["top_k"], 20);
        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
        assert!(body.get("tool_choice").is_none());
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][2]["role"], "assistant");
        // Arguments are the API's JSON-string form, not an object.
        let args = body["messages"][2]["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(args).unwrap(),
            serde_json::json!({"query": "loss", "k": 5})
        );
        assert_eq!(body["messages"][3]["tool_call_id"], "srv_1");
        assert_eq!(body["tools"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn last_turn_forces_finish_and_serves_only_finish() {
        let (base, rx) = serve_once(
            200,
            r#"{"choices":[{"finish_reason":"tool_calls","message":{"role":"assistant","content":null,"tool_calls":[{"id":"x","type":"function","function":{"name":"finish","arguments":"{\"locations\":[\"a.py:1-2\"]}"}}]}}]}"#,
        );
        let mut client = OpenAiChat::new(&base, "gemma", None, Duration::from_secs(5));
        let finish: Vec<Value> = tools()
            .into_iter()
            .filter(|t| t.pointer("/function/name").and_then(Value::as_str) == Some("finish"))
            .collect();
        let turn = client
            .next_turn(&conversation(), &finish, &options(true), &mut |_| {})
            .unwrap();
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].name, "finish");
        assert_eq!(
            turn.tool_calls[0].arguments["locations"],
            serde_json::json!(["a.py:1-2"])
        );
        assert!(turn.call_errors[0].is_none());
        let body: Value = serde_json::from_str(&rx.recv().unwrap()).unwrap();
        assert_eq!(body["tool_choice"], "required");
        assert_eq!(body["tools"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn malformed_arguments_and_unknown_tools_are_reported_not_executed() {
        let (base, _rx) = serve_once(
            200,
            r#"{"choices":[{"finish_reason":"tool_calls","message":{"role":"assistant","content":"","tool_calls":[
                {"id":"a","type":"function","function":{"name":"colgrep","arguments":"{not json"}},
                {"id":"b","type":"function","function":{"name":"nope","arguments":"{\"query\":\"x\"}"}},
                {"id":"c","type":"function","function":{"name":"terminal","arguments":"{}"}}
            ]}}]}"#,
        );
        let mut client = OpenAiChat::new(&base, "m", None, Duration::from_secs(5));
        let turn = client
            .next_turn(&conversation(), &tools(), &options(false), &mut |_| {})
            .unwrap();
        assert_eq!(turn.tool_calls.len(), 3);
        assert!(turn.call_errors[0]
            .as_deref()
            .unwrap()
            .contains("not valid JSON"));
        assert!(turn.call_errors[1]
            .as_deref()
            .unwrap()
            .contains("unknown tool 'nope'"));
        assert!(turn.call_errors[2]
            .as_deref()
            .unwrap()
            .contains("missing required argument 'command'"));
    }

    #[test]
    fn truncation_and_usage_are_reported() {
        let (base, _rx) = serve_once(
            200,
            r#"{"choices":[{"finish_reason":"length","message":{"role":"assistant","content":"thinking out loud"}}],
                "usage":{"prompt_tokens":5,"completion_tokens":8,"prompt_tokens_details":{"cached_tokens":3}},
                "timings":{"prompt_ms":12.5,"predicted_ms":40.0}}"#,
        );
        let mut client = OpenAiChat::new(&base, "m", None, Duration::from_secs(5));
        let turn = client
            .next_turn(&conversation(), &tools(), &options(false), &mut |_| {})
            .unwrap();
        assert!(turn.truncated);
        assert!(turn.tool_calls.is_empty());
        assert_eq!(
            (
                turn.generation.prompt_tokens,
                turn.generation.cached_tokens,
                turn.generation.completion_tokens
            ),
            (5, 3, 8)
        );
        assert_eq!(
            (turn.generation.prompt_ms, turn.generation.generation_ms),
            (12.5, 40.0)
        );
    }

    #[test]
    fn authentication_and_incompatible_responses_are_diagnosed() {
        let (base, _rx) = serve_once(401, r#"{"error":{"message":"invalid api key"}}"#);
        let mut client = OpenAiChat::new(&base, "m", None, Duration::from_secs(5));
        let err = client
            .next_turn(&conversation(), &tools(), &options(false), &mut |_| {})
            .unwrap_err();
        let text = err.to_string();
        assert!(text.contains("HTTP 401"));
        assert!(text.contains("COLGREP_AGENT_API_KEY"));

        let (base, _rx) = serve_once(200, r#"{"object":"list"}"#);
        let mut client = OpenAiChat::new(&base, "m", None, Duration::from_secs(5));
        let err = client
            .next_turn(&conversation(), &tools(), &options(false), &mut |_| {})
            .unwrap_err();
        assert!(err.to_string().contains("no choices"));
    }

    #[test]
    fn content_parts_are_joined() {
        let (base, _rx) = serve_once(
            200,
            r#"{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":[{"type":"text","text":"ab"},{"type":"text","text":"cd"}]}}]}"#,
        );
        let mut client = OpenAiChat::new(&base, "m", None, Duration::from_secs(5));
        let turn = client
            .next_turn(&conversation(), &tools(), &options(false), &mut |_| {})
            .unwrap();
        assert_eq!(turn.content, "abcd");
    }
}
