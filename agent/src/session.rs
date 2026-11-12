//! One localization session: the model searches, reads, then submits locations.
//!
//! A port of the training harness's session loop under the checkpoint's manifest
//! (`build_rl_manifest`): a 10-turn budget, up to 5 parallel tool calls per turn,
//! malformed replies resampled up to 3 times, and a last turn where only `finish` is
//! served after a final-turn reminder. No submission reminder, no rescue turns.

use std::path::Path;
use std::time::Instant;

use serde_json::Value;

use crate::llm::{GenParams, Generation, LlmError, TurnModel, TurnOptions};
use crate::protocol::{self, Message, ToolCall};
use crate::sandbox::Sandbox;
use crate::search::{self, SearchBackend};

/// What drives a session. Defaults reproduce the trained harness exactly.
#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub system_prompt: String,
    pub tools: Vec<Value>,
    pub max_turns: usize,
    /// Hits per `colgrep` call when the model does not pass `k`.
    pub search_top_k: usize,
    /// Serve only `finish` on the last turn, after [`protocol::FINAL_TURN_REMINDER`].
    pub force_finish_last_turn: bool,
    pub max_parallel_calls: usize,
    pub max_retries: usize,
    /// `Some(false)` renders the empty `<think>` block the model was trained with.
    pub enable_thinking: Option<bool>,
    pub sampling: GenParams,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            system_prompt: protocol::DEFAULT_SYSTEM_PROMPT.to_string(),
            tools: protocol::parse_tools(protocol::DEFAULT_TOOLS_JSON)
                .expect("bundled tool schema is valid JSON"),
            max_turns: 10,
            search_top_k: 10,
            force_finish_last_turn: true,
            max_parallel_calls: protocol::MAX_PARALLEL_TOOL_CALLS,
            max_retries: protocol::MAX_RETRIES,
            enable_thinking: Some(false),
            sampling: GenParams::default(),
        }
    }
}

/// A submitted code location.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Location {
    /// Repository-relative path.
    pub file: String,
    pub start_line: Option<usize>,
    pub end_line: Option<usize>,
}

impl std::fmt::Display for Location {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (self.start_line, self.end_line) {
            (Some(s), Some(e)) => write!(f, "{}:{s}-{e}", self.file),
            _ => write!(f, "{}", self.file),
        }
    }
}

/// Why a session ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndReason {
    /// `finish` was called with at least one valid location.
    Answered,
    /// `finish` was called but nothing in it was a valid location.
    AnsweredEmpty,
    /// The budget ran out without a `finish`.
    MaxTurns,
    /// Replies kept coming back without a tool call.
    MalformedRetries,
    ContextOverflow,
    LlmError,
}

/// Progress notifications, for live display.
#[derive(Debug)]
pub enum Event<'a> {
    TurnStarted {
        turn: usize,
        max_turns: usize,
        final_turn: bool,
    },
    /// A piece of text the model just generated (streamed while it writes).
    Token {
        turn: usize,
        text: &'a str,
    },
    ModelReplied {
        turn: usize,
        generation: &'a Generation,
        content: &'a str,
        calls: usize,
    },
    ToolCalled {
        turn: usize,
        tool: &'a str,
        command: &'a str,
    },
    ToolReturned {
        turn: usize,
        tool: &'a str,
        observation: &'a str,
        elapsed_ms: f64,
    },
    Retry {
        turn: usize,
        attempt: usize,
    },
}

pub trait Observer {
    fn on_event(&mut self, event: &Event<'_>);
}

impl Observer for () {
    fn on_event(&mut self, _: &Event<'_>) {}
}

/// One request the session sent to the model (diagnostics).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct RequestRecord {
    pub turn: usize,
    /// Only `finish` was served and its call was forced.
    pub final_turn: bool,
    /// Tool names sent with the request, in order.
    pub tools: Vec<String>,
    /// Messages in the conversation at the time of the request.
    pub messages: usize,
}

/// Everything a session produced.
#[derive(Debug, Clone)]
pub struct Outcome {
    /// What the model submitted through `finish` (validated, deduplicated).
    pub locations: Vec<Location>,
    /// Files the session read or retrieved, in first-seen order: the fallback answer
    /// when the model never submitted.
    pub inspected: Vec<Location>,
    pub end_reason: EndReason,
    pub error: Option<String>,
    pub turns: usize,
    pub n_searches: usize,
    pub n_reads: usize,
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub completion_tokens: usize,
    pub llm_ms: f64,
    pub tool_ms: f64,
    pub messages: Vec<Message>,
    /// What each turn sent: backend-independent (tools served, final-turn narrowing).
    pub requests: Vec<RequestRecord>,
    pub profile: Profile,
}

/// Where a session's time went, phase by phase (milliseconds).
#[derive(Debug, Clone, Default)]
pub struct Profile {
    /// Static prompt prefix: evaluated or restored from the on-disk cache.
    pub warm_prefix_ms: f64,
    /// Rendering the chat template each turn.
    pub render_ms: f64,
    pub tokenize_ms: f64,
    /// Prompt ingestion (prefill), up to the first generated token.
    pub prefill_ms: f64,
    /// Token generation after the first token.
    pub decode_ms: f64,
    /// Parsing replies into tool calls.
    pub parse_ms: f64,
    /// One entry per colgrep search.
    pub search_ms: Vec<f64>,
    /// One entry per terminal command.
    pub terminal_ms: Vec<f64>,
    /// Progress callbacks (terminal printing).
    pub observer_ms: f64,
    pub new_prompt_tokens: usize,
    pub generated_tokens: usize,
}

/// Run one session over `repo` for `task`.
///
/// `model` supplies the assistant turns (a prompt-rendered local/completion engine or a
/// chat-completions endpoint); everything else - the tool schemas, the executor, the
/// budgets and the `finish` validation - is backend-independent.
pub fn run(
    task: &str,
    repo: &Path,
    config: &SessionConfig,
    model: &mut dyn TurnModel,
    search_backend: &mut dyn SearchBackend,
    observer: &mut dyn Observer,
) -> std::io::Result<Outcome> {
    let mut sandbox = Sandbox::new(repo)?;
    let listing = sandbox.root_listing();
    let mut messages = vec![
        Message::System(config.system_prompt.clone()),
        Message::User(protocol::task_prompt(task, &listing)),
    ];
    let mut out = Outcome {
        locations: Vec::new(),
        inspected: Vec::new(),
        end_reason: EndReason::MaxTurns,
        error: None,
        turns: 0,
        n_searches: 0,
        n_reads: 0,
        prompt_tokens: 0,
        cached_tokens: 0,
        completion_tokens: 0,
        llm_ms: 0.0,
        tool_ms: 0.0,
        messages: Vec::new(),
        requests: Vec::new(),
        profile: Profile::default(),
    };
    let finish_only: Vec<Value> = config
        .tools
        .iter()
        .filter(|t| t.pointer("/function/name").and_then(Value::as_str) == Some("finish"))
        .cloned()
        .collect();
    let t = Instant::now();
    model
        .warm_prefix(&config.system_prompt, &config.tools, config.enable_thinking)
        .map_err(std::io::Error::other)?;
    out.profile.warm_prefix_ms = ms(t);
    let mut retries = 0;
    let mut next_call_id = 0usize;
    let max_turns = config.max_turns.max(1);
    let mut sampling = config.sampling.clone();

    'turns: for turn in 1..=max_turns {
        out.turns = turn;
        let final_turn =
            config.force_finish_last_turn && turn == max_turns && !finish_only.is_empty();
        observer.on_event(&Event::TurnStarted {
            turn,
            max_turns,
            final_turn,
        });
        if final_turn {
            messages.push(Message::User(protocol::FINAL_TURN_REMINDER.to_string()));
        }
        let tools = if final_turn {
            &finish_only
        } else {
            &config.tools
        };
        out.requests.push(RequestRecord {
            turn,
            final_turn,
            tools: tools
                .iter()
                .filter_map(|t| {
                    t.pointer("/function/name")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                })
                .collect(),
            messages: messages.len(),
        });
        // Vary the seed per turn so a resampled malformed turn is a fresh draw.
        sampling.seed = config.sampling.seed.wrapping_add(turn as u64);
        let mut reply = match model.next_turn(
            &messages,
            tools,
            &TurnOptions {
                final_turn,
                enable_thinking: config.enable_thinking,
                sampling: sampling.clone(),
            },
            &mut |text| observer.on_event(&Event::Token { turn, text }),
        ) {
            Ok(r) => r,
            Err(e) => {
                out.end_reason = match e {
                    LlmError::ContextOverflow { .. } => EndReason::ContextOverflow,
                    LlmError::Engine(_) => EndReason::LlmError,
                };
                out.error = Some(e.to_string());
                break;
            }
        };
        out.profile.parse_ms += reply.parse_ms;
        {
            let generation = &reply.generation;
            out.prompt_tokens += generation.prompt_tokens;
            out.cached_tokens += generation.cached_tokens;
            out.completion_tokens += generation.completion_tokens;
            out.llm_ms += generation.prompt_ms + generation.generation_ms + generation.tokenize_ms;
            out.profile.tokenize_ms += generation.tokenize_ms;
            out.profile.prefill_ms += generation.prompt_ms;
            out.profile.decode_ms += generation.generation_ms;
            out.profile.new_prompt_tokens += generation.prompt_tokens - generation.cached_tokens;
            out.profile.generated_tokens += generation.completion_tokens;

            let t_obs = Instant::now();
            observer.on_event(&Event::ModelReplied {
                turn,
                generation,
                content: &reply.content,
                calls: reply.tool_calls.len(),
            });
            out.profile.observer_ms += ms(t_obs);
        }
        // The session owns call ids: they stay unique across turns, and every assistant
        // call is paired with exactly one tool result. Servers generate their own ids,
        // which may repeat or be empty.
        for call in &mut reply.tool_calls {
            next_call_id += 1;
            call.id = format!("call_{next_call_id}");
        }

        if reply.tool_calls.is_empty() {
            // A reply with no tool call is malformed under the file/line protocol.
            messages.push(Message::Assistant {
                content: reply.assistant_text.clone(),
                tool_calls: Vec::new(),
            });
            if retries < config.max_retries {
                retries += 1;
                observer.on_event(&Event::Retry {
                    turn,
                    attempt: retries,
                });
                let mut feedback = protocol::retry_feedback(retries, config.max_retries);
                if reply.truncated {
                    feedback.push('\n');
                    feedback.push_str(&protocol::truncated_feedback(config.sampling.max_tokens));
                }
                messages.push(Message::User(feedback));
                continue;
            }
            out.end_reason = EndReason::MalformedRetries;
            break;
        }

        messages.push(Message::Assistant {
            content: reply.content.clone(),
            tool_calls: reply.tool_calls.clone(),
        });
        let mut answered: Vec<String> = Vec::new();
        let mut submitted = false;
        for (i, call) in reply
            .tool_calls
            .iter()
            .take(config.max_parallel_calls)
            .enumerate()
        {
            answered.push(call.id.clone());
            // A structured call that failed validation (bad arguments, unknown tool,
            // missing required argument) is answered but never executed.
            if let Some(reason) = reply.call_errors.get(i).and_then(Option::as_ref) {
                messages.push(Message::Tool {
                    call_id: call.id.clone(),
                    content: protocol::invalid_call(&call.name, reason),
                });
                continue;
            }
            if call.name == "finish" {
                let (locations, observation) = parse_finish(call, &sandbox);
                messages.push(Message::Tool {
                    call_id: call.id.clone(),
                    content: observation.to_string(),
                });
                out.end_reason = if locations.is_empty() {
                    EndReason::AnsweredEmpty
                } else {
                    EndReason::Answered
                };
                out.locations = locations;
                submitted = true;
                break;
            }
            let command = command_for(call);
            if command.is_empty() {
                messages.push(Message::Tool {
                    call_id: call.id.clone(),
                    content: protocol::UNPARSEABLE_CALL.to_string(),
                });
                continue;
            }
            retries = 0;
            let is_search =
                call.name == "colgrep" || command.split_whitespace().next() == Some("colgrep");
            let tool = if is_search { "colgrep" } else { "terminal" };
            let t_obs = Instant::now();
            observer.on_event(&Event::ToolCalled {
                turn,
                tool,
                command: &command,
            });
            out.profile.observer_ms += ms(t_obs);
            let started = Instant::now();
            let observation = if is_search {
                out.n_searches += 1;
                run_search(
                    &command,
                    &sandbox,
                    config.search_top_k,
                    search_backend,
                    &mut out.inspected,
                )
            } else {
                let result = sandbox.run(&command);
                if let Some((file, start, end)) = result.retrieved {
                    out.n_reads += 1;
                    note_inspected(&mut out.inspected, file, Some(start), Some(end));
                }
                result.output
            };
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            out.tool_ms += elapsed_ms;
            if is_search {
                out.profile.search_ms.push(elapsed_ms);
            } else {
                out.profile.terminal_ms.push(elapsed_ms);
            }
            observer.on_event(&Event::ToolReturned {
                turn,
                tool,
                observation: &observation,
                elapsed_ms,
            });
            messages.push(Message::Tool {
                call_id: call.id.clone(),
                content: protocol::format_observation(&command, &observation, turn, max_turns),
            });
        }
        // Every sampled call gets a response, including those after finish or past the cap.
        for call in &reply.tool_calls {
            if !answered.contains(&call.id) {
                messages.push(Message::Tool {
                    call_id: call.id.clone(),
                    content: protocol::NOT_EXECUTED.to_string(),
                });
            }
        }
        if submitted {
            break 'turns;
        }
    }
    out.messages = messages;
    Ok(out)
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

/// The command a non-finish call runs, or empty when it is unusable.
fn command_for(call: &ToolCall) -> String {
    match call.name.as_str() {
        "colgrep" => protocol::colgrep_command(&call.arguments),
        "terminal" => call
            .arguments
            .get("command")
            .and_then(Value::as_str)
            .map(|s| protocol::py_strip(s).to_string())
            .unwrap_or_default(),
        _ => String::new(),
    }
}

fn run_search(
    command: &str,
    sandbox: &Sandbox,
    default_top_k: usize,
    backend: &mut dyn SearchBackend,
    inspected: &mut Vec<Location>,
) -> String {
    match search::parse_command(command, sandbox, default_top_k) {
        Err(e) => format!("colgrep error: {e}"),
        Ok((req, opts)) => match backend.search(&req) {
            Err(e) => format!("colgrep error: {e}"),
            Ok(mut hits) => {
                hits.truncate(search::TOP_K_CAP);
                for h in &hits {
                    if h.start_line > 0 && h.end_line > 0 {
                        note_inspected(
                            inspected,
                            h.file.clone(),
                            Some(h.start_line),
                            Some(h.end_line),
                        );
                    }
                }
                search::render(&hits, opts)
            }
        },
    }
}

fn note_inspected(
    seen: &mut Vec<Location>,
    file: String,
    start: Option<usize>,
    end: Option<usize>,
) {
    if !seen.iter().any(|l| l.file == file) {
        seen.push(Location {
            file,
            start_line: start,
            end_line: end,
        });
    }
}

/// Validate a `finish` call into locations, and the tool response the harness gave.
///
/// The training contract asks for `file:START-END` entries. Model-visible behavior is
/// unchanged (the session ends either way), but here a partly malformed list keeps its
/// valid entries and a bare path counts as the whole file. Paths that do not exist or
/// leave the repository are dropped.
fn parse_finish(call: &ToolCall, sandbox: &Sandbox) -> (Vec<Location>, &'static str) {
    let entries: Vec<String> = match call.arguments.get("locations") {
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect(),
        Some(Value::String(s)) => s
            .split(['\n', ','])
            .map(|x| x.trim().trim_matches(['[', ']', '"', '\'']).to_string())
            .collect(),
        _ => Vec::new(),
    };
    let mut out: Vec<Location> = Vec::new();
    for entry in entries {
        let entry = entry
            .trim()
            .trim_start_matches(['-', '*', ' '])
            .trim_matches('`');
        if entry.is_empty() {
            continue;
        }
        let (file, range) = match entry.rsplit_once(':') {
            Some((f, r)) => match r.split_once('-') {
                Some((a, b)) => match (a.trim().parse::<usize>(), b.trim().parse::<usize>()) {
                    (Ok(a), Ok(b)) if a >= 1 && a <= b => (f, Some((a, b))),
                    _ => (f, None),
                },
                None => match r.trim().parse::<usize>() {
                    Ok(a) if a >= 1 => (f, Some((a, a))),
                    _ => (f, None),
                },
            },
            None => (entry, None),
        };
        let file = file.trim_start_matches("./");
        let Ok(abs) = sandbox.resolve_from(sandbox.root(), file) else {
            continue;
        };
        if !abs.is_file() {
            continue;
        }
        let rel = abs
            .strip_prefix(sandbox.root())
            .unwrap_or(&abs)
            .to_string_lossy()
            .replace('\\', "/");
        let loc = Location {
            file: rel,
            start_line: range.map(|r| r.0),
            end_line: range.map(|r| r.1),
        };
        if !out.contains(&loc) {
            out.push(loc);
        }
    }
    let observation = if out.is_empty() {
        "finish rejected: no valid locations"
    } else {
        "answer recorded"
    };
    (out, observation)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::llm::prompt::PromptModel;
    use crate::llm::{Generator, Turn};
    use crate::search::{SearchHit, SearchRequest};
    use crate::template::ChatTemplate;

    #[derive(Default)]
    struct Script {
        replies: Vec<&'static str>,
        prompts: Vec<String>,
    }

    /// A generator whose scripted replies are shared, so a test can inspect the prompts
    /// after the model has been moved into a `PromptModel`.
    #[derive(Clone, Default)]
    struct Scripted(Arc<Mutex<Script>>);

    impl Scripted {
        fn new(replies: Vec<&'static str>) -> Self {
            Self(Arc::new(Mutex::new(Script {
                replies,
                prompts: Vec::new(),
            })))
        }

        fn push(&self, reply: &'static str) {
            self.0.lock().unwrap().replies.push(reply);
        }

        fn prompts(&self) -> Vec<String> {
            self.0.lock().unwrap().prompts.clone()
        }
    }

    impl Generator for Scripted {
        fn generate_streaming(
            &mut self,
            prompt: &str,
            _: &GenParams,
            _: &mut dyn FnMut(&str),
        ) -> Result<Generation, LlmError> {
            let mut script = self.0.lock().unwrap();
            script.prompts.push(prompt.to_string());
            let text = if script.replies.is_empty() {
                ""
            } else {
                script.replies.remove(0)
            };
            Ok(Generation {
                text: text.to_string(),
                ..Default::default()
            })
        }
    }

    struct OneHit;
    impl SearchBackend for OneHit {
        fn search(&mut self, req: &SearchRequest) -> Result<Vec<SearchHit>, String> {
            assert_eq!(req.top_k, 10);
            Ok(vec![SearchHit {
                file: "src/auth.py".into(),
                start_line: 1,
                end_line: 2,
                name: "check".into(),
                unit_type: "function".into(),
                code: "def check():\n    pass".into(),
                score: 1.0,
            }])
        }
    }

    struct NeverSearch;
    impl SearchBackend for NeverSearch {
        fn search(&mut self, _: &SearchRequest) -> Result<Vec<SearchHit>, String> {
            panic!("a rejected call must not reach the search backend");
        }
    }

    fn repo() -> tempfile::TempDir {
        let d = tempfile::tempdir().unwrap();
        std::fs::create_dir(d.path().join("src")).unwrap();
        std::fs::write(d.path().join("src/auth.py"), "def check():\n    pass\n").unwrap();
        d
    }

    fn template() -> ChatTemplate {
        ChatTemplate::new(protocol::DEFAULT_CHAT_TEMPLATE, "<s>", "</s>").unwrap()
    }

    fn run_prompt(
        task: &str,
        repo: &Path,
        config: &SessionConfig,
        llm: &Scripted,
        backend: &mut dyn SearchBackend,
    ) -> Outcome {
        let mut model = PromptModel::new(template(), Box::new(llm.clone()));
        run(task, repo, config, &mut model, backend, &mut ()).unwrap()
    }

    #[test]
    fn search_read_finish() {
        let d = repo();
        let llm = Scripted::new(vec![
            "<function name=\"colgrep\"><param name=\"query\">token check</param></function>",
            "<function name=\"terminal\"><param name=\"command\">cat src/auth.py</param></function>",
            "<function name=\"finish\"><param name=\"locations\">[\"src/auth.py:1-2\", \"nope.py:1-2\"]</param></function>",
        ]);
        let out = run_prompt(
            "fix check",
            d.path(),
            &SessionConfig::default(),
            &llm,
            &mut OneHit,
        );
        assert_eq!(out.end_reason, EndReason::Answered);
        assert_eq!(
            out.locations
                .iter()
                .map(|l| l.to_string())
                .collect::<Vec<_>>(),
            vec!["src/auth.py:1-2"]
        );
        assert_eq!((out.n_searches, out.n_reads, out.turns), (1, 1, 3));
        let prompts = llm.prompts();
        let second_prompt = &prompts[1];
        assert!(second_prompt.contains(
            "<tool_response>\n$ colgrep 'token check'\n[1] src/auth.py:1-2  (function check)\n      def check():\n          pass\n\n[This is the output of turn 1 (10 turn limit).]\n</tool_response>"
        ));
        assert!(prompts[0].contains("Problem statement:\nfix check\n\nYou are at the repository root. Top-level contents:\nsrc/\n\n"));
    }

    #[test]
    fn last_turn_serves_only_finish() {
        let d = repo();
        let read = "<function name=\"terminal\"><param name=\"command\">ls</param></function>";
        let llm = Scripted::new(vec![read; 9]);
        llm.push("<param name=\"locations\">[\"src/auth.py:1-2\"]</param></function>");
        let out = run_prompt("t", d.path(), &SessionConfig::default(), &llm, &mut OneHit);
        assert_eq!(out.turns, 10);
        assert_eq!(out.end_reason, EndReason::Answered);
        let prompts = llm.prompts();
        let last = prompts.last().unwrap();
        assert!(last.ends_with(&format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<function name=\"finish\">",
            protocol::FINAL_TURN_REMINDER
        )));
        // Only the finish tool is listed in the final system turn.
        assert!(!last.contains("\"name\": \"colgrep\""));
        assert!(prompts[0].contains("\"name\": \"colgrep\""));
    }

    #[test]
    fn text_replies_are_retried_then_give_up() {
        let d = repo();
        let llm = Scripted::new(vec!["src/auth.py:1-2"; 4]);
        let out = run_prompt("t", d.path(), &SessionConfig::default(), &llm, &mut OneHit);
        assert_eq!(out.end_reason, EndReason::MalformedRetries);
        assert!(out.locations.is_empty());
        assert!(llm.prompts()[1].contains("That was not a valid action. Either call a tool with a command, or submit via finish / <answer>.\n(Attempt 1/3)"));
    }

    #[test]
    fn extra_parallel_calls_are_answered_not_executed() {
        let d = repo();
        let call = "<function name=\"terminal\"><param name=\"command\">pwd</param></function>";
        let six = [call; 6].join("\n");
        let six: &'static str = Box::leak(six.into_boxed_str());
        let llm = Scripted::new(vec![six]);
        let cfg = SessionConfig {
            max_turns: 2,
            force_finish_last_turn: false,
            ..Default::default()
        };
        let out = run_prompt("t", d.path(), &cfg, &llm, &mut OneHit);
        let tool_msgs: Vec<&Message> = out
            .messages
            .iter()
            .filter(|m| matches!(m, Message::Tool { .. }))
            .collect();
        assert_eq!(tool_msgs.len(), 6);
        assert!(
            matches!(tool_msgs[5], Message::Tool { content, .. } if content == protocol::NOT_EXECUTED)
        );
    }

    // ------------------------------------------------- structured (chat) turns

    fn call(name: &str, args: &str) -> ToolCall {
        ToolCall {
            id: String::new(),
            name: name.into(),
            arguments: serde_json::from_str(args).unwrap(),
        }
    }

    fn turn(calls: Vec<ToolCall>, errors: Vec<Option<String>>) -> Turn {
        Turn {
            tool_calls: calls,
            call_errors: errors,
            ..Default::default()
        }
    }

    /// A model that returns scripted turns and records what the session sent it.
    #[derive(Default)]
    struct TurnScript {
        turns: Vec<Turn>,
        seen: Vec<(Vec<Message>, Vec<String>, bool)>,
    }

    impl TurnModel for TurnScript {
        fn next_turn(
            &mut self,
            messages: &[Message],
            tools: &[Value],
            options: &TurnOptions,
            _: &mut dyn FnMut(&str),
        ) -> Result<Turn, LlmError> {
            self.seen.push((
                messages.to_vec(),
                tools
                    .iter()
                    .filter_map(|t| {
                        t.pointer("/function/name")
                            .and_then(Value::as_str)
                            .map(str::to_string)
                    })
                    .collect(),
                options.final_turn,
            ));
            Ok(self.turns.remove(0))
        }
    }

    fn tool_results(messages: &[Message]) -> Vec<(String, String)> {
        messages
            .iter()
            .filter_map(|m| match m {
                Message::Tool { call_id, content } => Some((call_id.clone(), content.clone())),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn chat_turns_pair_calls_with_results_and_keep_history() {
        let d = repo();
        let mut script = TurnScript {
            turns: vec![
                turn(
                    vec![
                        call("colgrep", r#"{"query":"check"}"#),
                        call("terminal", r#"{"command":"cat src/auth.py"}"#),
                    ],
                    vec![None, None],
                ),
                turn(
                    vec![call("finish", r#"{"locations":["src/auth.py:1-2"]}"#)],
                    vec![None],
                ),
            ],
            seen: Vec::new(),
        };
        let out = run(
            "t",
            d.path(),
            &SessionConfig::default(),
            &mut script,
            &mut OneHit,
            &mut (),
        )
        .unwrap();
        assert_eq!(out.end_reason, EndReason::Answered);
        assert_eq!(out.locations[0].to_string(), "src/auth.py:1-2");
        assert_eq!((out.n_searches, out.n_reads, out.turns), (1, 1, 2));

        // The second request carries the assistant's two calls and one result for each,
        // with matching ids (the server did not supply any).
        let second = &script.seen[1].0;
        let ids = match second.get(2) {
            Some(Message::Assistant { tool_calls, .. }) => {
                tool_calls.iter().map(|c| c.id.clone()).collect::<Vec<_>>()
            }
            other => panic!("expected the assistant turn, got {other:?}"),
        };
        assert_eq!(ids, vec!["call_1", "call_2"]);
        let results = tool_results(second);
        assert_eq!(
            results
                .iter()
                .map(|(id, _)| id.as_str())
                .collect::<Vec<_>>(),
            vec!["call_1", "call_2"]
        );
        assert!(results[0].1.contains("$ colgrep check"));
        assert!(results[1].1.contains("def check():"));
    }

    #[test]
    fn chat_rejects_invalid_calls_without_executing_them() {
        let d = repo();
        let mut script = TurnScript {
            turns: vec![
                turn(
                    vec![
                        call("colgrep", r#"{"query":"x"}"#),
                        call("terminal", r#"{"command":"pwd"}"#),
                    ],
                    vec![Some("arguments are not valid JSON: boom".into()), None],
                ),
                turn(vec![call("finish", r#"{"locations":[]}"#)], vec![None]),
            ],
            seen: Vec::new(),
        };
        let out = run(
            "t",
            d.path(),
            &SessionConfig::default(),
            &mut script,
            &mut NeverSearch,
            &mut (),
        )
        .unwrap();
        assert_eq!(out.end_reason, EndReason::AnsweredEmpty);
        let results = tool_results(&script.seen[1].0);
        assert!(results[0].1.contains("arguments are not valid JSON: boom"));
        assert!(results[0].1.contains("the 'colgrep' call was not executed"));
        // The valid call in the same turn still executes.
        assert!(results[1].1.contains("$ pwd"));
    }

    #[test]
    fn chat_calls_after_finish_are_answered_not_executed() {
        let d = repo();
        let mut script = TurnScript {
            turns: vec![turn(
                vec![
                    call("finish", r#"{"locations":["src/auth.py:1-2"]}"#),
                    call("terminal", r#"{"command":"pwd"}"#),
                ],
                vec![None, None],
            )],
            seen: Vec::new(),
        };
        let out = run(
            "t",
            d.path(),
            &SessionConfig::default(),
            &mut script,
            &mut NeverSearch,
            &mut (),
        )
        .unwrap();
        assert_eq!(out.end_reason, EndReason::Answered);
        assert_eq!(out.locations[0].to_string(), "src/auth.py:1-2");
        // The call after finish is paired with a result but never run.
        let results = tool_results(&out.messages);
        assert_eq!(results.len(), 2);
        assert_eq!(results[1].1, protocol::NOT_EXECUTED);
        assert!(out.profile.terminal_ms.is_empty());
    }

    #[test]
    fn chat_rejected_finish_is_answered_but_not_submitted() {
        let d = repo();
        let mut script = TurnScript {
            turns: vec![turn(
                vec![call("finish", r#"{"locations":["src/auth.py:1-2"]}"#)],
                vec![Some("arguments are not valid JSON: boom".into())],
            )],
            seen: Vec::new(),
        };
        let cfg = SessionConfig {
            max_turns: 1,
            ..Default::default()
        };
        let out = run("t", d.path(), &cfg, &mut script, &mut NeverSearch, &mut ()).unwrap();
        assert_eq!(out.end_reason, EndReason::MaxTurns);
        assert!(out.locations.is_empty());
        let results = tool_results(&out.messages);
        assert_eq!(results.len(), 1);
        assert!(results[0].1.contains("the 'finish' call was not executed"));
    }

    #[test]
    fn chat_truncated_reply_asks_for_a_call() {
        let d = repo();
        let mut script = TurnScript {
            turns: vec![
                Turn {
                    assistant_text: "I was writing a long explanation".into(),
                    truncated: true,
                    ..Default::default()
                },
                turn(
                    vec![call("finish", r#"{"locations":["src/auth.py:1-2"]}"#)],
                    vec![None],
                ),
            ],
            seen: Vec::new(),
        };
        let out = run(
            "t",
            d.path(),
            &SessionConfig::default(),
            &mut script,
            &mut OneHit,
            &mut (),
        )
        .unwrap();
        assert_eq!(out.end_reason, EndReason::Answered);
        let second = &script.seen[1].0;
        assert!(matches!(
            second.last(),
            Some(Message::User(text)) if text.contains("cut off at the 2048-token limit")
        ));
        assert!(matches!(
            second.get(2),
            Some(Message::Assistant { content, .. }) if content == "I was writing a long explanation"
        ));
    }

    #[test]
    fn chat_final_turn_serves_only_finish() {
        let d = repo();
        let mut script = TurnScript {
            turns: vec![
                turn(vec![call("colgrep", r#"{"query":"check"}"#)], vec![None]),
                turn(vec![call("colgrep", r#"{"query":"check"}"#)], vec![None]),
                turn(
                    vec![call("finish", r#"{"locations":["src/auth.py:1-2"]}"#)],
                    vec![None],
                ),
            ],
            seen: Vec::new(),
        };
        let cfg = SessionConfig {
            max_turns: 3,
            ..Default::default()
        };
        let out = run("t", d.path(), &cfg, &mut script, &mut OneHit, &mut ()).unwrap();
        assert_eq!(out.end_reason, EndReason::Answered);
        assert!(!script.seen[0].2);
        assert!(script.seen[2].2);
        assert_eq!(script.seen[2].1, vec!["finish"]);
        // The recorded requests make the final-turn narrowing auditable.
        assert_eq!(out.requests.len(), 3);
        assert_eq!(out.requests[0].tools.len(), 3);
        assert!(!out.requests[0].final_turn);
        assert_eq!(out.requests[2].tools, vec!["finish"]);
        assert!(out.requests[2].final_turn);
        assert!(matches!(
            script.seen[2].0.last(),
            Some(Message::User(text)) if text == protocol::FINAL_TURN_REMINDER
        ));
    }
}
