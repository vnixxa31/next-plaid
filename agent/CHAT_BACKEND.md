# Chat-completions backend

Implementation summary for `colgrep --agent` over an OpenAI-compatible
`/chat/completions` endpoint (first target: Gemma 4 E4B on llama.cpp). User-facing
configuration and troubleshooting live in `colgrep/README.md`.

## Shape of the change

The session loop was already the right owner of the workflow: tool schemas, the
sandboxed executor, turn budgets, the final-turn reminder and `finish` validation. Only
the "ask the model for the next action" step depended on a prompt-rendered template.
That step is now a trait, `llm::TurnModel`:

```text
session::run
  └─ TurnModel::next_turn(messages, tools, options) -> Turn
       ├─ prompt::PromptModel   (existing path: render template, generate text,
       │                         parse <function> markup; still the default)
       └─ chat::OpenAiChat      (new path: POST structured messages + tools,
                                 read structured tool_calls)
```

`Turn` carries the assistant text, the calls in order, a per-call validation error, a
truncation flag and token/timing counts. Both backends therefore share one loop, one
executor and one final-result format; the chat path cannot bypass application-side
validation.

The session assigns call ids (`call_1`, `call_2`, …) instead of trusting server ids.
This keeps assistant calls paired with exactly one tool result across turns even when a
server returns empty or duplicate ids (llama.cpp ignores ids when rendering and resolves
results by function name, last-write-wins on duplicates).

## Why `tool_choice` is `"required"` on the final turn

On the tested llama.cpp build, a named function choice
(`{"type":"function","function":{"name":"finish"}}`) is accepted but ignored: the prompt
and token counts are unchanged and the model picks its own action. `"required"` is
enforced. The final turn already serves only `finish`, so `"required"` guarantees a
tool call and lets the existing `finish` validator decide whether the answer is valid.
(Some servers silently degrade a malformed `tool_choice` to `auto`; `required` also
avoids that.)

## Validation of structured calls

Server-parsed calls are still untrusted. Before execution, `chat::parse_response`
rejects, per call:

- arguments that are not a JSON object (objects, JSON-object strings and empty/missing
  values are accepted; other JSON shapes and invalid JSON are rejected);
- tool names not in the served tool list (the error lists the available tools);
- calls missing a required argument declared by that tool's schema.

A rejected call is still placed in the history (association preserved) and answered
with an `ERROR: …; the '<name>' call was not executed.` tool message. It never reaches
the search backend or the sandbox. Unknown names on the final turn are flagged the same
way because only `finish` is served there.

## Compatibility notes

- No client-side template, BOS handling, `--special`, `preserved_tokens` or token
  filtering is involved. The server applies the model's Jinja template.
- Reasoning is off by default and sent as
  `chat_template_kwargs: {"enable_thinking": …}` (llama.cpp/vLLM extension; the Gemma 4
  template consumes it). `--agent-thinking` exposes it.
- History tool calls send `function.arguments` as a JSON string (the API contract).
  Invalid-argument calls are stored with `{}` in history so the server never sees a
  non-object `arguments` value (llama.cpp answers those with a 500).
- Every assistant turn with calls is answered with one tool message per call (including
  calls above the parallel cap or after `finish`), so a follow-up request never ends on
  an assistant tool-call message (llama.cpp rejects that).
- `--agent-chat-template` is ignored with a warning: format ownership moved to the
  server. The system prompt, `--agent-tools`, budgets, sampling and
  `--agent-final-finish` still apply.
- `--agent-context` is local-only; the chat backend uses the server's context and maps
  `exceed_context_size_error` / "maximum context length" bodies to the existing
  `ContextOverflow` end reason.
- The model id is sent as configured. llama.cpp does not validate it (an unknown name
  still returns 200), so the endpoint's loaded model is the source of truth.

## Failure behavior

| Situation | Behavior |
|---|---|
| connection refused / timeout | `LlmError::Engine` with the endpoint URL; session ends `LlmError`, CLI exits non-zero |
| 401/403 | error names `COLGREP_AGENT_API_KEY` / `OPENAI_API_KEY` |
| 404 | error suggests the missing `/v1` prefix |
| non-JSON or no `choices` | error includes a short excerpt of the body |
| context overflow | `EndReason::ContextOverflow` with the existing "raise the context" message |
| `finish_reason: "length"` with no call | treated as a malformed turn; retry feedback says the reply was cut off at the token limit; after 3 retries `MalformedRetries` |
| empty reply / prose instead of a call | existing retry loop (3 attempts) |
| invalid call | tool-visible error, not executed; the loop continues within the budget |
| invalid `finish` (or no valid location) | `AnsweredEmpty`, never reported as success |
| budget exhausted | `MaxTurns`; the CLI shows the files the agent inspected instead of a submitted answer |

Retries stay bounded by the existing `max_retries` and `max_turns`; there is no
transport-level retry loop.

## Observability

- `colgrep --agent` prints `model @ endpoint (chat)` in the header.
- `COLGREP_AGENT_PROFILE=1` includes the parse/prefill/decode timings and token counts
  reported by the server.
- `COLGREP_AGENT_TRACE=<file>` writes an object with `backend`, `model`, the session
  tool definitions, a per-turn `requests` list (messages in the conversation, final-turn
  flag, tool names actually served), the full message transcript, turn count, end reason
  and error. Messages use the existing OpenAI-shaped `to_template_value` form; server
  ids are the session's.
- API keys are only placed in the request's `Authorization` header and are never logged.

## Tests and limits

`agent/src/llm/chat.rs` uses a loopback HTTP server to check the request shape (messages,
tools, sampling, `chat_template_kwargs`, final-turn `tool_choice`), response parsing
(content parts, calls, usage/timings, truncation) and diagnostics (401, no choices,
malformed arguments, unknown tools, missing required arguments).
`agent/src/session.rs` drives scripted `TurnModel`s to check call/result pairing,
history across turns, rejection without execution, truncation feedback, the final-turn
tool list and the existing prompt-path regressions. Engine tests assert backend
selection. Routine tests need neither a model nor a server.

Known limits (intentional for the first version): non-streaming responses, text input
only, no transport retries, no client-side context budgeting (the server's limit ends
the session), and no attempt to match the trained checkpoint's search quality. The
`chat_template_kwargs` and `top_k` are extensions beyond the minimal OpenAI schema;
`seed` is standard but not universally supported. A strict provider may reject the
extensions. Non-object `tool_choice` shapes are not used.
