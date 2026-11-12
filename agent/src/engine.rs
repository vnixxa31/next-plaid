//! Build the model a session runs on, from the user's settings.

use std::time::Duration;

use crate::config::{AgentSettings, EndpointApi};
use crate::llm::chat::OpenAiChat;
use crate::llm::openai::OpenAiCompletions;
use crate::llm::prompt::PromptModel;
use crate::llm::TurnModel;
use crate::protocol;
use crate::template::ChatTemplate;

/// Which inference path a session uses (for diagnostics).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// llama.cpp compiled into colgrep (Apple Silicon).
    Builtin,
    /// The managed `llama-server` subprocess.
    Server,
    /// A user-configured endpoint's raw `/completions` API.
    Completions,
    /// A user-configured endpoint's `/chat/completions` API.
    Chat,
}

impl Backend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Builtin => "builtin",
            Self::Server => "llama-server",
            Self::Completions => "completions",
            Self::Chat => "chat-completions",
        }
    }
}

/// A ready-to-use model and a one-line description for display.
pub struct Engine {
    pub model: Box<dyn TurnModel>,
    pub backend: Backend,
    pub description: String,
}

/// Load the model the settings describe: an OpenAI-compatible endpoint when one is
/// configured, local inference otherwise.
///
/// For the prompt backends the chat template is, in order: the configured template file,
/// the template embedded in the model file, then the bundled template of the default
/// model. The chat backend has no client-side template: the server applies the model's.
pub fn load_engine(
    settings: &AgentSettings,
    force_cpu: bool,
    progress: bool,
) -> Result<Engine, String> {
    if let Some(endpoint) = &settings.endpoint {
        let model_id = settings
            .endpoint_model
            .clone()
            .unwrap_or_else(|| settings.model().to_string());
        let key = crate::llm::openai::api_key_from_env();
        let timeout = Duration::from_secs(settings.endpoint_timeout_secs());
        match settings.endpoint_api() {
            EndpointApi::Chat => {
                if settings.chat_template_file.is_some() {
                    eprintln!(
                        "warning: --agent-chat-template is ignored by the chat backend: the \
                         server applies the model's own template (clear it with \
                         `colgrep settings --agent-chat-template default`)"
                    );
                }
                return Ok(Engine {
                    model: Box::new(OpenAiChat::new(endpoint, &model_id, key, timeout)),
                    backend: Backend::Chat,
                    description: format!("{model_id} @ {endpoint} (chat)"),
                });
            }
            EndpointApi::Completions => {
                let source = settings
                    .chat_template_override()?
                    .unwrap_or_else(|| protocol::DEFAULT_CHAT_TEMPLATE.to_string());
                let template = ChatTemplate::new(&source, "<s>", "</s>")
                    .map_err(|e| format!("chat template: {e}"))?;
                return Ok(Engine {
                    model: Box::new(PromptModel::new(
                        template,
                        Box::new(OpenAiCompletions::new(
                            endpoint, &model_id, "<s>", key, timeout,
                        )),
                    )),
                    backend: Backend::Completions,
                    description: format!("{model_id} @ {endpoint} (completions)"),
                });
            }
        }
    }
    if settings.endpoint_api == Some(EndpointApi::Chat) {
        return Err("the chat backend needs an endpoint: set one with \
             `colgrep settings --agent-endpoint http://host:port/v1`"
            .into());
    }
    let template_override = settings.chat_template_override()?;
    load_local(settings, template_override, force_cpu, progress)
}

/// Start initializing the local engine in the background when the settings will use
/// the built-in llama.cpp: on Metal this compiles the GPU kernels, which takes ~20 s
/// when macOS's shader cache misses. Call it as early as possible.
///
/// `kernels_dir` is where pre-compiled Metal kernels are kept (next to the colgrep
/// indices); with them, initialization takes milliseconds instead.
pub fn prewarm(settings: &AgentSettings, kernels_dir: Option<&std::path::Path>) {
    #[cfg(builtin_llama)]
    if settings.endpoint.is_none() && settings.runtime.as_deref().unwrap_or("auto") != "server" {
        if let Some(base) = kernels_dir {
            use_precompiled_kernels(base);
        }
        crate::llm::llama::prewarm_backend();
    }
    #[cfg(not(builtin_llama))]
    let _ = (settings, kernels_dir);
}

/// Point ggml at the pre-compiled Metal kernels, installing them under `base` first.
/// A `GGML_METAL_LIB_DIR` the user set wins. Must run before the backend initializes.
#[cfg(builtin_llama)]
fn use_precompiled_kernels(base: &std::path::Path) {
    if !crate::metal::AVAILABLE || std::env::var_os("GGML_METAL_LIB_DIR").is_some() {
        return;
    }
    let t = std::time::Instant::now();
    match crate::metal::install(base) {
        // Set before any thread reads the environment (the backend starts after this).
        Ok(dir) => std::env::set_var("GGML_METAL_LIB_DIR", dir),
        // Not fatal: ggml compiles the kernels from source instead.
        Err(e) => {
            if crate::profile::enabled() {
                eprintln!("  load · pre-compiled Metal kernels unavailable: {e}");
            }
        }
    }
    crate::profile::step("install pre-compiled Metal kernels", t);
}

/// Whether the engine's one-time initialization (GPU kernels) is done. Always true
/// for engines that have none.
pub fn engine_ready() -> bool {
    #[cfg(builtin_llama)]
    return crate::llm::llama::backend_ready();
    #[cfg(not(builtin_llama))]
    true
}

/// Download whatever the local engine still needs (the model weights; on platforms
/// without built-in llama.cpp, the llama.cpp runtime), with progress bars when
/// `progress` is set. Call it before showing any spinner, so the bars draw cleanly;
/// [`load_engine`] then finds everything in the caches. A no-op for endpoints.
pub fn prepare(settings: &AgentSettings, progress: bool) -> Result<(), String> {
    if settings.endpoint.is_some() {
        return Ok(());
    }
    #[cfg(feature = "local")]
    {
        crate::model::resolve_model_file(settings, progress)?;
        let builtin = match settings.runtime.as_deref().unwrap_or("auto") {
            "server" => false,
            _ => BUILTIN_AVAILABLE,
        };
        if !builtin {
            crate::runtime::llama_server(settings.llama_server.as_deref(), progress)?;
        }
    }
    #[cfg(not(feature = "local"))]
    let _ = progress;
    Ok(())
}

/// Whether `colgrep --agent` can answer right away: an endpoint is configured, or the
/// model is already downloaded (on disk, no network check).
pub fn agent_ready(settings: &AgentSettings) -> bool {
    settings.endpoint.is_some() || crate::model::installed_model_file(settings).is_some()
}

/// Whether llama.cpp is compiled into this build (Apple Silicon).
pub const BUILTIN_AVAILABLE: bool = cfg!(builtin_llama);

#[cfg(feature = "local")]
fn load_local(
    settings: &AgentSettings,
    template_override: Option<String>,
    force_cpu: bool,
    progress: bool,
) -> Result<Engine, String> {
    let builtin = match settings.runtime.as_deref().unwrap_or("auto") {
        "auto" => BUILTIN_AVAILABLE,
        "builtin" if BUILTIN_AVAILABLE => true,
        "builtin" => {
            return Err(
                "the built-in engine is only compiled on Apple Silicon; use \
                 `--agent-runtime server` (or auto)"
                    .into(),
            )
        }
        "server" => false,
        other => {
            return Err(format!(
                "unknown agent runtime: {other} (auto, builtin, server)"
            ))
        }
    };
    let t = std::time::Instant::now();
    let path = crate::model::resolve_model_file(settings, progress)?;
    crate::profile::step("resolve model file", t);
    // Request every layer on the GPU; llama.cpp keeps them on the CPU when it finds none.
    let gpu_layers = if force_cpu {
        0
    } else {
        settings.gpu_layers.unwrap_or(999)
    };
    #[cfg(builtin_llama)]
    if builtin {
        return load_builtin(settings, &path, template_override, gpu_layers);
    }
    // Without llama.cpp compiled in, `builtin` was rejected above.
    #[cfg(not(builtin_llama))]
    debug_assert!(!builtin);
    load_server(settings, &path, template_override, gpu_layers, progress)
}

#[cfg(feature = "local")]
fn load_server(
    settings: &AgentSettings,
    path: &std::path::Path,
    template_override: Option<String>,
    gpu_layers: u32,
    progress: bool,
) -> Result<Engine, String> {
    use crate::llm::server::{LlamaServer, ServerOptions};

    let bin = crate::runtime::llama_server(settings.llama_server.as_deref(), progress)?;
    let server = LlamaServer::start(
        &bin,
        path,
        &ServerOptions {
            context_size: settings.context_size(),
            threads: Some(settings.threads.unwrap_or_else(server_threads)),
            gpu_layers,
            progress,
        },
    )?;
    let props = server.props.clone();
    let source = template_override
        .or(props.chat_template)
        .unwrap_or_else(|| protocol::DEFAULT_CHAT_TEMPLATE.to_string());
    let bos = if props.bos_token.is_empty() {
        "<s>".to_string()
    } else {
        props.bos_token
    };
    let eos = if props.eos_token.is_empty() {
        "</s>".to_string()
    } else {
        props.eos_token
    };
    let template =
        ChatTemplate::new(&source, &bos, &eos).map_err(|e| format!("chat template: {e}"))?;
    let description = format!(
        "{} (llama.cpp {}, {})",
        file_name(path),
        crate::runtime::LLAMA_CPP_RELEASE,
        server.device
    );
    Ok(Engine {
        model: Box::new(PromptModel::new(template, Box::new(server))),
        backend: Backend::Server,
        description,
    })
}

/// CPU threads for llama-server. Left to itself it takes every core, and its threads
/// spin-wait on each other: one preempted thread stalls the rest, so on a busy 64-core
/// server generation fell from ~63 tok/s (24-56 threads) to 0.5 tok/s (64 threads).
/// Prefill (most of an agent turn) keeps scaling with cores, so take as many as is safe:
/// keep a quarter of the cores free (honouring cgroup / affinity limits) and cap the count
/// (48 threads: 30 s per question vs 40 s at 32 and 58 s at 16 on that machine).
#[cfg(feature = "local")]
fn server_threads() -> usize {
    let available = std::thread::available_parallelism().map_or(1, |n| n.get());
    let cores = num_cpus::get_physical().min(available).max(1);
    (cores - cores / 4).clamp(1, 48)
}

#[cfg(builtin_llama)]
fn load_builtin(
    settings: &AgentSettings,
    path: &std::path::Path,
    template_override: Option<String>,
    gpu_layers: u32,
) -> Result<Engine, String> {
    use crate::llm::llama::{LlamaEngine, LlamaOptions};

    let threads = settings.threads.unwrap_or_else(default_threads);
    let engine = LlamaEngine::load(
        path,
        &LlamaOptions {
            context_size: settings.context_size(),
            threads: Some(threads),
            gpu_layers,
            prefix_cache_dir: std::env::var_os("COLGREP_AGENT_NO_PREFIX_CACHE")
                .is_none()
                .then(|| dirs::cache_dir().map(|d| d.join("colgrep").join("agent-prefix")))
                .flatten(),
        },
    )
    .map_err(|e| e.to_string())?;
    let (bos, eos) = engine.special_tokens();
    let source = template_override
        .or_else(|| engine.embedded_chat_template())
        .unwrap_or_else(|| protocol::DEFAULT_CHAT_TEMPLATE.to_string());
    let template =
        ChatTemplate::new(&source, &bos, &eos).map_err(|e| format!("chat template: {e}"))?;
    let device = if gpu_layers > 0 {
        "GPU".to_string()
    } else {
        format!("CPU, {threads} threads")
    };
    Ok(Engine {
        model: Box::new(PromptModel::new(template, Box::new(engine))),
        backend: Backend::Builtin,
        description: format!("{} ({device})", file_name(path)),
    })
}

#[cfg(feature = "local")]
fn file_name(path: &std::path::Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// CPU threads for local inference: the performance cores. Generation is memory-bound,
/// and threads on efficiency cores (or SMT siblings) slow every token down to their
/// pace — on an M3 Pro, 5 threads generate ~40 tok/s and 11 threads ~10 tok/s.
#[cfg(builtin_llama)]
fn default_threads() -> usize {
    macos_performance_cores().unwrap_or_else(|| num_cpus::get_physical().max(1))
}

#[cfg(builtin_llama)]
fn macos_performance_cores() -> Option<usize> {
    let mut value: i32 = 0;
    let mut size = std::mem::size_of::<i32>();
    let name = c"hw.perflevel0.physicalcpu";
    // SAFETY: valid NUL-terminated name, out-pointer sized for an i32.
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            (&mut value as *mut i32).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0 && value > 0).then_some(value as usize)
}

#[cfg(not(feature = "local"))]
fn load_local(
    _settings: &AgentSettings,
    _template_override: Option<String>,
    _force_cpu: bool,
    _progress: bool,
) -> Result<Engine, String> {
    Err(
        "this build has no local inference; point the agent at a server with \
         `colgrep settings --agent-endpoint http://host:port/v1`"
            .into(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AgentSettings, EndpointApi};

    #[test]
    fn a_chat_endpoint_selects_the_chat_backend() {
        let s = AgentSettings {
            endpoint: Some("http://127.0.0.1:9/v1".into()),
            endpoint_api: Some(EndpointApi::Chat),
            ..Default::default()
        };
        let engine = load_engine(&s, false, false).unwrap();
        assert_eq!(engine.backend, Backend::Chat);
        assert!(engine.description.ends_with("(chat)"));
    }

    #[test]
    fn an_endpoint_without_an_api_keeps_the_completions_backend() {
        let s = AgentSettings {
            endpoint: Some("http://127.0.0.1:9/v1".into()),
            ..Default::default()
        };
        let engine = load_engine(&s, false, false).unwrap();
        assert_eq!(engine.backend, Backend::Completions);
        assert!(engine.description.ends_with("(completions)"));
    }

    #[test]
    fn the_chat_backend_does_not_read_the_ignored_template() {
        let s = AgentSettings {
            endpoint: Some("http://127.0.0.1:9/v1".into()),
            endpoint_api: Some(EndpointApi::Chat),
            chat_template_file: Some("/definitely/missing.jinja".into()),
            ..Default::default()
        };
        assert_eq!(
            load_engine(&s, false, false).unwrap().backend,
            Backend::Chat
        );
    }

    #[test]
    fn the_chat_backend_requires_an_endpoint() {
        let s = AgentSettings {
            endpoint_api: Some(EndpointApi::Chat),
            ..Default::default()
        };
        match load_engine(&s, false, false) {
            Err(err) => assert!(err.contains("needs an endpoint"), "{err}"),
            Ok(_) => panic!("the chat backend must not load without an endpoint"),
        }
    }
}
