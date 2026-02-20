use serde::Serialize;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Serialize)]
struct CommandResult {
    ok: bool,
    command: String,
    stdout: String,
    stderr: String,
    exit_code: i32,
}

#[derive(Serialize)]
struct ModelListResult {
    ok: bool,
    models: Vec<String>,
    raw_stdout: String,
    stderr: String,
}

fn find_tool_root() -> PathBuf {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    for dir in cwd.ancestors() {
        if is_tool_root(dir) {
            return dir.to_path_buf();
        }
    }

    let home = std::env::var("HOME").unwrap_or_default();
    if !home.is_empty() {
        let candidate = PathBuf::from(home).join("Tools").join("Draftenheimer");
        if is_tool_root(&candidate) {
            return candidate;
        }
    }

    cwd
}

fn is_tool_root(path: &Path) -> bool {
    path.join("qa_scan.py").exists() && path.join("draftenheimer").exists()
}

fn command_result(command_path: &Path, args: &[String], output: std::process::Output) -> CommandResult {
    let rendered = format!(
        "{} {}",
        command_path.display(),
        args.iter()
            .map(|x| shell_escape::escape(x.as_str().into()).to_string())
            .collect::<Vec<_>>()
            .join(" ")
    );

    CommandResult {
        ok: output.status.success(),
        command: rendered,
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        exit_code: output.status.code().unwrap_or(-1),
    }
}

fn run_bin(tool_dir: &str, program: PathBuf, args: &[String]) -> Result<CommandResult, String> {
    let mut cmd = Command::new(program.clone());
    cmd.current_dir(tool_dir);
    cmd.args(args);
    let output = cmd
        .output()
        .map_err(|e| format!("Failed to run command {}: {e}", program.display()))?;
    Ok(command_result(&program, args, output))
}

fn run_python(tool_dir: &str, script_name: &str, args: &[String]) -> Result<CommandResult, String> {
    let script = PathBuf::from(tool_dir).join(script_name);
    if !script.exists() {
        return Err(format!("Missing script: {}", script.display()));
    }

    let mut full_args = vec![script.display().to_string()];
    full_args.extend(args.iter().cloned());
    run_bin(tool_dir, PathBuf::from("python3"), &full_args)
}

fn run_draftenheimer(tool_dir: &str, args: &[String]) -> Result<CommandResult, String> {
    let script = PathBuf::from(tool_dir).join("draftenheimer");
    if !script.exists() {
        return Err(format!("Missing draftenheimer wrapper: {}", script.display()));
    }
    run_bin(tool_dir, script, args)
}

fn parse_models_from_output(raw: &str) -> Vec<String> {
    let mut set = BTreeSet::new();
    for line in raw.lines() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }

        let upper = t.to_ascii_uppercase();
        if upper.starts_with("NAME ") || upper == "NAME" {
            continue;
        }

        let first = t.split_whitespace().next().unwrap_or("").trim();
        if first.is_empty() {
            continue;
        }
        if first.contains(':') || first.contains('/') || first.contains('-') || first.contains('.') {
            set.insert(first.to_string());
        }
    }
    set.into_iter().collect()
}

#[tauri::command]
fn default_tool_dir() -> String {
    find_tool_root().display().to_string()
}


#[tauri::command]
fn pick_docx_file() -> Option<String> {
    rfd::FileDialog::new()
        .set_title("Select DOCX File")
        .add_filter("Word Document", &["docx"])
        .pick_file()
        .map(|p| p.display().to_string())
}

#[tauri::command]
fn pick_json_save_path(default_name: Option<String>) -> Option<String> {
    let mut dialog = rfd::FileDialog::new().set_title("Save JSON Output").add_filter("JSON", &["json"]);
    if let Some(name) = default_name {
        if !name.trim().is_empty() {
            dialog = dialog.set_file_name(name.trim());
        }
    }
    dialog.save_file().map(|p| p.display().to_string())
}

#[tauri::command]
fn pick_docx_save_path(default_name: Option<String>) -> Option<String> {
    let mut dialog = rfd::FileDialog::new().set_title("Save Annotated DOCX").add_filter("Word Document", &["docx"]);
    if let Some(name) = default_name {
        if !name.trim().is_empty() {
            dialog = dialog.set_file_name(name.trim());
        }
    }
    dialog.save_file().map(|p| p.display().to_string())
}

#[tauri::command]
fn pick_folder_path() -> Option<String> {
    rfd::FileDialog::new()
        .set_title("Select Folder")
        .pick_folder()
        .map(|p| p.display().to_string())
}

#[tauri::command]
fn pick_any_file() -> Option<String> {
    rfd::FileDialog::new()
        .set_title("Select File")
        .pick_file()
        .map(|p| p.display().to_string())
}

#[tauri::command]
fn list_local_models(tool_dir: String, ollama_url: Option<String>) -> Result<ModelListResult, String> {
    let mut args = Vec::new();
    if let Some(url) = ollama_url {
        if !url.trim().is_empty() {
            args.push("--ollama-url".to_string());
            args.push(url);
        }
    }
    args.push("list".to_string());

    let res = run_python(&tool_dir, "qa_models.py", &args)?;
    Ok(ModelListResult {
        ok: res.ok,
        models: parse_models_from_output(&res.stdout),
        raw_stdout: res.stdout,
        stderr: res.stderr,
    })
}

#[tauri::command]
fn pull_local_model(tool_dir: String, model: String, ollama_url: Option<String>) -> Result<CommandResult, String> {
    if model.trim().is_empty() {
        return Err("model is required".into());
    }
    let mut args = Vec::new();
    if let Some(url) = ollama_url {
        if !url.trim().is_empty() {
            args.push("--ollama-url".to_string());
            args.push(url);
        }
    }
    args.push("pull".to_string());
    args.push(model);
    run_python(&tool_dir, "qa_models.py", &args)
}

#[tauri::command]
fn start_local_model_runtime(tool_dir: String) -> Result<CommandResult, String> {
    run_bin(&tool_dir, PathBuf::from(tool_dir.clone()).join("slm_start.sh"), &[])
}

#[tauri::command]
fn stop_local_model_runtime(tool_dir: String, full: bool) -> Result<CommandResult, String> {
    let mut args = Vec::new();
    if full {
        args.push("--full".to_string());
    }
    run_bin(&tool_dir, PathBuf::from(tool_dir.clone()).join("slm_stop.sh"), &args)
}

#[tauri::command]
async fn rebuild_learning_profile(
    tool_dir: String,
    reports_dir: Option<String>,
    pair_mode: Option<String>,
    track_changes: Option<bool>,
    track_weight: Option<i32>,
    ai_compare: Option<bool>,
    provider: Option<String>,
    model: Option<String>,
    ollama_url: Option<String>,
    bedrock_region: Option<String>,
    bedrock_profile: Option<String>,
) -> Result<CommandResult, String> {
    let mut args: Vec<String> = Vec::new();

    if let Some(v) = reports_dir {
        if !v.trim().is_empty() {
            args.push("--reports-dir".into());
            args.push(v);
        }
    }

    if let Some(v) = pair_mode {
        let v = v.trim().to_ascii_lowercase();
        if v == "consecutive" || v == "latest" {
            args.push("--pair-mode".into());
            args.push(v);
        }
    }

    if let Some(v) = track_changes {
        if !v {
            args.push("--no-track-changes".into());
        }
    }

    if let Some(v) = track_weight {
        if v > 0 {
            args.push("--track-weight".into());
            args.push(v.to_string());
        }
    }

    let use_ai_compare = ai_compare.unwrap_or(false);
    if use_ai_compare {
        let provider_name = provider.unwrap_or_else(|| "ollama".to_string()).to_ascii_lowercase();
        args.push("--ai-compare".into());
        args.push("--ai-provider".into());
        args.push(provider_name.clone());

        let model_name = model.unwrap_or_default();
        if model_name.trim().is_empty() {
            return Err("AI Compare requires a model".into());
        }
        args.push("--ai-model".into());
        args.push(model_name);

        if provider_name == "bedrock" {
            if let Some(v) = bedrock_region {
                if !v.trim().is_empty() {
                    args.push("--bedrock-region".into());
                    args.push(v);
                }
            }
            if let Some(v) = bedrock_profile {
                if !v.trim().is_empty() {
                    args.push("--bedrock-profile".into());
                    args.push(v);
                }
            }
        } else if let Some(v) = ollama_url {
            if !v.trim().is_empty() {
                args.push("--ollama-url".into());
                args.push(v);
            }
        }
    }

    tauri::async_runtime::spawn_blocking(move || run_python(&tool_dir, "build_learned_profile.py", &args))
        .await
        .map_err(|e| format!("Rebuild task join error: {e}"))?
}

#[allow(clippy::too_many_arguments)]
#[tauri::command]
async fn run_scan(
    tool_dir: String,
    docx_path: String,
    provider: String,
    model: String,
    ollama_url: Option<String>,
    bedrock_region: Option<String>,
    bedrock_profile: Option<String>,
    llm: bool,
    llm_pull: bool,
    auto_learn: bool,
    auto_learn_ai: bool,
    annotate: bool,
    json_out: Option<String>,
    annotate_out: Option<String>,
    reports_dir: Option<String>,
    learn_pair_mode: Option<String>,
    learn_track_changes: Option<bool>,
    learn_track_weight: Option<i32>,
    ignore_config: Option<String>,
    feedback_config: Option<String>,
) -> Result<CommandResult, String> {
    if docx_path.trim().is_empty() {
        return Err("docx_path is required".into());
    }

    let mut args: Vec<String> = vec![docx_path];

    if llm {
        args.push("--llm".into());
    }
    if llm_pull {
        args.push("--llm-pull".into());
    }

    args.push("--provider".into());
    args.push(provider.clone());

    if !model.trim().is_empty() {
        args.push("--model".into());
        args.push(model);
    }

    if let Some(v) = ollama_url {
        if !v.trim().is_empty() {
            args.push("--ollama-url".into());
            args.push(v);
        }
    }
    if let Some(v) = bedrock_region {
        if !v.trim().is_empty() {
            args.push("--bedrock-region".into());
            args.push(v);
        }
    }
    if let Some(v) = bedrock_profile {
        if !v.trim().is_empty() {
            args.push("--bedrock-profile".into());
            args.push(v);
        }
    }

    if auto_learn {
        args.push("--auto-learn".into());
    }
    if auto_learn_ai {
        args.push("--auto-learn-ai".into());
    }

    if annotate {
        args.push("--annotate".into());
    }

    if let Some(v) = json_out {
        if !v.trim().is_empty() {
            args.push("--json-out".into());
            args.push(v);
        }
    }
    if let Some(v) = annotate_out {
        if !v.trim().is_empty() {
            args.push("--annotate-out".into());
            args.push(v);
        }
    }
    if let Some(v) = reports_dir {
        if !v.trim().is_empty() {
            args.push("--reports-dir".into());
            args.push(v);
        }
    }
    if let Some(v) = learn_pair_mode {
        let v = v.trim().to_ascii_lowercase();
        if v == "consecutive" || v == "latest" {
            args.push("--learn-pair-mode".into());
            args.push(v);
        }
    }
    if let Some(v) = learn_track_changes {
        if !v {
            args.push("--learn-no-track-changes".into());
        }
    }
    if let Some(v) = learn_track_weight {
        if v > 0 {
            args.push("--learn-track-weight".into());
            args.push(v.to_string());
        }
    }
    if let Some(v) = ignore_config {
        if !v.trim().is_empty() {
            args.push("--ignore-config".into());
            args.push(v);
        }
    }
    if let Some(v) = feedback_config {
        if !v.trim().is_empty() {
            args.push("--feedback-config".into());
            args.push(v);
        }
    }

    tauri::async_runtime::spawn_blocking(move || run_draftenheimer(&tool_dir, &args))
        .await
        .map_err(|e| format!("Scan task join error: {e}"))?
}

#[tauri::command]
async fn import_feedback_docx(
    tool_dir: String,
    feedback_docx: String,
    feedback_author: Option<String>,
    dry_run: bool,
) -> Result<CommandResult, String> {
    if feedback_docx.trim().is_empty() {
        return Err("feedback_docx is required".into());
    }

    let mut args: Vec<String> = vec!["--import-feedback-docx".into(), feedback_docx];
    if let Some(author) = feedback_author {
        args.push("--feedback-author".into());
        args.push(author);
    }
    if dry_run {
        args.push("--feedback-dry-run".into());
    }

    tauri::async_runtime::spawn_blocking(move || run_draftenheimer(&tool_dir, &args))
        .await
        .map_err(|e| format!("Import task join error: {e}"))?
}


#[tauri::command]
fn open_file_path(path: String) -> Result<(), String> {
    if path.trim().is_empty() {
        return Err("path is required".into());
    }
    let p = PathBuf::from(path.trim());
    if !p.exists() {
        return Err(format!("Path does not exist: {}", p.display()));
    }

    #[cfg(target_os = "macos")]
    {
        Command::new("open")
            .arg(&p)
            .status()
            .map_err(|e| format!("Failed to open file: {e}"))?;
    }

    #[cfg(target_os = "windows")]
    {
        Command::new("cmd")
            .args(["/C", "start", "", &p.display().to_string()])
            .status()
            .map_err(|e| format!("Failed to open file: {e}"))?;
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        Command::new("xdg-open")
            .arg(&p)
            .status()
            .map_err(|e| format!("Failed to open file: {e}"))?;
    }

    Ok(())
}

#[tauri::command]
fn reveal_in_file_manager(path: String) -> Result<(), String> {
    if path.trim().is_empty() {
        return Err("path is required".into());
    }
    let p = PathBuf::from(path.trim());
    if !p.exists() {
        return Err(format!("Path does not exist: {}", p.display()));
    }

    #[cfg(target_os = "macos")]
    {
        Command::new("open")
            .args(["-R", &p.display().to_string()])
            .status()
            .map_err(|e| format!("Failed to reveal file: {e}"))?;
    }

    #[cfg(target_os = "windows")]
    {
        Command::new("explorer")
            .args(["/select,", &p.display().to_string()])
            .status()
            .map_err(|e| format!("Failed to reveal file: {e}"))?;
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let parent = p.parent().unwrap_or(Path::new("."));
        Command::new("xdg-open")
            .arg(parent)
            .status()
            .map_err(|e| format!("Failed to open folder: {e}"))?;
    }

    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_window_state::Builder::default().build())
        .invoke_handler(tauri::generate_handler![
            default_tool_dir,
            pick_docx_file,
            pick_json_save_path,
            pick_docx_save_path,
            pick_folder_path,
            pick_any_file,
            list_local_models,
            pull_local_model,
            start_local_model_runtime,
            stop_local_model_runtime,
            rebuild_learning_profile,
            run_scan,
            import_feedback_docx,
            open_file_path,
            reveal_in_file_manager,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
