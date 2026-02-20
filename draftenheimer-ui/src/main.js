const { invoke } = window.__TAURI__.core;
const dialogApi = window.__TAURI__.dialog || {};
const dialogOpen = dialogApi.open;
const dialogSave = dialogApi.save;

const state = {
  toolDir: "",
  busyCount: 0,
};

const SETTINGS_KEY_PREFIX = "draftenheimer.settings.";

const el = {};
const $ = (id) => document.getElementById(id);

function bind() {
  el.openSettingsBtn = $("open-settings-btn");
  el.closeSettingsBtn = $("close-settings-btn");
  el.settingsDrawer = $("settings-drawer");

  el.themeSelect = $("theme-select");
  el.statusPill = $("status-pill");
  el.statusText = $("status-text");
  el.brandSymbol = $("brand-symbol");
  el.brandWordmark = $("brand-wordmark");

  el.scanDocx = $("scan-docx");
  el.pickScanDocxBtn = $("pick-scan-docx-btn");
  el.scanJsonOut = $("scan-json-out");
  el.pickJsonOutBtn = $("pick-json-out-btn");
  el.scanAnnotateOut = $("scan-annotate-out");
  el.pickAnnotateOutBtn = $("pick-annotate-out-btn");

  el.scanLlm = $("scan-llm");
  el.scanLlmPull = $("scan-llm-pull");
  el.scanAutoLearn = $("scan-auto-learn");
  el.scanAutoLearnAi = $("scan-auto-learn-ai");
  el.scanAnnotate = $("scan-annotate");
  el.scanOpenOutput = $("scan-open-output");
  el.scanRevealOutput = $("scan-reveal-output");
  el.runScanBtn = $("run-scan-btn");
  el.runScanStatus = $("run-scan-status");
  el.runScanStatusText = $("run-scan-status-text");

  el.scanProvider = $("scan-provider");
  el.scanModelSelect = $("scan-model-select");
  el.scanModelCustom = $("scan-model-custom");
  el.refreshModelsBtn = $("refresh-models-btn");
  el.pullModelBtn = $("pull-model-btn");
  el.runtimeStartBtn = $("runtime-start-btn");
  el.runtimeStopBtn = $("runtime-stop-btn");
  el.runtimeStopFullBtn = $("runtime-stop-full-btn");

  el.ollamaUrl = $("ollama-url");
  el.bedrockRegion = $("bedrock-region");
  el.bedrockProfile = $("bedrock-profile");
  el.reportsDir = $("reports-dir");
  el.pickReportsDirBtn = $("pick-reports-dir-btn");
  el.learnPairMode = $("learn-pair-mode");
  el.learnTrackWeight = $("learn-track-weight");
  el.learnTrackChanges = $("learn-track-changes");
  el.rebuildAiCompare = $("rebuild-ai-compare");
  el.rebuildLearningBtn = $("rebuild-learning-btn");
  el.rebuildLearningStatus = $("rebuild-learning-status");
  el.rebuildLearningStatusText = $("rebuild-learning-status-text");
  el.ignoreConfig = $("ignore-config");
  el.pickIgnoreConfigBtn = $("pick-ignore-config-btn");
  el.feedbackConfig = $("feedback-config");
  el.pickFeedbackConfigBtn = $("pick-feedback-config-btn");

  el.feedbackDocx = $("feedback-docx");
  el.pickFeedbackDocxBtn = $("pick-feedback-docx-btn");
  el.feedbackAuthor = $("feedback-author");
  el.feedbackDryRun = $("feedback-dry-run");
  el.importFeedbackBtn = $("import-feedback-btn");

  el.output = $("output");
}

function openSettings() {
  el.settingsDrawer.classList.remove("hidden");
  el.settingsDrawer.setAttribute("aria-hidden", "false");
}

function closeSettings() {
  el.settingsDrawer.classList.add("hidden");
  el.settingsDrawer.setAttribute("aria-hidden", "true");
}

function currentResolvedTheme(themeChoice) {
  if (themeChoice === "system") {
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  return themeChoice;
}

function updateImageForTheme(img, defaultSrc, darkSrc, themeChoice) {
  if (!img) return;
  const resolved = currentResolvedTheme(themeChoice);
  if (resolved === "dark") {
    img.dataset.fallback = defaultSrc;
    img.src = darkSrc;
  } else {
    img.dataset.fallback = "";
    img.src = defaultSrc;
  }
}

function updateBrandForTheme(themeChoice) {
  updateImageForTheme(
    el.brandWordmark,
    "/assets/draftenheimer-wordmark.png",
    "/assets/draftenheimer-wordmark-dark.png",
    themeChoice,
  );
  updateImageForTheme(
    el.brandSymbol,
    "/assets/draftenheimer-symbol.png",
    "/assets/draftenheimer-symbol-dark.png",
    themeChoice,
  );
}

function setTheme(theme) {
  localStorage.setItem("draftenheimer.theme", theme);
  if (theme === "system") {
    document.documentElement.removeAttribute("data-theme");
  } else {
    document.documentElement.setAttribute("data-theme", theme);
  }
  updateBrandForTheme(theme);
}

function initTheme() {
  const saved = localStorage.getItem("draftenheimer.theme") || "system";
  el.themeSelect.value = saved;
  setTheme(saved);
  el.themeSelect.addEventListener("change", () => setTheme(el.themeSelect.value));
}

function settingKey(name) {
  return `${SETTINGS_KEY_PREFIX}${name}`;
}

function saveSetting(name, value) {
  localStorage.setItem(settingKey(name), String(value ?? ""));
}

function loadSetting(name, fallback = "") {
  const v = localStorage.getItem(settingKey(name));
  return v == null ? fallback : v;
}

function loadBoolSetting(name, fallback = false) {
  const v = localStorage.getItem(settingKey(name));
  if (v == null) return fallback;
  return v === "1" || v === "true";
}

function restorePersistedSettings() {
  el.ollamaUrl.value = loadSetting("ollama_url", el.ollamaUrl.value || "http://localhost:11434");
  el.bedrockRegion.value = loadSetting("bedrock_region", el.bedrockRegion.value || "us-east-1");
  el.bedrockProfile.value = loadSetting("bedrock_profile", el.bedrockProfile.value || "sci_bedrock");

  el.reportsDir.value = loadSetting("reports_dir", el.reportsDir.value || "");
  el.ignoreConfig.value = loadSetting("ignore_config", el.ignoreConfig.value || "");
  el.feedbackConfig.value = loadSetting("feedback_config", el.feedbackConfig.value || "");

  el.learnPairMode.value = loadSetting("learn_pair_mode", el.learnPairMode.value || "consecutive");
  el.learnTrackWeight.value = loadSetting("learn_track_weight", el.learnTrackWeight.value || "2");
  el.learnTrackChanges.checked = loadBoolSetting("learn_track_changes", true);
  el.rebuildAiCompare.checked = loadBoolSetting("rebuild_ai_compare", false);

  el.scanProvider.value = loadSetting("scan_provider", el.scanProvider.value || "ollama");
  el.scanModelCustom.value = loadSetting("scan_model_custom", el.scanModelCustom.value || "");

  el.scanLlm.checked = loadBoolSetting("scan_llm", true);
  el.scanLlmPull.checked = loadBoolSetting("scan_llm_pull", false);
  el.scanAutoLearn.checked = loadBoolSetting("scan_auto_learn", true);
  el.scanAutoLearnAi.checked = loadBoolSetting("scan_auto_learn_ai", false);
  el.scanAnnotate.checked = loadBoolSetting("scan_annotate", true);
  el.scanOpenOutput.checked = loadBoolSetting("scan_open_output", false);
  el.scanRevealOutput.checked = loadBoolSetting("scan_reveal_output", false);
}

function wireSettingsPersistence() {
  const textFields = [
    [el.ollamaUrl, "ollama_url"],
    [el.bedrockRegion, "bedrock_region"],
    [el.bedrockProfile, "bedrock_profile"],
    [el.reportsDir, "reports_dir"],
    [el.ignoreConfig, "ignore_config"],
    [el.feedbackConfig, "feedback_config"],
    [el.scanModelCustom, "scan_model_custom"],
  ];
  for (const [node, key] of textFields) {
    node.addEventListener("change", () => saveSetting(key, node.value.trim()));
  }

  const selectFields = [
    [el.learnPairMode, "learn_pair_mode"],
    [el.learnTrackWeight, "learn_track_weight"],
    [el.scanProvider, "scan_provider"],
  ];
  for (const [node, key] of selectFields) {
    node.addEventListener("change", () => saveSetting(key, node.value));
  }

  const checkFields = [
    [el.learnTrackChanges, "learn_track_changes"],
    [el.rebuildAiCompare, "rebuild_ai_compare"],
    [el.scanLlm, "scan_llm"],
    [el.scanLlmPull, "scan_llm_pull"],
    [el.scanAutoLearn, "scan_auto_learn"],
    [el.scanAutoLearnAi, "scan_auto_learn_ai"],
    [el.scanAnnotate, "scan_annotate"],
    [el.scanOpenOutput, "scan_open_output"],
    [el.scanRevealOutput, "scan_reveal_output"],
  ];
  for (const [node, key] of checkFields) {
    node.addEventListener("change", () => saveSetting(key, node.checked ? "1" : "0"));
  }
}

function initBrandFallbacks() {
  const hideImage = (img) => {
    if (!img) return;
    img.classList.add("hidden");
  };

  if (el.brandSymbol) {
    el.brandSymbol.addEventListener("error", () => hideImage(el.brandSymbol));
  }

  if (el.brandWordmark) {
    el.brandWordmark.addEventListener("error", () => {
      const fallback = el.brandWordmark.dataset.fallback;
      if (fallback && el.brandWordmark.src.indexOf(fallback) === -1) {
        el.brandWordmark.dataset.fallback = "";
        el.brandWordmark.src = fallback;
        return;
      }
      hideImage(el.brandWordmark);
    });
  }
}

function startBusy(label = "Working...") {
  state.busyCount += 1;
  el.statusPill.classList.remove("idle", "success", "error");
  el.statusPill.classList.add("running");
  el.statusText.textContent = label;
}

function endBusy(ok = true, label = null) {
  state.busyCount = Math.max(0, state.busyCount - 1);
  if (state.busyCount > 0) return;

  el.statusPill.classList.remove("running", "idle", "success", "error");
  if (ok) {
    el.statusPill.classList.add("success");
    el.statusText.textContent = label || "Completed";
  } else {
    el.statusPill.classList.add("error");
    el.statusText.textContent = label || "Failed";
  }
}

function setRunScanStatus(stateName, text) {
  el.runScanStatus.classList.remove("idle", "running", "success", "error");
  el.runScanStatus.classList.add(stateName);
  el.runScanStatusText.textContent = text;
}

function setRebuildLearningStatus(stateName, text) {
  el.rebuildLearningStatus.classList.remove("idle", "running", "success", "error");
  el.rebuildLearningStatus.classList.add(stateName);
  el.rebuildLearningStatusText.textContent = text;
}

async function flushUiFrame() {
  await new Promise((resolve) => requestAnimationFrame(() => resolve()));
  await new Promise((resolve) => setTimeout(resolve, 0));
}

function commandText(res) {
  const parts = [
    `ok: ${res.ok}`,
    `exit_code: ${res.exit_code}`,
    `command: ${res.command}`,
  ];
  if (res.stdout?.trim()) {
    parts.push("\n--- stdout ---");
    parts.push(res.stdout.trim());
  }
  if (res.stderr?.trim()) {
    parts.push("\n--- stderr ---");
    parts.push(res.stderr.trim());
  }
  return parts.join("\n");
}

function print(title, body) {
  const now = new Date().toLocaleTimeString();
  el.output.textContent = `[${now}] ${title}\n${body}`;
}

function setButtonBusy(btn, busy) {
  btn.disabled = busy;
  if (busy) {
    btn.dataset.original = btn.textContent;
    btn.textContent = "Working...";
  } else if (btn.dataset.original) {
    btn.textContent = btn.dataset.original;
  }
}

function inferOutputPathFromResult(res) {
  const stdout = res?.stdout || "";
  const lines = stdout.split(/\r?\n/).map((x) => x.trim()).filter(Boolean);

  // If annotate is on and a path is printed at end, prefer it.
  for (let i = lines.length - 1; i >= 0; i -= 1) {
    const line = lines[i];
    if (/\.docx$/i.test(line) || /\.json$/i.test(line)) {
      return line;
    }
  }

  if (el.scanAnnotate.checked && el.scanAnnotateOut.value.trim()) {
    return el.scanAnnotateOut.value.trim();
  }
  if (el.scanJsonOut.value.trim()) {
    return el.scanJsonOut.value.trim();
  }
  return el.scanDocx.value.trim();
}


async function maybeOpenOrRevealOutput(res) {
  if (!res?.ok) return;
  if (!el.scanOpenOutput.checked && !el.scanRevealOutput.checked) return;

  const target = inferOutputPathFromResult(res);
  if (!target) return;

  try {
    if (el.scanOpenOutput.checked) {
      await invoke("open_file_path", { path: target });
    }
    if (el.scanRevealOutput.checked) {
      await invoke("reveal_in_file_manager", { path: target });
    }
  } catch (err) {
    print("Post-Scan Action", `Could not open/reveal output: ${String(err)}
Target: ${target}`);
  }
}

function selectedModel() {
  const custom = el.scanModelCustom.value.trim();
  if (custom) return custom;
  return el.scanModelSelect.value || "";
}

async function detectToolDir() {
  state.toolDir = await invoke("default_tool_dir");
}

async function pickPath(kind, defaultName = null) {
  const viaPlugin = async () => {
    if (kind === "docx-open") {
      return await dialogOpen({
        multiple: false,
        filters: [{ name: "Word Document", extensions: ["docx"] }],
      });
    }
    if (kind === "json-save") {
      return await dialogSave({
        defaultPath: defaultName || "report.qa.json",
        filters: [{ name: "JSON", extensions: ["json"] }],
      });
    }
    if (kind === "docx-save") {
      return await dialogSave({
        defaultPath: defaultName || "report.annotated.docx",
        filters: [{ name: "Word Document", extensions: ["docx"] }],
      });
    }
    if (kind === "folder") {
      return await dialogOpen({ directory: true, multiple: false });
    }
    if (kind === "file") {
      return await dialogOpen({ multiple: false });
    }
    return null;
  };

  const viaRustFallback = async () => {
    if (kind === "docx-open") return await invoke("pick_docx_file");
    if (kind === "json-save") return await invoke("pick_json_save_path", { defaultName: defaultName || "report.qa.json" });
    if (kind === "docx-save") return await invoke("pick_docx_save_path", { defaultName: defaultName || "report.annotated.docx" });
    if (kind === "folder") return await invoke("pick_folder_path");
    if (kind === "file") return await invoke("pick_any_file");
    return null;
  };

  try {
    if (typeof dialogOpen === "function" && typeof dialogSave === "function") {
      const picked = await viaPlugin();
      if (picked) return picked;
    }
    return await viaRustFallback();
  } catch (err) {
    try {
      return await viaRustFallback();
    } catch (fallbackErr) {
      print("Path Picker", `error: ${String(err)} | fallback error: ${String(fallbackErr)}`);
      return null;
    }
  }
}

async function refreshModels() {
  setButtonBusy(el.refreshModelsBtn, true);
  startBusy("Refreshing models...");
  await flushUiFrame();
  try {
    const res = await invoke("list_local_models", {
      toolDir: state.toolDir,
      ollamaUrl: el.ollamaUrl.value.trim() || null,
    });

    el.scanModelSelect.innerHTML = "";
    if (res.models.length === 0) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "No models found";
      el.scanModelSelect.appendChild(opt);
      print("Refresh Models", "No local models found. Start runtime and pull a model.");
      endBusy(false, "No Models Found");
    } else {
      for (const m of res.models) {
        const opt = document.createElement("option");
        opt.value = m;
        opt.textContent = m;
        el.scanModelSelect.appendChild(opt);
      }
      if (!el.scanModelCustom.value.trim()) {
        el.scanModelSelect.value = res.models.includes("qwen2.5:14b") ? "qwen2.5:14b" : res.models[0];
      }
      print("Refresh Models", `Found ${res.models.length} model(s): ${res.models.join(", ")}`);
      endBusy(true, "Model Ready");
    }
  } catch (err) {
    print("Refresh Models", `error: ${String(err)}`);
    endBusy(false, "Model Refresh Failed");
  } finally {
    setButtonBusy(el.refreshModelsBtn, false);
  }
}

async function runRuntimeAction(button, label, invokeName, params = {}) {
  setButtonBusy(button, true);
  startBusy(label);
  await flushUiFrame();
  try {
    const res = await invoke(invokeName, { toolDir: state.toolDir, ...params });
    print(label, commandText(res));
    endBusy(res.ok, res.ok ? "Runtime Updated" : "Runtime Error");
  } catch (err) {
    print(label, `error: ${String(err)}`);
    endBusy(false, "Runtime Error");
  } finally {
    setButtonBusy(button, false);
  }
}

async function pullModel() {
  const model = selectedModel();
  if (!model) {
    print("Pull Model", "No model selected.");
    return;
  }
  setButtonBusy(el.pullModelBtn, true);
  startBusy("Pulling model...");
  await flushUiFrame();
  try {
    const res = await invoke("pull_local_model", {
      toolDir: state.toolDir,
      model,
      ollamaUrl: el.ollamaUrl.value.trim() || null,
    });
    print("Pull Model", commandText(res));
    endBusy(res.ok, res.ok ? "Model Pulled" : "Pull Failed");
    if (res.ok) await refreshModels();
  } catch (err) {
    print("Pull Model", `error: ${String(err)}`);
    endBusy(false, "Pull Failed");
  } finally {
    setButtonBusy(el.pullModelBtn, false);
  }
}

async function runScan() {
  if (!el.scanDocx.value.trim()) {
    print("Run Scan", "Report DOCX is required.");
    setRunScanStatus("error", "Missing report path");
    return;
  }

  setButtonBusy(el.runScanBtn, true);
  setRunScanStatus("running", "Running...");
  startBusy("Running scan...");
  await flushUiFrame();
  try {
    const res = await invoke("run_scan", {
      toolDir: state.toolDir,
      docxPath: el.scanDocx.value.trim(),
      provider: el.scanProvider.value,
      model: selectedModel(),
      ollamaUrl: el.ollamaUrl.value.trim() || null,
      bedrockRegion: el.bedrockRegion.value.trim() || null,
      bedrockProfile: el.bedrockProfile.value.trim() || null,
      llm: !!el.scanLlm.checked,
      llmPull: !!el.scanLlmPull.checked,
      autoLearn: !!el.scanAutoLearn.checked,
      autoLearnAi: !!el.scanAutoLearnAi.checked,
      annotate: !!el.scanAnnotate.checked,
      jsonOut: el.scanJsonOut.value.trim() || null,
      annotateOut: el.scanAnnotateOut.value.trim() || null,
      reportsDir: el.reportsDir.value.trim() || null,
      learnPairMode: el.learnPairMode?.value || "consecutive",
      learnTrackChanges: !!el.learnTrackChanges?.checked,
      learnTrackWeight: Number.parseInt(el.learnTrackWeight?.value || "2", 10) || 2,
      ignoreConfig: el.ignoreConfig.value.trim() || null,
      feedbackConfig: el.feedbackConfig.value.trim() || null,
    });

    print("Run Scan", commandText(res));
    await maybeOpenOrRevealOutput(res);
    setRunScanStatus(res.ok ? "success" : "error", res.ok ? "Complete" : "Failed");
    endBusy(res.ok, res.ok ? "Scan Complete" : "Scan Failed");
  } catch (err) {
    print("Run Scan", `error: ${String(err)}`);
    setRunScanStatus("error", "Failed");
    endBusy(false, "Scan Failed");
  } finally {
    setButtonBusy(el.runScanBtn, false);
  }
}

async function rebuildLearningProfile() {
  setButtonBusy(el.rebuildLearningBtn, true);
  setRebuildLearningStatus("running", "Running...");
  startBusy("Rebuilding learning profile...");
  await flushUiFrame();
  try {
    const model = selectedModel();
    const aiCompare = !!el.rebuildAiCompare?.checked;
    if (aiCompare && !model) {
      print("Rebuild Learning", "AI Compare is enabled but no model is selected.");
      setRebuildLearningStatus("error", "Model Required");
      endBusy(false, "Rebuild Failed");
      return;
    }

    const res = await invoke("rebuild_learning_profile", {
      toolDir: state.toolDir,
      reportsDir: el.reportsDir.value.trim() || null,
      pairMode: el.learnPairMode?.value || "consecutive",
      trackChanges: !!el.learnTrackChanges?.checked,
      trackWeight: Number.parseInt(el.learnTrackWeight?.value || "2", 10) || 2,
      aiCompare,
      provider: el.scanProvider.value,
      model: model || null,
      ollamaUrl: el.ollamaUrl.value.trim() || null,
      bedrockRegion: el.bedrockRegion.value.trim() || null,
      bedrockProfile: el.bedrockProfile.value.trim() || null,
    });

    print("Rebuild Learning", commandText(res));
    setRebuildLearningStatus(res.ok ? "success" : "error", res.ok ? "Complete" : "Failed");
    endBusy(res.ok, res.ok ? "Learning Profile Rebuilt" : "Rebuild Failed");
  } catch (err) {
    print("Rebuild Learning", `error: ${String(err)}`);
    setRebuildLearningStatus("error", "Failed");
    endBusy(false, "Rebuild Failed");
  } finally {
    setButtonBusy(el.rebuildLearningBtn, false);
  }
}

async function importFeedback() {
  if (!el.feedbackDocx.value.trim()) {
    print("Import Feedback", "Reviewed annotated DOCX path is required.");
    return;
  }

  setButtonBusy(el.importFeedbackBtn, true);
  startBusy("Importing feedback...");
  await flushUiFrame();
  try {
    const author = el.feedbackAuthor.value;
    const res = await invoke("import_feedback_docx", {
      toolDir: state.toolDir,
      feedbackDocx: el.feedbackDocx.value.trim(),
      feedbackAuthor: author === "" ? null : author,
      dryRun: !!el.feedbackDryRun.checked,
    });
    print("Import Feedback", commandText(res));
    endBusy(res.ok, res.ok ? "Feedback Imported" : "Import Failed");
  } catch (err) {
    print("Import Feedback", `error: ${String(err)}`);
    endBusy(false, "Import Failed");
  } finally {
    setButtonBusy(el.importFeedbackBtn, false);
  }
}

function wirePickers() {
  el.pickScanDocxBtn.addEventListener("click", async () => {
    const p = await pickPath("docx-open");
    if (p) el.scanDocx.value = p;
  });

  el.pickFeedbackDocxBtn.addEventListener("click", async () => {
    const p = await pickPath("docx-open");
    if (p) el.feedbackDocx.value = p;
  });

  el.pickJsonOutBtn.addEventListener("click", async () => {
    const p = await pickPath("json-save", "report.qa.json");
    if (p) el.scanJsonOut.value = p;
  });

  el.pickAnnotateOutBtn.addEventListener("click", async () => {
    const p = await pickPath("docx-save", "report.annotated.docx");
    if (p) el.scanAnnotateOut.value = p;
  });

  el.pickReportsDirBtn.addEventListener("click", async () => {
    const p = await pickPath("folder");
    if (p) {
      el.reportsDir.value = p;
      saveSetting("reports_dir", p);
    }
  });

  el.pickIgnoreConfigBtn.addEventListener("click", async () => {
    const p = await pickPath("file");
    if (p) {
      el.ignoreConfig.value = p;
      saveSetting("ignore_config", p);
    }
  });

  el.pickFeedbackConfigBtn.addEventListener("click", async () => {
    const p = await pickPath("file");
    if (p) {
      el.feedbackConfig.value = p;
      saveSetting("feedback_config", p);
    }
  });
}

window.addEventListener("DOMContentLoaded", async () => {
  bind();
  initTheme();
  restorePersistedSettings();
  wireSettingsPersistence();
  initBrandFallbacks();
  try {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onThemeChange = () => {
      const theme = localStorage.getItem("draftenheimer.theme") || "system";
      if (theme === "system") {
        updateBrandForTheme(theme);
      }
    };
    if (typeof mq.addEventListener === "function") {
      mq.addEventListener("change", onThemeChange);
    } else if (typeof mq.addListener === "function") {
      mq.addListener(onThemeChange);
    }
  } catch (_) {
    // Keep app functional if media-query listeners are unavailable.
  }
  setRunScanStatus("idle", "Idle");
  setRebuildLearningStatus("idle", "Idle");

  el.openSettingsBtn.addEventListener("click", openSettings);
  el.closeSettingsBtn.addEventListener("click", closeSettings);

  el.refreshModelsBtn.addEventListener("click", refreshModels);
  el.pullModelBtn.addEventListener("click", pullModel);
  el.runtimeStartBtn.addEventListener("click", () => runRuntimeAction(el.runtimeStartBtn, "Start Runtime", "start_local_model_runtime"));
  el.runtimeStopBtn.addEventListener("click", () => runRuntimeAction(el.runtimeStopBtn, "Stop Runtime", "stop_local_model_runtime", { full: false }));
  el.runtimeStopFullBtn.addEventListener("click", () => runRuntimeAction(el.runtimeStopFullBtn, "Stop Runtime + App", "stop_local_model_runtime", { full: true }));

  el.runScanBtn.addEventListener("click", runScan);
  el.rebuildLearningBtn.addEventListener("click", rebuildLearningProfile);
  el.importFeedbackBtn.addEventListener("click", importFeedback);

  wirePickers();

  try {
    await detectToolDir();
    print("Startup", `Tool directory: ${state.toolDir}`);
  } catch (err) {
    print("Startup", `Could not detect tool directory: ${String(err)}`);
    endBusy(false, "Tool Path Error");
  }

  await refreshModels();
});
