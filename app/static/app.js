const POLL_INTERVAL_MS = 4000;
const BUSY_POLL_INTERVAL_MS = 900;

const state = {
  activeEngineId: null,
  currentStatus: null,
  currentHealth: null,
  engines: [],
  voices: [],
  pollHandle: null,
  pendingEngineId: null,
  selectedVoiceId: null,
  selectedLanguage: null,
};

const elements = {
  engineList: document.getElementById("engine-list"),
  stateBadge: document.getElementById("state-badge"),
  statusDevice: document.getElementById("status-device"),
  resourceUsage: document.getElementById("resource-usage"),
  resourceUsageLabel: document.getElementById("resource-usage-label"),
  healthStatus: document.getElementById("health-status"),
  healthReady: document.getElementById("health-ready"),
  healthVoiceCount: document.getElementById("health-voice-count"),
  selectedEngineChip: document.getElementById("selected-engine-chip"),
  refreshButton: document.getElementById("refresh-button"),
  voiceSelect: document.getElementById("voice-select"),
  languageSelect: document.getElementById("language-select"),
  voiceSummary: document.getElementById("voice-summary"),
  synthesizeForm: document.getElementById("synthesize-form"),
  synthesizeButton: document.getElementById("synthesize-button"),
  textInput: document.getElementById("text-input"),
  audioPlayer: document.getElementById("audio-player"),
  metricLoad: document.getElementById("metric-load"),
  metricSynth: document.getElementById("metric-synth"),
  metricE2e: document.getElementById("metric-e2e"),
  metricAudio: document.getElementById("metric-audio"),
  metricRtf: document.getElementById("metric-rtf"),
  actionStatus: document.getElementById("action-status"),
};

function setButtonBusy(button, isBusy, busyLabel, idleLabel) {
  button.disabled = isBusy;
  button.dataset.busy = isBusy ? "true" : "false";
  if (busyLabel && idleLabel) {
    button.textContent = isBusy ? busyLabel : idleLabel;
  }
}

async function request(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(payload.detail || response.statusText);
  }
  return response.json();
}

function formatMs(value) {
  return typeof value === "number" ? `${value.toFixed(2)} ms` : "-";
}

function updateActionStatus(message, variant = "neutral") {
  elements.actionStatus.textContent = message;
  elements.actionStatus.dataset.variant = variant;
}

function isEngineBusy() {
  const status = state.currentStatus;
  if (!status) {
    return false;
  }
  return status.loading || status.state === "loading" || status.state === "unloading";
}

function schedulePolling() {
  window.clearTimeout(state.pollHandle);
  const interval = isEngineBusy() || state.pendingEngineId ? BUSY_POLL_INTERVAL_MS : POLL_INTERVAL_MS;
  state.pollHandle = window.setTimeout(() => {
    bootstrap({ silent: true });
  }, interval);
}

function resetMetrics() {
  elements.metricLoad.textContent = "-";
  elements.metricSynth.textContent = "-";
  elements.metricE2e.textContent = "-";
  elements.metricAudio.textContent = "-";
  elements.metricRtf.textContent = "-";
}

function renderVoices() {
  const currentVoice = elements.voiceSelect.value || state.selectedVoiceId || state.currentStatus?.last_voice || null;
  const currentLanguage = elements.languageSelect.value || state.selectedLanguage || state.currentStatus?.last_language || null;

  elements.voiceSelect.innerHTML = "";
  elements.languageSelect.innerHTML = "";

  if (!state.voices.length) {
    const voiceOption = document.createElement("option");
    voiceOption.value = "";
    voiceOption.textContent = "Load an engine first";
    elements.voiceSelect.appendChild(voiceOption);

    const languageOption = document.createElement("option");
    languageOption.value = "";
    languageOption.textContent = "Unavailable";
    elements.languageSelect.appendChild(languageOption);

    elements.voiceSummary.textContent = "No voices available yet for the active engine.";
    elements.languageSelect.disabled = true;
    return;
  }

  for (const voice of state.voices) {
    const option = document.createElement("option");
    option.value = voice.id;
    option.textContent = `${voice.label} [${voice.languages.join(", ")}]`;
    elements.voiceSelect.appendChild(option);
  }

  const selectedVoice = state.voices.find((voice) => voice.id === currentVoice) || state.voices[0];
  elements.voiceSelect.value = selectedVoice.id;
  state.selectedVoiceId = selectedVoice.id;

  for (const language of selectedVoice.languages) {
    const option = document.createElement("option");
    option.value = language;
    option.textContent = language;
    elements.languageSelect.appendChild(option);
  }

  const supportsLanguageControl = state.currentStatus?.extra?.supports_language_control !== false;
  const languageNote = state.currentStatus?.extra?.language_note || "";
  const selectedLanguage = selectedVoice.languages.includes(currentLanguage)
    ? currentLanguage
    : (state.currentStatus?.last_language && selectedVoice.languages.includes(state.currentStatus.last_language))
      ? state.currentStatus.last_language
      : (selectedVoice.default_language || selectedVoice.languages[0] || "");

  elements.languageSelect.value = selectedLanguage;
  state.selectedLanguage = selectedLanguage;
  elements.languageSelect.disabled = !supportsLanguageControl;
  if (!supportsLanguageControl) {
    elements.voiceSummary.textContent = "This engine does not expose reliable per-request language control in the local backend.";
    return;
  }

  elements.voiceSummary.textContent = languageNote || `${state.voices.length} voice(s) exposed by the active engine.`;
}

function renderStatus(status) {
  state.currentStatus = status;
  state.activeEngineId = status.active_engine_id;
  elements.stateBadge.textContent = status.state;
  elements.stateBadge.dataset.state = status.state;
  elements.statusDevice.textContent = status.device || "-";
  elements.selectedEngineChip.textContent = status.display_name;
  elements.resourceUsage.textContent = status.resource_usage?.display || "-";
  elements.resourceUsageLabel.textContent = status.resource_usage?.label || "-";

  const ready = status.loaded && status.state === "ready";
  elements.synthesizeButton.disabled = !ready;

  if (state.pendingEngineId && state.pendingEngineId === status.active_engine_id && ready) {
    state.pendingEngineId = null;
    updateActionStatus(`${status.display_name} is loaded and ready.`, "success");
  }
}

function renderHealth(health) {
  state.currentHealth = health;
  elements.healthStatus.textContent = health.status;
  elements.healthStatus.dataset.state = health.ready ? "ready" : health.active_engine_state;
  elements.healthReady.textContent = health.ready ? "Yes" : "No";
  elements.healthReady.dataset.state = health.ready ? "ready" : health.active_engine_state;
  elements.healthVoiceCount.textContent = String(health.available_voice_count ?? 0);
}

function renderMetrics(metrics) {
  elements.metricLoad.textContent = formatMs(metrics.load_time_ms);
  elements.metricSynth.textContent = formatMs(metrics.synthesis_time_ms);
  elements.metricE2e.textContent = formatMs(metrics.end_to_end_time_ms);
  elements.metricAudio.textContent = formatMs(metrics.audio_duration_ms);
  elements.metricRtf.textContent = typeof metrics.real_time_factor === "number"
    ? metrics.real_time_factor.toFixed(4)
    : "-";
}

function renderEngineList() {
  elements.engineList.innerHTML = "";

  if (!state.engines.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "No engines available.";
    elements.engineList.appendChild(empty);
    return;
  }

  for (const engine of state.engines) {
    const item = document.createElement("article");
    item.className = `engine-item ${engine.active ? "active" : ""} ${engine.selected ? "selected" : ""}`;

    const header = document.createElement("div");
    header.className = "engine-header";

    const meta = document.createElement("div");
    meta.className = "engine-meta";

    const title = document.createElement("strong");
    title.textContent = engine.display_name;

    const details = document.createElement("span");
    const voices = engine.status.available_voices.length;
    const memoryHint = engine.status.extra?.memory_hint;
    const detailParts = [`${engine.status.device || "-"}`, `${voices} voice(s)`];
    if (memoryHint) {
      detailParts.push(memoryHint);
    }
    details.textContent = detailParts.join(" · ");
    details.className = "engine-details";
    meta.append(title, details);

    const controls = document.createElement("div");
    controls.className = "engine-controls";

    const badge = document.createElement("span");
    badge.className = "badge compact";
    badge.dataset.state = engine.status.state;
    badge.textContent = engine.status.state;

    const button = document.createElement("button");
    button.type = "button";
    button.className = engine.active ? "ghost-button" : "engine-button";
    const pending = state.pendingEngineId === engine.engine_id;
    const isSelectedOnly = engine.selected && !engine.active;
    button.textContent = pending ? "Loading..." : engine.active ? "Active" : "Load";
    button.disabled = pending || engine.active;
    if (isSelectedOnly) {
      item.dataset.selected = "true";
    }
    button.addEventListener("click", () => {
      selectAndLoadEngine(engine);
    });

    controls.append(badge, button);
    header.append(meta, controls);
    item.appendChild(header);
    elements.engineList.appendChild(item);
  }
}

async function selectAndLoadEngine(engine) {
  try {
    state.pendingEngineId = engine.engine_id;
    renderEngineList();
    updateActionStatus(`Switching to ${engine.display_name} and loading it.`, "neutral");
    await request("/api/engines/activate", {
      method: "POST",
      body: JSON.stringify({ engine_id: engine.engine_id }),
    });
    await bootstrap({ silent: true });
  } catch (error) {
    state.pendingEngineId = null;
    updateActionStatus(error.message, "error");
    renderEngineList();
  } finally {
    schedulePolling();
  }
}

async function loadStatus() {
  const [status, engines, voices, health] = await Promise.all([
    request("/api/status"),
    request("/api/engines"),
    request("/api/voices"),
    request("/health"),
  ]);

  state.engines = engines.engines;
  state.voices = voices.voices;
  renderStatus(status);
  renderHealth(health);
  renderEngineList();
  renderVoices();
}

async function bootstrap(options = {}) {
  try {
    await loadStatus();
    if (!options.silent && !state.pendingEngineId) {
      updateActionStatus("Status refreshed.", "neutral");
    }
  } catch (error) {
    updateActionStatus(error.message, "error");
  } finally {
    schedulePolling();
  }
}

elements.refreshButton.addEventListener("click", async () => {
  try {
    setButtonBusy(elements.refreshButton, true, "Refreshing...", "Refresh");
    await bootstrap();
  } finally {
    setButtonBusy(elements.refreshButton, false, "Refreshing...", "Refresh");
  }
});

elements.voiceSelect.addEventListener("change", () => {
  const selected = state.voices.find((voice) => voice.id === elements.voiceSelect.value);
  if (!selected) {
    return;
  }
  state.selectedVoiceId = selected.id;
  const previousLanguage = elements.languageSelect.value || state.selectedLanguage;
  elements.languageSelect.innerHTML = "";
  for (const language of selected.languages) {
    const option = document.createElement("option");
    option.value = language;
    option.textContent = language;
    elements.languageSelect.appendChild(option);
  }
  const nextLanguage = selected.languages.includes(previousLanguage)
    ? previousLanguage
    : (selected.default_language || selected.languages[0] || "");
  elements.languageSelect.value = nextLanguage;
  state.selectedLanguage = nextLanguage;
  elements.voiceSummary.textContent = selected.description || `${selected.label} ready for synthesis.`;
});

elements.languageSelect.addEventListener("change", () => {
  state.selectedLanguage = elements.languageSelect.value || null;
});

elements.synthesizeForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    updateActionStatus("Generating speech sample.", "neutral");
    setButtonBusy(elements.synthesizeButton, true, "Generating...", "Generate Speech");
    const payload = await request("/api/synthesize", {
      method: "POST",
      body: JSON.stringify({
        text: elements.textInput.value,
        voice: elements.voiceSelect.value || null,
        language: elements.languageSelect.value || null,
      }),
    });
    state.selectedVoiceId = payload.voice || elements.voiceSelect.value || state.selectedVoiceId;
    state.selectedLanguage = payload.language || elements.languageSelect.value || state.selectedLanguage;
    renderMetrics(payload.metrics);
    elements.audioPlayer.src = `data:audio/wav;base64,${payload.wav_base64}`;
    elements.audioPlayer.play().catch(() => null);
    updateActionStatus("Speech generated successfully.", "success");
  } catch (error) {
    updateActionStatus(error.message, "error");
  } finally {
    const ready = state.currentStatus?.loaded && state.currentStatus?.state === "ready";
    setButtonBusy(elements.synthesizeButton, false, "Generating...", "Generate Speech");
    elements.synthesizeButton.disabled = !ready;
    schedulePolling();
  }
});

resetMetrics();
bootstrap();
