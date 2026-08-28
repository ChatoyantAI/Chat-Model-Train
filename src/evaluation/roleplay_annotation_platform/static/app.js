"use strict";

const state = {
  username: null,
  rubric: null,
  sample: null,
  dirty: false,
  loading: false,
};

const el = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const response = await fetch(path, {
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  let payload = null;
  try { payload = await response.json(); } catch (_) { payload = {}; }
  if (!response.ok) {
    throw new Error(payload.detail || `请求失败（${response.status}）`);
  }
  return payload;
}

function toast(message, error = false) {
  const box = el("toast");
  box.textContent = message;
  box.classList.toggle("error", error);
  box.classList.remove("hidden");
  window.clearTimeout(toast.timer);
  toast.timer = window.setTimeout(() => box.classList.add("hidden"), 3200);
}

function setDirty(value) {
  state.dirty = value;
  if (value) el("save-state").textContent = "有未保存修改";
}

function updateProgress(progress) {
  const percent = progress.total ? (progress.submitted / progress.total) * 100 : 0;
  el("progress-text").textContent = `已提交 ${progress.submitted} / ${progress.total}`;
  el("progress-fill").style.width = `${percent}%`;
}

function showLogin() {
  el("login-view").classList.remove("hidden");
  el("app-view").classList.add("hidden");
  el("username").focus();
}

async function enterApp(session) {
  state.username = session.username;
  el("current-user").textContent = session.username;
  el("login-view").classList.add("hidden");
  el("app-view").classList.remove("hidden");
  updateProgress(session.progress);
  if (!state.rubric) {
    state.rubric = await api("/api/rubric");
    renderRubric();
  }
  const seq = session.next_seq || 1;
  await loadSample(seq);
}

function createRadio(name, value, label, checked = false) {
  const wrapper = document.createElement("label");
  wrapper.className = "radio-chip";
  const input = document.createElement("input");
  input.type = "radio";
  input.name = name;
  input.value = value;
  input.checked = checked;
  const span = document.createElement("span");
  span.textContent = label;
  wrapper.append(input, span);
  return wrapper;
}

function renderGates() {
  const root = el("gate-groups");
  root.replaceChildren();
  for (const group of state.rubric.gates) {
    const section = document.createElement("section");
    section.className = "gate-group";
    const title = document.createElement("h4");
    title.textContent = group.group;
    section.append(title);
    for (const item of group.items) {
      const row = document.createElement("div");
      row.className = "gate-item";
      const question = document.createElement("div");
      question.textContent = `${item.id} · ${item.question}`;
      const options = document.createElement("div");
      options.className = "gate-options";
      const choices = item.type === "boolean"
        ? [{ value: "false", label: "否" }, { value: "true", label: "是" }]
        : item.options;
      choices.forEach((choice, index) => {
        let defaultChoice = index === 0;
        if (item.id === "G-S1") defaultChoice = choice.value === "no";
        if (item.id === "G-S2") defaultChoice = choice.value === "N/A";
        if (item.id === "G-S3") defaultChoice = choice.value === "no";
        const radio = createRadio(`gate-${item.id}`, choice.value, choice.label, defaultChoice);
        radio.querySelector("input").dataset.gateId = item.id;
        options.append(radio);
      });
      row.append(question, options);
      section.append(row);
    }
    root.append(section);
  }
}

function renderRubric() {
  renderGates();
  const root = el("rubric-dimensions");
  root.replaceChildren();
  state.rubric.dimensions.forEach((dimension, dimensionIndex) => {
    const details = document.createElement("details");
    details.className = "dimension";
    details.open = dimensionIndex === 0;
    const summary = document.createElement("summary");
    summary.textContent = dimension.name;
    details.append(summary);
    const description = document.createElement("p");
    description.className = "dimension-description";
    description.textContent = dimension.description;
    details.append(description);

    for (const item of dimension.items) {
      const card = document.createElement("article");
      card.className = "rubric-item";
      card.dataset.itemId = item.id;

      const heading = document.createElement("div");
      heading.className = "item-heading";
      const itemId = document.createElement("span");
      itemId.className = "item-id";
      itemId.textContent = item.id;
      const itemName = document.createElement("span");
      itemName.className = "item-name";
      itemName.textContent = item.name;
      heading.append(itemId, itemName);

      const question = document.createElement("p");
      question.className = "item-question";
      question.textContent = item.question;

      const options = document.createElement("div");
      options.className = "score-options";
      ["0", "1", "2", "3", "N/A"].forEach((score) => {
        const radio = createRadio(`score-${item.id}`, score, score);
        const input = radio.querySelector("input");
        input.dataset.itemId = item.id;
        options.append(radio);
      });

      const anchor = document.createElement("div");
      anchor.className = "anchor-text";
      anchor.dataset.anchorFor = item.id;
      anchor.textContent = "选择分数后显示对应判定标准";

      const evidence = document.createElement("textarea");
      evidence.className = "evidence";
      evidence.dataset.evidenceFor = item.id;
      evidence.placeholder = "证据 / 原因（可选）";

      const examples = document.createElement("details");
      examples.className = "examples";
      const exampleSummary = document.createElement("summary");
      exampleSummary.textContent = "查看 0–3 分角色扮演回复示例";
      const background = document.createElement("p");
      background.className = "example-background";
      background.textContent = `背景：${item.example.background}`;
      const grid = document.createElement("div");
      grid.className = "example-grid";
      ["0", "1", "2", "3"].forEach((score) => {
        const cell = document.createElement("div");
        cell.className = "example-cell";
        const scoreLabel = document.createElement("span");
        scoreLabel.className = "example-score";
        scoreLabel.textContent = `${score} 分回复`;
        const text = document.createElement("span");
        text.textContent = item.example[score].replaceAll("<br>", "\n");
        cell.append(scoreLabel, text);
        grid.append(cell);
      });
      examples.append(exampleSummary, background, grid);
      card.append(heading, question, options, anchor, evidence, examples);
      details.append(card);
    }
    root.append(details);
  });

  document.querySelectorAll("input[type=radio], textarea").forEach((control) => {
    control.addEventListener("change", () => {
      setDirty(true);
      if (control.dataset.itemId) updateAnchor(control.dataset.itemId);
      updateScoreCounter();
    });
    if (control.tagName === "TEXTAREA") {
      control.addEventListener("input", () => {
        setDirty(true);
        if (control.dataset.evidenceFor) updateAnchor(control.dataset.evidenceFor);
      });
    }
  });
}

function itemById(itemId) {
  for (const dimension of state.rubric.dimensions) {
    const found = dimension.items.find((item) => item.id === itemId);
    if (found) return found;
  }
  return null;
}

function selectedScore(itemId) {
  const selected = document.querySelector(`input[name="score-${itemId}"]:checked`);
  return selected ? selected.value : null;
}

function updateAnchor(itemId) {
  const item = itemById(itemId);
  const score = selectedScore(itemId);
  const anchor = document.querySelector(`[data-anchor-for="${itemId}"]`);
  const evidence = document.querySelector(`[data-evidence-for="${itemId}"]`);
  anchor.textContent = score ? `${score}：${item.anchors[score]}` : "选择分数后显示对应判定标准";
  evidence.placeholder = "证据 / 原因（可选）";
  evidence.classList.remove("required-empty");
}

function updateScoreCounter() {
  const total = state.rubric.dimensions.reduce((n, dimension) => n + dimension.items.length, 0);
  const completed = state.rubric.dimensions.reduce(
    (n, dimension) => n + dimension.items.filter((item) => selectedScore(item.id) !== null).length,
    0,
  );
  el("score-counter").textContent = `${completed} / ${total}`;
}

function resetForm() {
  document.querySelectorAll('input[name^="score-"]').forEach((input) => { input.checked = false; });
  document.querySelectorAll(".evidence").forEach((textarea) => {
    textarea.value = "";
    textarea.classList.remove("required-empty");
  });
  const gateDefaults = {
    "G-D1": "false",
    "G-D2": "false",
    "G-D3": "false",
    "G-S1": "no",
    "G-S2": "N/A",
    "G-S3": "no",
    "G-P1": "false",
    "G-P2": "false",
    "G-P3": "false",
  };
  document.querySelectorAll('input[name^="gate-"]').forEach((input) => {
    input.checked = input.value === gateDefaults[input.name.slice(5)];
  });
  el("gate-notes").value = "";
  el("overall-notes").value = "";
  document.querySelectorAll("[data-anchor-for]").forEach((anchor) => {
    anchor.textContent = "选择分数后显示对应判定标准";
  });
  updateScoreCounter();
}

function applyAnnotation(annotation) {
  if (!annotation) return;
  for (const [gateId, value] of Object.entries(annotation.gates || {})) {
    const stringValue = typeof value === "boolean" ? String(value) : value;
    const input = document.querySelector(`input[name="gate-${gateId}"][value="${CSS.escape(stringValue)}"]`);
    if (input) input.checked = true;
  }
  for (const [itemId, record] of Object.entries(annotation.scores || {})) {
    const input = document.querySelector(`input[name="score-${itemId}"][value="${CSS.escape(String(record.value))}"]`);
    if (input) input.checked = true;
    const evidence = document.querySelector(`[data-evidence-for="${itemId}"]`);
    if (evidence) evidence.value = record.evidence || "";
    updateAnchor(itemId);
  }
  el("gate-notes").value = annotation.gate_notes || "";
  el("overall-notes").value = annotation.overall_notes || "";
  el("save-state").textContent = annotation.status === "submitted"
    ? `已提交 · ${new Date(annotation.updated_at).toLocaleString()}`
    : `草稿 · ${new Date(annotation.updated_at).toLocaleString()}`;
  updateScoreCounter();
}

function renderMessages(messages) {
  const root = el("messages");
  root.replaceChildren();
  const labels = {
    system: "SYSTEM",
    greeting: "GREETING",
    user: "USER",
    assistant: "ASSISTANT HISTORY",
  };
  for (const message of messages) {
    const card = document.createElement("article");
    card.className = `message message-${message.display_role}`;
    const label = document.createElement("div");
    label.className = "message-label";
    label.textContent = labels[message.display_role] || message.display_role;
    const content = document.createElement("pre");
    content.textContent = message.content;
    card.append(label, content);
    root.append(card);
  }
}

async function loadSample(seq) {
  if (state.loading) return;
  state.loading = true;
  try {
    const sample = await api(`/api/sample/${seq}`);
    state.sample = sample;
    resetForm();
    renderMessages(sample.messages);
    el("candidate-content").textContent = sample.candidate.content;
    el("jump-input").value = sample.seq;
    el("jump-input").max = sample.total;
    el("sample-total").textContent = `/ ${sample.total}`;
    el("trigger-badge").textContent = `触发：${sample.trigger_type}`;
    el("prev-button").disabled = sample.seq <= 1;
    el("next-button").disabled = sample.seq >= sample.total;
    applyAnnotation(sample.annotation);
    updateProgress(sample.progress);
    if (!sample.annotation) el("save-state").textContent = "尚未保存";
    setDirty(false);
    window.scrollTo({ top: 0, behavior: "instant" });
  } catch (error) {
    toast(error.message, true);
  } finally {
    state.loading = false;
  }
}

function collectGates() {
  const gates = {};
  document.querySelectorAll('input[name^="gate-"]:checked').forEach((input) => {
    const gateId = input.name.slice(5);
    gates[gateId] = ["true", "false"].includes(input.value)
      ? input.value === "true"
      : input.value;
  });
  return gates;
}

function collectScores() {
  const scores = {};
  for (const dimension of state.rubric.dimensions) {
    for (const item of dimension.items) {
      const score = selectedScore(item.id);
      if (score === null) continue;
      scores[item.id] = {
        value: score === "N/A" ? "N/A" : Number(score),
        evidence: document.querySelector(`[data-evidence-for="${item.id}"]`).value.trim(),
      };
    }
  }
  return scores;
}

function validateSubmission(scores) {
  const missing = [];
  for (const dimension of state.rubric.dimensions) {
    for (const item of dimension.items) {
      if (!scores[item.id]) {
        missing.push(item.id);
      }
    }
  }
  if (missing.length) throw new Error(`还有 ${missing.length} 项未评分：${missing.join("、")}`);
}

async function save(status) {
  if (!state.sample || state.loading) return;
  const scores = collectScores();
  if (status === "submitted") validateSubmission(scores);
  const payload = {
    seq: state.sample.seq,
    status,
    gates: collectGates(),
    scores,
    gate_notes: el("gate-notes").value,
    overall_notes: el("overall-notes").value,
  };
  el("save-button").disabled = true;
  el("submit-button").disabled = true;
  try {
    const result = await api("/api/annotation", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    updateProgress(result.progress);
    setDirty(false);
    el("save-state").textContent = status === "submitted" ? "已提交" : "草稿已保存";
    toast(status === "submitted" ? "提交成功" : "草稿已保存");
    if (status === "submitted" && result.next_seq) await loadSample(result.next_seq);
  } catch (error) {
    toast(error.message, true);
  } finally {
    el("save-button").disabled = false;
    el("submit-button").disabled = false;
  }
}

function canNavigate() {
  return !state.dirty || window.confirm("当前修改尚未保存，确定离开这条样本吗？");
}

document.addEventListener("DOMContentLoaded", async () => {
  el("login-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      const session = await api("/api/login", {
        method: "POST",
        body: JSON.stringify({ username: el("username").value }),
      });
      await enterApp(session);
    } catch (error) {
      toast(error.message, true);
    }
  });

  el("logout-button").addEventListener("click", async () => {
    if (!canNavigate()) return;
    await api("/api/logout", { method: "POST", body: "{}" });
    state.username = null;
    state.sample = null;
    showLogin();
  });
  el("prev-button").addEventListener("click", () => {
    if (canNavigate()) loadSample(state.sample.seq - 1);
  });
  el("next-button").addEventListener("click", () => {
    if (canNavigate()) loadSample(state.sample.seq + 1);
  });
  el("jump-button").addEventListener("click", () => {
    const seq = Number(el("jump-input").value);
    if (canNavigate()) loadSample(seq);
  });
  el("jump-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter") el("jump-button").click();
  });
  el("save-button").addEventListener("click", () => save("draft"));
  el("submit-button").addEventListener("click", () => save("submitted"));
  document.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
      event.preventDefault();
      save("draft");
    }
  });
  window.addEventListener("beforeunload", (event) => {
    if (state.dirty) event.preventDefault();
  });
  window.setInterval(() => {
    if (state.dirty && !state.loading) save("draft");
  }, 60000);

  try {
    const session = await api("/api/me");
    await enterApp(session);
  } catch (_) {
    showLogin();
  }
});
