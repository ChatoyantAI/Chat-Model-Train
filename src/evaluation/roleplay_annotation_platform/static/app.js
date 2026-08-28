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

function appendText(parent, className, text, tag = "p") {
  if (!text) return null;
  const node = document.createElement(tag);
  node.className = className;
  node.textContent = text;
  parent.append(node);
  return node;
}

function renderReferenceTable(parent, headers, rows) {
  const wrapper = document.createElement("div");
  wrapper.className = "reference-table-wrap";
  const table = document.createElement("table");
  table.className = "reference-table";
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  headers.forEach((header) => appendText(headRow, "", header, "th"));
  head.append(headRow);
  const body = document.createElement("tbody");
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    headers.forEach((header) => appendText(tr, "", row[header], "td"));
    body.append(tr);
  });
  table.append(head, body);
  wrapper.append(table);
  parent.append(wrapper);
}

function renderGuide() {
  const root = el("rubric-guide");
  root.replaceChildren();
  const heading = document.createElement("h3");
  heading.textContent = state.rubric.title;
  root.append(heading);
  appendText(root, "guide-intro", state.rubric.intro);

  const scoreHeading = document.createElement("h4");
  scoreHeading.textContent = "原子项统一评分";
  root.append(scoreHeading);
  renderReferenceTable(root, state.rubric.score_headers, state.rubric.score_scale_rows);
  renderReferenceTable(root, state.rubric.scale_usage_headers, state.rubric.scale_usage);

  const ruleHeading = document.createElement("h4");
  ruleHeading.textContent = "规则";
  root.append(ruleHeading);
  const rules = document.createElement("ol");
  rules.className = "guide-rules";
  state.rubric.rules.forEach((rule) => appendText(rules, "", rule, "li"));
  root.append(rules);
  appendText(root, "case-note", state.rubric.case_note);

  const source = document.createElement("details");
  source.className = "complete-source";
  const summary = document.createElement("summary");
  summary.textContent = `查看 ${state.rubric.source_filename} 完整原文（无删节）`;
  const hash = document.createElement("div");
  hash.className = "source-hash";
  hash.textContent = `SHA-256：${state.rubric.source_sha256}`;
  const pre = document.createElement("pre");
  pre.textContent = state.rubric.source_markdown;
  source.append(summary, hash, pre);
  root.append(source);
}

function renderGates() {
  const root = el("gate-groups");
  root.replaceChildren();
  appendText(root, "gate-intro", state.rubric.gate_intro);
  for (const group of state.rubric.gates) {
    const section = document.createElement("section");
    section.className = "gate-group";
    const title = document.createElement("h4");
    title.textContent = group.group;
    section.append(title);
    for (const item of group.items) {
      const row = document.createElement("div");
      row.className = "gate-item";
      const copy = document.createElement("div");
      const question = document.createElement("div");
      question.className = "gate-question";
      question.textContent = `${item.id} · ${item.question}`;
      copy.append(question);
      if (item.yes_result || item.no_result) {
        appendText(copy, "gate-condition", `是：${item.yes_result}；否：${item.no_result}`);
      } else if (item.yes_condition) {
        appendText(copy, "gate-condition", `记“是”的条件：${item.yes_condition}`);
      } else if (item.result_text) {
        appendText(copy, "gate-condition", `结果：${item.result_text}`);
      }
      const options = document.createElement("div");
      options.className = "gate-options";
      const choices = item.type === "boolean"
        ? [{ value: "false", label: "否" }, { value: "true", label: "是" }]
        : item.options;
      choices.forEach((choice) => {
        const defaultValue = item.type === "boolean" ? String(item.default) : item.default;
        const defaultChoice = choice.value === defaultValue;
        const radio = createRadio(`gate-${item.id}`, choice.value, choice.label, defaultChoice);
        radio.querySelector("input").dataset.gateId = item.id;
        options.append(radio);
      });
      row.append(copy, options);
      section.append(row);
    }
    root.append(section);
  }
  appendText(root, "gate-case", state.rubric.gate_case);
}

function renderRubric() {
  renderGuide();
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
      itemName.textContent = `${item.name}${item.starred ? " ★" : ""}`;
      const scale = document.createElement("span");
      scale.className = "scale-badge";
      scale.textContent = `尺度 ${item.scale_label}`;
      heading.append(itemId, itemName, scale);

      const guidance = document.createElement("section");
      guidance.className = "item-guidance";
      appendText(guidance, "item-intro", item.intro);
      const guidanceTitle = document.createElement("h4");
      guidanceTitle.textContent = "判定步骤";
      guidance.append(guidanceTitle);
      const steps = document.createElement("ol");
      item.decision_steps.forEach((step) => appendText(steps, "", step, "li"));
      guidance.append(steps);
      appendText(guidance, "decision-notes", item.decision_notes);
      if (item.not_counted) {
        const notCounted = document.createElement("p");
        notCounted.className = "labelled-guidance";
        const label = document.createElement("strong");
        label.textContent = "不计入：";
        notCounted.append(label, document.createTextNode(item.not_counted));
        guidance.append(notCounted);
      }
      const na = document.createElement("p");
      na.className = "labelled-guidance";
      const naLabel = document.createElement("strong");
      naLabel.textContent = "N/A：";
      na.append(naLabel, document.createTextNode(item.na_condition));
      guidance.append(na);

      const standards = document.createElement("fieldset");
      standards.className = "score-standards";
      standards.dataset.standardsFor = item.id;
      const standardsLegend = document.createElement("legend");
      standardsLegend.textContent = "评分档位（先读标准，再选择一项）";
      standards.append(standardsLegend);
      [...item.scale, "N/A"].forEach((score) => {
        const row = document.createElement("label");
        row.className = "score-standard";
        row.dataset.score = score;
        const input = document.createElement("input");
        input.type = "radio";
        input.name = `score-${item.id}`;
        input.value = score;
        input.dataset.itemId = item.id;
        const scoreLabel = document.createElement("span");
        scoreLabel.className = "score-standard-label";
        scoreLabel.textContent = score;
        const description = document.createElement("span");
        description.className = "score-standard-description";
        description.textContent = item.anchors[score];
        row.append(input, scoreLabel, description);
        standards.append(row);
      });

      const evidence = document.createElement("textarea");
      evidence.className = "evidence";
      evidence.dataset.evidenceFor = item.id;
      evidence.placeholder = "证据 / 原因（可选）";

      const examples = document.createElement("details");
      examples.className = "examples";
      const exampleSummary = document.createElement("summary");
      const exampleCount = item.cases.reduce((count, exampleCase) => count + exampleCase.examples.length, 0);
      exampleSummary.textContent = `查看完整案例（${item.cases.length} 组，${exampleCount} 条评分回复）`;
      examples.append(exampleSummary);
      item.cases.forEach((exampleCase) => {
        const caseBlock = document.createElement("section");
        caseBlock.className = "example-case";
        const caseTitle = document.createElement("h5");
        caseTitle.textContent = exampleCase.label;
        caseBlock.append(caseTitle);
        appendText(
          caseBlock,
          "example-background",
          `${exampleCase.background_label}：${exampleCase.background}`,
        );
        const grid = document.createElement("div");
        grid.className = "example-grid";
        exampleCase.examples.forEach((example) => {
          const cell = document.createElement("article");
          cell.className = `example-cell example-score-${example.score}`;
          const scoreLabel = document.createElement("div");
          scoreLabel.className = "example-score";
          scoreLabel.textContent = example.label;
          const reply = document.createElement("pre");
          reply.className = "example-reply";
          reply.textContent = example.reply;
          const explanation = document.createElement("p");
          explanation.className = "example-explanation";
          explanation.textContent = example.explanation;
          cell.append(scoreLabel, reply, explanation);
          grid.append(cell);
        });
        caseBlock.append(grid);
        examples.append(caseBlock);
      });
      card.append(heading, guidance, standards, evidence, examples);
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

function selectedScore(itemId) {
  const selected = document.querySelector(`input[name="score-${itemId}"]:checked`);
  return selected ? selected.value : null;
}

function updateAnchor(itemId) {
  const score = selectedScore(itemId);
  const standards = document.querySelector(`[data-standards-for="${itemId}"]`);
  const evidence = document.querySelector(`[data-evidence-for="${itemId}"]`);
  standards.querySelectorAll(".score-standard").forEach((row) => {
    row.classList.toggle("selected", row.dataset.score === score);
  });
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
  const gateDefaults = {};
  state.rubric.gates.forEach((group) => {
    group.items.forEach((item) => {
      gateDefaults[item.id] = item.type === "boolean" ? String(item.default) : item.default;
    });
  });
  document.querySelectorAll('input[name^="gate-"]').forEach((input) => {
    input.checked = input.value === gateDefaults[input.name.slice(5)];
  });
  el("gate-notes").value = "";
  el("overall-notes").value = "";
  document.querySelectorAll(".score-standard.selected").forEach((row) => row.classList.remove("selected"));
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
