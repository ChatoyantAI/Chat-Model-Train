const state = {
  username: null,
  rubric: null,
  sample: null,
  dirty: false,
  loading: false,
  models: [],
  primaryRun: null,
};

const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    credentials: "same-origin",
    ...options,
  });
  let payload = null;
  try {
    payload = await response.json();
  } catch (_) {
    payload = {};
  }
  if (!response.ok)
    throw new Error(payload.detail || `请求失败（HTTP ${response.status}）`);
  return payload;
}

function toast(message, isError = false) {
  const node = $("toast");
  node.textContent = message;
  node.classList.toggle("error-toast", isError);
  node.classList.remove("hidden");
  window.clearTimeout(toast.timer);
  toast.timer = window.setTimeout(() => node.classList.add("hidden"), 4200);
}

function setDirty(value) {
  state.dirty = value;
  if (value) $("save-state").textContent = "有未保存修改";
}

function radio(name, value, label, className = "") {
  const wrapper = document.createElement("label");
  wrapper.className = `choice ${className}`.trim();
  const input = document.createElement("input");
  input.type = "radio";
  input.name = name;
  input.value = value;
  const span = document.createElement("span");
  span.textContent = label;
  wrapper.append(input, span);
  return wrapper;
}

function definitionLine(label, text) {
  const p = document.createElement("p");
  p.className = "definition";
  const strong = document.createElement("strong");
  strong.textContent = `${label}：`;
  p.append(strong, document.createTextNode(text));
  return p;
}

function rubricItems() {
  return (state.rubric?.dimensions || []).flatMap((dimension) =>
    (dimension.items || []).map((item) => ({
      ...item,
      dimensionName: dimension.name,
    })),
  );
}

function scoreChoice(item, value) {
  const label = value === "N/A" ? "N/A" : value;
  const className =
    value === "N/A" ? "na" : value === "0" || value === "1" ? "fail" : "";
  return radio(`item-${item.id}`, value, label, className);
}

function renderRubric() {
  const gatesRoot = $("gate-list");
  gatesRoot.replaceChildren();
  state.rubric.gates.forEach((gate) => {
    const card = document.createElement("article");
    card.className = "gate-card";
    const head = document.createElement("div");
    head.className = "gate-head";
    const title = document.createElement("div");
    title.innerHTML = `<span class="check-id"></span><span class="check-name"></span>`;
    title.querySelector(".check-id").textContent = gate.id;
    title.querySelector(".check-name").textContent = gate.name;
    const choices = document.createElement("div");
    choices.className = "choices";
    choices.append(
      radio(`gate-${gate.id}`, "false", "没有"),
      radio(`gate-${gate.id}`, "true", "发现错误", "error-choice"),
    );
    head.append(title, choices);
    card.append(head, definitionLine("何时算错误", gate.error_if));
    gatesRoot.append(card);
  });

  const root = $("item-groups");
  root.replaceChildren();
  state.rubric.dimensions.forEach((dimension) => {
    const section = document.createElement("section");
    section.className = "item-group";
    const heading = document.createElement("h4");
    heading.textContent = dimension.name;
    const list = document.createElement("div");
    list.className = "item-list";
    dimension.items.forEach((item) => {
      const card = document.createElement("article");
      card.className = "item-card";
      card.dataset.itemId = item.id;
      const head = document.createElement("div");
      head.className = "item-head";
      const title = document.createElement("div");
      const id = document.createElement("span");
      id.className = "check-id";
      id.textContent = item.id;
      const name = document.createElement("span");
      name.className = "check-name";
      name.textContent = `${item.name}${item.starred ? " ★" : ""}`;
      title.append(id, name);
      if (!item.allows_na) {
        const fixed = document.createElement("span");
        fixed.className = "always";
        fixed.textContent = "始终适用";
        title.append(fixed);
      }
      const choices = document.createElement("div");
      choices.className = "choices";
      item.scale.forEach((value) =>
        choices.append(scoreChoice(item, String(value))),
      );
      if (item.allows_na) choices.append(scoreChoice(item, "N/A"));
      head.append(title, choices);
      const anchors = document.createElement("div");
      anchors.className = "score-anchors";
      [...item.scale.map(String), ...(item.allows_na ? ["N/A"] : [])].forEach(
        (value) => {
          const row = document.createElement("div");
          row.className = "anchor-row";
          const badge = document.createElement("span");
          badge.className = `anchor-badge anchor-${value.replace("/", "")}`;
          badge.textContent = value;
          const text = document.createElement("span");
          text.textContent = item.anchors[value];
          row.append(badge, text);
          anchors.append(row);
        },
      );
      const details = document.createElement("details");
      details.className = "rubric-details";
      const detailsTitle = document.createElement("summary");
      detailsTitle.textContent = "查看判定步骤、边界和案例";
      const pre = document.createElement("pre");
      pre.textContent = item.reference_markdown || item.intro || "";
      details.append(detailsTitle, pre);
      card.append(head, anchors, details);
      list.append(card);
    });
    section.append(heading, list);
    root.append(section);
  });

  document.querySelectorAll('input[type="radio"]').forEach((input) => {
    input.addEventListener("change", () => {
      setDirty(true);
      updateFormState();
    });
  });
}

function selected(name) {
  return (
    document.querySelector(`input[name="${CSS.escape(name)}"]:checked`)
      ?.value ?? null
  );
}

function collectGates() {
  const gates = {};
  state.rubric.gates.forEach((gate) => {
    const value = selected(`gate-${gate.id}`);
    if (value !== null) gates[gate.id] = value === "true";
  });
  return gates;
}

function collectItems() {
  const items = {};
  rubricItems().forEach((item) => {
    const value = selected(`item-${item.id}`);
    if (value !== null) items[item.id] = value;
  });
  return items;
}

function hasGateError() {
  return Object.values(collectGates()).some(Boolean);
}

function updateFormState() {
  const items = collectItems();
  $("answer-count").textContent =
    `${Object.keys(items).length} / ${rubricItems().length}`;
  $("quality-score").textContent = calculateHumanScore(items);
  $("gate-warning").classList.toggle("hidden", !hasGateError());
  document
    .querySelectorAll(".item-card")
    .forEach((card) => card.classList.remove("incomplete"));
}

function calculateHumanScore(items) {
  if (!state.rubric) return "—";
  const groupScores = [];
  state.rubric.dimensions.forEach((dimension) => {
    const values = dimension.items.flatMap((item) => {
      const value = items[item.id];
      if (value === undefined || value === "N/A") return [];
      const maximum = Math.max(...item.scale.map(Number));
      return [Number(value) / maximum];
    });
    if (values.length)
      groupScores.push(values.reduce((a, b) => a + b, 0) / values.length);
  });
  if (!groupScores.length) return "—";
  const score =
    (100 * groupScores.reduce((a, b) => a + b, 0)) / groupScores.length;
  return `${score.toFixed(1)} / 100`;
}

function resetForm() {
  document.querySelectorAll('input[type="radio"]').forEach((input) => {
    input.checked = false;
  });
  state.rubric.gates.forEach((gate) => {
    const input = document.querySelector(
      `input[name="gate-${gate.id}"][value="false"]`,
    );
    if (input) input.checked = true;
  });
  $("notes").value = "";
  $("llm-result").className = "llm-result hidden";
  $("llm-result").replaceChildren();
  $("verify-controls").classList.add("hidden");
  $("llm-status").textContent = "";
  state.primaryRun = null;
  updateFormState();
}

function applyReview(review) {
  if (!review) return;
  Object.entries(review.gates || {}).forEach(([id, value]) => {
    const input = document.querySelector(
      `input[name="gate-${id}"][value="${String(value)}"]`,
    );
    if (input) input.checked = true;
  });
  Object.entries(review.items || {}).forEach(([id, value]) => {
    const input = document.querySelector(
      `input[name="item-${id}"][value="${CSS.escape(value)}"]`,
    );
    if (input) input.checked = true;
  });
  $("notes").value = review.notes || "";
  $("save-state").textContent =
    review.status === "submitted"
      ? `已提交 · ${review.decision}`
      : "草稿已保存";
  updateFormState();
}

function renderMessages(messages) {
  const labels = {
    system: "SYSTEM",
    greeting: "GREETING",
    user: "USER",
    assistant: "ASSISTANT HISTORY",
  };
  const root = $("messages");
  root.replaceChildren();
  messages.forEach((message) => {
    const article = document.createElement("article");
    article.className = `message message-${message.display_role}`;
    const label = document.createElement("div");
    label.className = "message-label";
    label.textContent =
      labels[message.display_role] || message.display_role.toUpperCase();
    const content = document.createElement("div");
    content.className = "message-content";
    content.textContent = message.content;
    content.setAttribute("translate", "yes");
    article.append(label, content);
    root.append(article);
  });
}

function updateProgress(progress) {
  $("progress-text").textContent =
    `已提交 ${progress.submitted} / ${progress.total}`;
}

async function loadSample(seq) {
  if (state.loading) return;
  state.loading = true;
  try {
    const sample = await api(`/api/sample/${seq}`);
    state.sample = sample;
    resetForm();
    renderMessages(sample.messages);
    $("candidate-content").textContent = sample.candidate.content;
    $("jump-input").value = sample.seq;
    $("jump-input").max = sample.total;
    $("sample-total").textContent = ` / ${sample.total}`;
    $("trigger-badge").textContent = `触发：${sample.trigger_type}`;
    $("prev-button").disabled = sample.seq <= 1;
    $("next-button").disabled = sample.seq >= sample.total;
    applyReview(sample.review);
    updateProgress(sample.progress);
    if (!sample.review) $("save-state").textContent = "尚未保存";
    setDirty(false);
    document.querySelector(".context-panel").scrollTop = 0;
    document.querySelector(".review-panel").scrollTop = 0;
  } catch (error) {
    toast(error.message, true);
  } finally {
    state.loading = false;
  }
}

function validateHuman() {
  const gates = collectGates();
  if (Object.keys(gates).length !== state.rubric.gates.length)
    throw new Error("请完成全部 Gate 判断");
  if (Object.values(gates).some(Boolean)) return;
  const items = collectItems();
  const missing = rubricItems().filter((item) => !(item.id in items));
  if (missing.length) {
    missing.forEach((item) =>
      document
        .querySelector(`[data-item-id="${item.id}"]`)
        ?.classList.add("incomplete"),
    );
    throw new Error(
      `还有 ${missing.length} 项未判断：${missing.map((item) => item.id).join("、")}`,
    );
  }
}

async function save(status) {
  if (!state.sample || state.loading) return;
  if (status === "submitted") validateHuman();
  const buttons = [$("save-button"), $("submit-button")];
  buttons.forEach((button) => {
    button.disabled = true;
  });
  try {
    const result = await api("/api/review", {
      method: "POST",
      body: JSON.stringify({
        seq: state.sample.seq,
        protocol_version: state.rubric.version,
        status,
        gates: collectGates(),
        items: collectItems(),
        notes: $("notes").value,
      }),
    });
    updateProgress(result.progress);
    setDirty(false);
    $("save-state").textContent =
      status === "submitted" ? `已提交 · ${result.decision}` : "草稿已保存";
    toast(status === "submitted" ? "V2 审核已提交" : "草稿已保存");
    if (status === "submitted" && result.next_seq)
      await loadSample(result.next_seq);
  } catch (error) {
    toast(error.message, true);
  } finally {
    buttons.forEach((button) => {
      button.disabled = false;
    });
  }
}

function formatMoney(value) {
  return `$${Number(value || 0).toFixed(5)}`;
}

function formatPercent(value) {
  return value === null || value === undefined
    ? "—"
    : `${(100 * Number(value)).toFixed(1)}%`;
}

function renderScoreSummary(root, scoreSummary) {
  if (!scoreSummary) return;
  const summary = document.createElement("section");
  summary.className = "llm-score-summary";
  const overall = document.createElement("div");
  overall.className = "llm-overall-score";
  overall.innerHTML = `<strong></strong><span>综合质量分</span>`;
  overall.querySelector("strong").textContent =
    scoreSummary.quality_score === null
      ? "—"
      : Number(scoreSummary.quality_score).toFixed(1);
  const diagnostics = document.createElement("div");
  diagnostics.className = "score-diagnostics";
  diagnostics.textContent = `可用评分分布 ${scoreSummary.logprob_usable_items}/${scoreSummary.scored_items} · 覆盖率 ${formatPercent(scoreSummary.logprob_coverage)} · 平均分布熵 ${scoreSummary.mean_distribution_entropy ?? "—"}`;
  const groups = document.createElement("div");
  groups.className = "group-score-grid";
  Object.entries(scoreSummary.group_scores || {}).forEach(([id, record]) => {
    const card = document.createElement("div");
    card.innerHTML = `<span></span><strong></strong>`;
    card.querySelector("span").textContent = `${id} · ${record.name}`;
    card.querySelector("strong").textContent = Number(record.score).toFixed(1);
    groups.append(card);
  });
  summary.append(overall, diagnostics, groups);
  root.append(summary);
}

function renderScoreItems(root, scores) {
  const entries = Object.entries(scores || {});
  if (!entries.length) return;
  const details = document.createElement("details");
  details.className = "score-details";
  details.open = true;
  const heading = document.createElement("summary");
  heading.textContent = `查看 ${entries.length} 项评分与概率`;
  const grid = document.createElement("div");
  grid.className = "score-result-grid";
  entries.forEach(([id, record]) => {
    const item = document.createElement("article");
    item.className = "score-result-item";
    const top = document.createElement("div");
    top.className = "score-result-top";
    const title = document.createElement("strong");
    title.textContent = `${id} · ${record.value}`;
    const expected = document.createElement("span");
    expected.textContent =
      record.expected_score === undefined || record.expected_score === null
        ? "离散分"
        : `期望 ${Number(record.expected_score).toFixed(3)}`;
    top.append(title, expected);
    const meta = document.createElement("div");
    meta.className = "score-result-meta";
    meta.textContent = `低于2分 ${formatPercent(record.low_quality_probability)} · N/A ${formatPercent(record.na_probability)} · ${record.distribution_method || "无分布"}`;
    item.append(top, meta);
    if (record.distribution) {
      const bars = document.createElement("div");
      bars.className = "distribution-bars";
      Object.entries(record.distribution).forEach(([value, probability]) => {
        const bar = document.createElement("div");
        bar.innerHTML = `<span></span><i><b></b></i><em></em>`;
        bar.querySelector("span").textContent = value;
        bar.querySelector("b").style.width =
          `${Math.max(0, Math.min(100, 100 * Number(probability)))}%`;
        bar.querySelector("em").textContent = formatPercent(probability);
        bars.append(bar);
      });
      item.append(bars);
    }
    grid.append(item);
  });
  details.append(heading, grid);
  root.append(details);
}

function renderLlmResult(result, phase) {
  const decision = result.decision.decision;
  const root = $("llm-result");
  root.className = `llm-result result-${decision}`;
  root.replaceChildren();
  const title = document.createElement("div");
  title.className = "result-title";
  const labels = {
    keep: "保留候选",
    review: "需要独立复核",
    reject: "确认硬错误，淘汰",
    exclude: "确认质量缺陷，排除",
  };
  title.textContent = `${phase === "primary" ? "主审核" : "独立复核"}：${labels[decision] || decision}`;
  const meta = document.createElement("div");
  meta.className = "result-meta";
  meta.textContent = `${result.model_resolved || result.model} · ${result.provider || "provider 未返回"} · ${result.usage.prompt_tokens} 输入 / ${result.usage.completion_tokens} 输出 · ${formatMoney(result.usage.cost_usd)} · ${result.latency_ms}ms`;
  root.append(title, meta);
  renderScoreSummary(root, result.output.score_summary);
  const targets =
    phase === "primary"
      ? result.decision.verification_targets
      : {
          gates: result.decision.confirmed_gates,
          items: result.decision.confirmed_items,
        };
  const lines = [];
  if (targets?.gates?.length) lines.push(`Gate：${targets.gates.join("、")}`);
  if (targets?.items?.length) lines.push(`原子项：${targets.items.join("、")}`);
  if (result.decision.applicability_conflicts?.length)
    lines.push(
      `适用性冲突：${result.decision.applicability_conflicts.join("、")}`,
    );
  if (lines.length) {
    const list = document.createElement("ul");
    list.className = "result-list";
    lines.forEach((line) => {
      const li = document.createElement("li");
      li.textContent = line;
      list.append(li);
    });
    root.append(list);
  }
  renderScoreItems(root, result.output.scores);
  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = "查看结构化输出";
  const pre = document.createElement("pre");
  pre.textContent = JSON.stringify(result.output, null, 2);
  details.append(summary, pre);
  root.append(details);
  root.classList.remove("hidden");
}

async function runPrimary() {
  if (!state.sample) return;
  const model = $("primary-model").value;
  if (!model) return toast("请选择模型", true);
  $("run-primary").disabled = true;
  $("llm-status").textContent = "正在审核当前样本，请保持页面打开……";
  try {
    const result = await api("/api/llm/primary", {
      method: "POST",
      body: JSON.stringify({ seq: state.sample.seq, model }),
    });
    state.primaryRun = result;
    renderLlmResult(result, "primary");
    if (result.decision.decision === "review") {
      $("verify-controls").classList.remove("hidden");
      [...$("verify-model").options].forEach((option) => {
        option.disabled =
          option.value.split("/", 1)[0] === model.split("/", 1)[0];
      });
      const first = [...$("verify-model").options].find(
        (option) => !option.disabled,
      );
      if (first) $("verify-model").value = first.value;
    } else {
      $("verify-controls").classList.add("hidden");
    }
    $("llm-status").textContent = "主审核完成，结果已保存到 V2 独立结果库。";
  } catch (error) {
    $("llm-status").textContent = "";
    toast(error.message, true);
  } finally {
    $("run-primary").disabled = false;
  }
}

async function runVerification() {
  if (!state.primaryRun) return;
  const model = $("verify-model").value;
  $("run-verify").disabled = true;
  $("llm-status").textContent = "正在用不同模型家族独立复核争议项……";
  try {
    const result = await api("/api/llm/verify", {
      method: "POST",
      body: JSON.stringify({ run_id: state.primaryRun.run_id, model }),
    });
    renderLlmResult(result, "verification");
    $("verify-controls").classList.add("hidden");
    $("llm-status").textContent = "独立复核完成，最终落档结果已保存。";
  } catch (error) {
    $("llm-status").textContent = "";
    toast(error.message, true);
  } finally {
    $("run-verify").disabled = false;
  }
}

async function showPrompt() {
  if (!state.sample) return;
  $("prompt-button").disabled = true;
  try {
    const preview = await api(`/api/prompt-preview/${state.sample.seq}`);
    $("preview-gate").textContent = JSON.stringify(preview.gate, null, 2);
    $("preview-score").textContent = JSON.stringify(
      preview.score_requests,
      null,
      2,
    );
    $("preview-schema").textContent = JSON.stringify(
      {
        gate: preview.gate.schema,
        score_groups: preview.score_requests.map((request) => ({
          group: request.group,
          schema: request.schema,
        })),
      },
      null,
      2,
    );
    $("prompt-dialog").showModal();
  } catch (error) {
    toast(error.message, true);
  } finally {
    $("prompt-button").disabled = false;
  }
}

function canNavigate() {
  return !state.dirty || window.confirm("当前修改尚未保存，确定离开吗？");
}

async function enterApp(session) {
  state.username = session.username;
  $("user-label").textContent = session.username;
  $("login-view").classList.add("hidden");
  $("app-view").classList.remove("hidden");
  updateProgress(session.progress);
  state.rubric = await api("/api/rubric");
  renderRubric();
  const config = await api("/api/llm/config");
  state.models = config.models || [];
  const modelMeta = new Map(
    (config.catalog?.models || []).map((entry) => [entry.id, entry]),
  );
  [$("primary-model"), $("verify-model")].forEach((select) => {
    select.replaceChildren();
    state.models.forEach((model) => {
      const option = document.createElement("option");
      option.value = model;
      option.textContent = `${model}${modelMeta.get(model)?.reports_token_logprobs ? " · 可请求评分概率" : " · 仅离散分"}`;
      select.append(option);
    });
  });
  $("run-primary").disabled = !config.enabled || !state.models.length;
  await loadSample(session.next_seq || 1);
}

document.addEventListener("DOMContentLoaded", async () => {
  $("login-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      const session = await api("/api/login", {
        method: "POST",
        body: JSON.stringify({ username: $("username").value }),
      });
      await enterApp(session);
    } catch (error) {
      $("login-error").textContent = error.message;
    }
  });
  $("logout-button").addEventListener("click", async () => {
    if (!canNavigate()) return;
    await api("/api/logout", { method: "POST", body: "{}" });
    window.location.reload();
  });
  $("prev-button").addEventListener("click", () => {
    if (canNavigate()) loadSample(state.sample.seq - 1);
  });
  $("next-button").addEventListener("click", () => {
    if (canNavigate()) loadSample(state.sample.seq + 1);
  });
  $("jump-button").addEventListener("click", () => {
    if (canNavigate()) loadSample(Number($("jump-input").value));
  });
  $("jump-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter") $("jump-button").click();
  });
  $("notes").addEventListener("input", () => setDirty(true));
  $("save-button").addEventListener("click", () => save("draft"));
  $("submit-button").addEventListener("click", () => save("submitted"));
  $("run-primary").addEventListener("click", runPrimary);
  $("run-verify").addEventListener("click", runVerification);
  $("prompt-button").addEventListener("click", showPrompt);
  window.addEventListener("beforeunload", (event) => {
    if (state.dirty) event.preventDefault();
  });
  try {
    const session = await api("/api/me");
    await enterApp(session);
  } catch (_) {
    $("login-view").classList.remove("hidden");
  }
});
