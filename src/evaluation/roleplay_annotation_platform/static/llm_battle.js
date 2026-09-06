"use strict";

const PAGE_SIZE = 50;
const state = {
  config: null,
  rubric: null,
  jobs: [],
  selectedJobId: null,
  selectedPairSeq: null,
  pairOffset: 0,
  pairTotal: 0,
  weightFit: null,
  fittingWeights: false,
  pollTimer: null,
};
const el = (id) => document.getElementById(id);

const statusLabels = {
  queued: "等待中",
  running: "运行中",
  paused: "已暂停",
  completed: "已完成",
  completed_with_errors: "完成但有错误",
  cancelled: "已停止",
  budget_exhausted: "预算已用完",
  pending: "待处理",
  discarded: "Gate 淘汰",
  invalid: "结果无效",
  failed: "调用失败",
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  let payload = {};
  try {
    payload = await response.json();
  } catch (_) {
    payload = {};
  }
  if (!response.ok) {
    const error = new Error(payload.detail || `请求失败（${response.status}）`);
    error.status = response.status;
    throw error;
  }
  return payload;
}

function toast(message, error = false) {
  const box = el("toast");
  box.textContent = message;
  box.classList.toggle("error", error);
  box.classList.remove("hidden");
  window.clearTimeout(toast.timer);
  toast.timer = window.setTimeout(() => box.classList.add("hidden"), 3500);
}

function renderProviderRoute() {
  const model = el("judge-battle-model").value;
  const entry = (state.config.models || []).find(
    (candidate) => candidate.id === model,
  );
  if (entry?.gateway === "tatu") {
    el("judge-battle-provider-route").textContent =
      `渠道：TatuCloud · 精确路由 ${entry.deployer || "已配置部署商"} · Chat Completions · Token 概率与美元费用不回传 · 长输出建议先单条试跑`;
    return;
  }
  const routing = state.config.provider_routing || {};
  const preferred =
    (routing.preferred_order_by_model || {})[model] ||
    routing.preferred_order ||
    [];
  const ignored = [
    ...new Set([
      ...(routing.ignore || []),
      ...((routing.ignore_by_model || {})[model] || []),
    ]),
  ];
  const parts = [];
  parts.push(
    preferred.length
      ? `该模型支持时优先 ${preferred.join(" → ")}`
      : "由 OpenRouter 自动选择渠道",
  );
  if (ignored.length) parts.push(`排除 ${ignored.join("、")}`);
  parts.push(
    routing.allow_fallbacks ? "不可用时自动切换其他渠道" : "不允许其他渠道兜底",
  );
  if (entry?.reports_token_logprobs !== true) {
    parts.push("当前渠道不提供可信的评分 Token 概率");
  }
  el("judge-battle-provider-route").textContent = `渠道：${parts.join(" · ")}`;
}

function renderModelCatalog() {
  const select = el("judge-battle-model");
  const current = select.value;
  const query = el("judge-battle-model-search")
    .value.trim()
    .toLocaleLowerCase();
  const models = state.config.models.filter((entry) => {
    const searchable =
      `${entry.id} ${entry.name || ""} ${entry.label || ""}`.toLocaleLowerCase();
    return !query || searchable.includes(query);
  });
  select.replaceChildren();
  if (!models.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = state.config.models.length
      ? "没有匹配模型"
      : "服务器尚未配置模型";
    select.append(option);
    select.disabled = true;
  } else {
    models.forEach((entry) => {
      const option = document.createElement("option");
      option.value = entry.id;
      option.textContent = entry.label;
      select.append(option);
    });
    select.disabled = false;
    if (models.some((entry) => entry.id === current)) select.value = current;
  }
  const catalog = state.config.model_catalog;
  const source = catalog.source?.includes("tatu")
    ? `混合目录（${catalog.source}）`
    : catalog.source === "openrouter"
      ? "OpenRouter 实时目录"
      : "服务器备用目录";
  const stale = catalog.stale ? " · 当前为缓存/备用数据" : "";
  el("judge-battle-model-catalog").textContent =
    `${source} · ${state.config.models.length} 个兼容模型${stale}`;
  renderProviderRoute();
}

async function refreshModelCatalog() {
  const button = el("judge-battle-refresh-models");
  button.disabled = true;
  button.textContent = "刷新中…";
  try {
    const payload = await api("/api/llm-judge/models/refresh", {
      method: "POST",
    });
    state.config.models = payload.models;
    const { models: _, ...catalog } = payload;
    state.config.model_catalog = catalog;
    renderModelCatalog();
    toast(
      payload.stale
        ? "刷新失败，继续使用服务器备用目录"
        : `已加载 ${payload.count} 个兼容模型`,
      payload.stale,
    );
  } catch (error) {
    toast(error.message, true);
  } finally {
    button.disabled = false;
    button.textContent = "刷新目录";
  }
}

function scopePayload() {
  const type = el("judge-battle-scope").value;
  const payload = { scope_type: type };
  if (type === "current")
    payload.pair_seq = Number(el("judge-battle-pair").value);
  if (type === "range") {
    payload.start_pair = Number(el("judge-battle-start").value);
    payload.end_pair = Number(el("judge-battle-end").value);
  }
  if (type === "random")
    payload.random_count = Number(el("judge-battle-random-count").value);
  return payload;
}

function scopeDescription(scope) {
  if (scope.type === "battle") {
    if (scope.selection === "current") return `Battle ${scope.pair_seq}`;
    if (scope.selection === "range")
      return `Battle ${scope.start_pair}–${scope.end_pair}`;
    if (scope.selection === "random") return `随机 ${scope.count} 组 Battle`;
    if (scope.selection === "all") return "全部 Battle";
  }
  return "Battle 任务";
}

function renderScopeFields() {
  const type = el("judge-battle-scope").value;
  ["current", "range", "random"].forEach((name) => {
    el(`judge-battle-scope-${name}`).classList.toggle("hidden", name !== type);
  });
}

function requestedPairCount(payload) {
  if (payload.scope_type === "current") return 1;
  if (payload.scope_type === "range")
    return Math.max(0, payload.end_pair - payload.start_pair + 1);
  if (payload.scope_type === "random") return payload.random_count;
  return state.config.battle_limits.pair_count;
}

function renderJobs() {
  const root = el("judge-battle-history");
  root.replaceChildren();
  if (!state.jobs.length) {
    const empty = document.createElement("p");
    empty.className = "llm-help";
    empty.textContent = "还没有 LLM Judge Battle 任务。";
    root.append(empty);
    return;
  }
  state.jobs.forEach((job) => {
    const button = document.createElement("button");
    button.className = "llm-job";
    button.classList.toggle("selected", job.job_id === state.selectedJobId);
    const title = document.createElement("strong");
    title.textContent = `${scopeDescription(job.scope)} · ${statusLabels[job.status] || job.status}`;
    const model = document.createElement("span");
    model.textContent = job.model;
    const progress = document.createElement("span");
    progress.textContent = `${Math.floor(job.finished_count / 2)}/${job.scope.pair_count} 组 · ${job.finished_count}/${job.total_count} 候选 · $${job.actual_cost_usd.toFixed(4)}`;
    button.append(title, model, progress);
    button.addEventListener("click", () => selectJob(job.job_id));
    root.append(button);
  });
}

async function loadJobs(selectNewest = false) {
  const payload = await api("/api/llm-judge/battles");
  state.jobs = payload.battles;
  if (selectNewest && state.jobs.length)
    state.selectedJobId = state.jobs[0].job_id;
  renderJobs();
  if (state.selectedJobId) await loadSelectedJob();
}

function renderJob(job) {
  el("judge-battle-empty").classList.add("hidden");
  el("judge-battle-detail").classList.remove("hidden");
  el("judge-battle-title").textContent = scopeDescription(job.scope);
  el("judge-battle-status").textContent =
    statusLabels[job.status] || job.status;
  el("judge-battle-status").classList.toggle(
    "error",
    ["completed_with_errors", "budget_exhausted"].includes(job.status),
  );
  const meta = el("judge-battle-meta");
  meta.replaceChildren();
  [
    `模型：${job.model}`,
    `范围：${scopeDescription(job.scope)}`,
    `费用：$${job.actual_cost_usd.toFixed(4)} / $${job.budget_limit_usd.toFixed(2)}`,
  ].forEach((text) => {
    const span = document.createElement("span");
    span.textContent = text;
    meta.append(span);
  });
  const percent = job.total_count
    ? (job.finished_count / job.total_count) * 100
    : 0;
  el("judge-battle-progress-fill").style.width = `${percent}%`;
  const counts = job.counts;
  el("judge-battle-progress-text").textContent =
    `候选完成 ${counts.completed} · Gate淘汰 ${counts.discarded} · 运行 ${counts.running} · 等待 ${counts.pending} · 无效 ${counts.invalid} · 失败 ${counts.failed} · 取消 ${counts.cancelled}`;
  el("judge-battle-pause").disabled = !["queued", "running"].includes(
    job.status,
  );
  el("judge-battle-resume").disabled = job.status !== "paused";
  el("judge-battle-cancel").disabled = ![
    "queued",
    "running",
    "paused",
  ].includes(job.status);
  const fitButton = el("judge-battle-fit-weights");
  fitButton.disabled =
    state.fittingWeights || Boolean(counts.pending || counts.running);
  fitButton.textContent = state.fittingWeights
    ? "拟合中…"
    : state.weightFit
      ? "重新拟合"
      : "拟合权重";
}

function percent(value, digits = 1) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? `${(numeric * 100).toFixed(digits)}%` : "—";
}

function metricCard(label, value, detail) {
  const card = document.createElement("div");
  card.className = "judge-battle-fit-metric";
  const name = document.createElement("span");
  name.textContent = label;
  const main = document.createElement("strong");
  main.textContent = value;
  const foot = document.createElement("small");
  foot.textContent = detail;
  card.append(name, main, foot);
  return card;
}

function weightBar(value) {
  const track = document.createElement("div");
  track.className = "judge-battle-weight-bar";
  const fill = document.createElement("span");
  fill.style.width = `${Math.max(0, Math.min(100, Number(value) * 100))}%`;
  track.append(fill);
  return track;
}

function renderWeightFit(fit) {
  state.weightFit = fit || null;
  const empty = el("judge-battle-weight-fit-empty");
  const report = el("judge-battle-weight-fit-report");
  const button = el("judge-battle-fit-weights");
  button.textContent = state.fittingWeights
    ? "拟合中…"
    : fit
      ? "重新拟合"
      : "拟合权重";
  if (!fit) {
    empty.classList.remove("hidden");
    report.classList.add("hidden");
    return;
  }
  empty.classList.add("hidden");
  report.classList.remove("hidden");

  const warnings = el("judge-battle-weight-fit-warnings");
  warnings.replaceChildren();
  if (fit.recommendation?.meaning) {
    const recommendation = document.createElement("p");
    recommendation.className = `judge-battle-fit-warning recommendation-${fit.recommendation.status || "unknown"}`;
    recommendation.textContent = `使用结论：${fit.recommendation.meaning}`;
    warnings.append(recommendation);
  }
  (fit.warnings || []).forEach((message) => {
    const warning = document.createElement("p");
    warning.className = "judge-battle-fit-warning";
    warning.textContent = message;
    warnings.append(warning);
  });

  const baseline = fit.metrics.equal_weight_baseline;
  const training = fit.metrics.training;
  const validation = fit.metrics.cross_validation;
  const metrics = el("judge-battle-weight-fit-metrics");
  metrics.replaceChildren(
    metricCard(
      "评分对五折 AUC",
      Number(validation.scored_auc ?? validation.auc).toFixed(3),
      `等权基线 ${Number(baseline.scored_auc ?? baseline.auc).toFixed(3)}`,
    ),
    metricCard(
      "评分对方向命中",
      percent(
        validation.scored_accuracy ??
          validation.direction_accuracy_excluding_ties,
      ),
      `训练集 ${percent(training.scored_accuracy ?? training.direction_accuracy_excluding_ties)}`,
    ),
    metricCard(
      "五折 Log loss",
      Number(validation.log_loss).toFixed(3),
      `Brier ${Number(validation.brier_score).toFixed(3)}`,
    ),
    metricCard(
      "单边 Gate 命中",
      percent(validation.gate_accuracy),
      `${fit.data.one_sided_gate_pairs} 对 · 与评分项分开`,
    ),
    metricCard(
      "可用 Battle",
      String(fit.data.usable_pairs),
      `${fit.data.scored_pairs} 评分对 · ${fit.data.one_sided_gate_pairs} 单边Gate`,
    ),
  );

  const dimensionRoot = el("judge-battle-dimension-weights");
  dimensionRoot.replaceChildren();
  const dimensionTable = document.createElement("table");
  dimensionTable.className = "judge-battle-compare-table";
  const dimensionHead = document.createElement("thead");
  const dimensionHeader = document.createElement("tr");
  ["维度", "最终权重", "比例", "跨折均值 ± 波动"].forEach((text) => {
    const th = document.createElement("th");
    th.textContent = text;
    dimensionHeader.append(th);
  });
  dimensionHead.append(dimensionHeader);
  dimensionTable.append(dimensionHead);
  const dimensionBody = document.createElement("tbody");
  fit.parameters.dimensions.forEach((entry) => {
    const row = document.createElement("tr");
    const name = document.createElement("td");
    name.textContent = `${entry.id} · ${entry.name}`;
    const value = document.createElement("td");
    value.textContent = percent(entry.weight, 2);
    const bar = document.createElement("td");
    bar.append(weightBar(entry.weight));
    const stability = document.createElement("td");
    stability.textContent = `${percent(entry.cross_fold_mean, 2)} ± ${percent(entry.cross_fold_std, 2)}`;
    row.append(name, value, bar, stability);
    dimensionBody.append(row);
  });
  dimensionTable.append(dimensionBody);
  dimensionRoot.append(dimensionTable);

  const parameterRoot = el("judge-battle-confidence-weights");
  parameterRoot.replaceChildren();
  [
    ["分差温度", fit.parameters.temperature],
    ["单边 Gate 隐含胜率", fit.parameters.gate_implied_probability],
  ].forEach(([label, value], index) => {
    const card = document.createElement("div");
    card.className = "judge-battle-fit-parameter";
    const name = document.createElement("span");
    name.textContent = label;
    const main = document.createElement("strong");
    main.textContent =
      index === 1 ? percent(value, 2) : Number(value).toFixed(3);
    card.append(name, main);
    parameterRoot.append(card);
  });

  const itemRoot = el("judge-battle-item-weights");
  itemRoot.replaceChildren();
  const itemTable = document.createElement("table");
  itemTable.className = "judge-battle-compare-table";
  const itemHead = document.createElement("thead");
  const itemHeader = document.createElement("tr");
  [
    "原子项",
    "所属维度",
    "总权重",
    "产生分差",
    "分差方向命中",
    "跨折波动",
  ].forEach((text) => {
    const th = document.createElement("th");
    th.textContent = text;
    itemHeader.append(th);
  });
  itemHead.append(itemHeader);
  itemTable.append(itemHead);
  const itemBody = document.createElement("tbody");
  const signalByItem = new Map(
    (fit.score_signal?.items || []).map((entry) => [entry.id, entry]),
  );
  fit.parameters.items.forEach((entry) => {
    const row = document.createElement("tr");
    const signal = signalByItem.get(entry.id) || {};
    [
      `${entry.id} · ${entry.name}`,
      entry.dimension_id,
      percent(entry.overall_weight, 2),
      `${percent(signal.different_rate)} (${signal.different_pairs ?? "—"})`,
      signal.direction_accuracy_when_different === null ||
      signal.direction_accuracy_when_different === undefined
        ? "—"
        : percent(signal.direction_accuracy_when_different),
      `± ${percent(entry.cross_fold_std, 2)}`,
    ].forEach((value) => {
      const td = document.createElement("td");
      td.textContent = value;
      row.append(td);
    });
    itemBody.append(row);
  });
  itemTable.append(itemBody);
  itemRoot.append(itemTable);

  const methodRoot = el("judge-battle-fit-method");
  methodRoot.replaceChildren();
  [
    `公式：${fit.method.score_formula}`,
    `权重：${fit.method.weight_constraints}`,
    `N/A：${fit.method.na_policy}`,
    `Gate：${fit.method.gate_policy}`,
    `数据：双方N/A ${fit.data.both_na_items} 项；单边N/A ${fit.data.one_sided_na_items} 项，涉及 ${fit.data.pairs_with_one_sided_na} 对。`,
  ].forEach((text) => {
    const p = document.createElement("p");
    p.textContent = text;
    methodRoot.append(p);
  });
}

function candidateStatus(pair, label) {
  return (
    pair.candidates.find((entry) => entry.label === label) || {
      status: "pending",
    }
  );
}

function renderPairs(page) {
  state.pairTotal = page.total;
  const root = el("judge-battle-pairs");
  root.replaceChildren();
  page.pairs.forEach((pair) => {
    const row = document.createElement("tr");
    if (pair.pair_seq === state.selectedPairSeq)
      row.classList.add("selected-row");
    [pair.position, pair.pair_seq].forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = String(value);
      row.append(cell);
    });
    ["A", "B"].forEach((label) => {
      const item = candidateStatus(pair, label);
      const cell = document.createElement("td");
      cell.textContent = statusLabels[item.status] || item.status;
      cell.className = `status-${item.status}`;
      row.append(cell);
    });
    const action = document.createElement("td");
    const button = document.createElement("button");
    button.className = "ghost small";
    button.textContent = "对比";
    button.addEventListener("click", () => selectPair(pair.pair_seq));
    action.append(button);
    row.append(action);
    row.addEventListener("dblclick", () => selectPair(pair.pair_seq));
    root.append(row);
  });
  const start = page.total ? page.offset + 1 : 0;
  const end = Math.min(page.offset + page.pairs.length, page.total);
  el("judge-battle-page-label").textContent = `${start}–${end} / ${page.total}`;
  el("judge-battle-page-prev").disabled = page.offset <= 0;
  el("judge-battle-page-next").disabled =
    page.offset + page.pairs.length >= page.total;
}

async function selectJob(jobId) {
  state.selectedJobId = jobId;
  state.selectedPairSeq = null;
  state.weightFit = null;
  state.pairOffset = 0;
  el("judge-battle-pair-detail").classList.add("hidden");
  renderJobs();
  await loadSelectedJob();
}

async function loadSelectedJob() {
  if (!state.selectedJobId) return;
  const [job, page, weightFitPayload] = await Promise.all([
    api(`/api/llm-judge/battles/${state.selectedJobId}`),
    api(
      `/api/llm-judge/battles/${state.selectedJobId}/pairs?offset=${state.pairOffset}&limit=${PAGE_SIZE}`,
    ),
    api(`/api/llm-judge/battles/${state.selectedJobId}/weight-fit`),
  ]);
  const index = state.jobs.findIndex((entry) => entry.job_id === job.job_id);
  if (index >= 0) state.jobs[index] = job;
  renderJobs();
  renderWeightFit(weightFitPayload.fit);
  renderJob(job);
  renderPairs(page);
  if (state.selectedPairSeq !== null)
    await loadPair(state.selectedPairSeq, false);
}

async function fitWeights() {
  if (!state.selectedJobId || state.fittingWeights) return;
  state.fittingWeights = true;
  const selected = state.jobs.find((job) => job.job_id === state.selectedJobId);
  if (selected) renderJob(selected);
  try {
    const payload = await api(
      `/api/llm-judge/battles/${state.selectedJobId}/weight-fit`,
      {
        method: "POST",
        body: "{}",
      },
    );
    renderWeightFit(payload.fit);
    toast(
      `权重拟合完成：五折 AUC ${Number(payload.fit.metrics.cross_validation.auc).toFixed(3)}`,
    );
  } catch (error) {
    toast(error.message, true);
  } finally {
    state.fittingWeights = false;
    const current = state.jobs.find(
      (job) => job.job_id === state.selectedJobId,
    );
    if (current) renderJob(current);
  }
}

function appendContextMessage(root, role, content, index = null) {
  const card = document.createElement("article");
  card.className = "context-message";
  const header = document.createElement("header");
  header.textContent = index === null ? role : `${index + 1}. ${role}`;
  const body = document.createElement("pre");
  body.textContent = content;
  card.append(header, body);
  root.append(card);
}

function resultByLabel(payload, label) {
  return (
    payload.results.find((entry) => entry.label === label) || {
      label,
      item: { status: "pending" },
      judgment: null,
    }
  );
}

function scoreValue(record) {
  if (!record) return null;
  const value = record.value;
  if (value === "N/A") return "N/A";
  if (typeof value === "boolean") return value;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : value;
}

function candidateScores(result) {
  return result.judgment?.judgment?.scores || null;
}

function candidateGates(result) {
  return result.judgment?.judgment?.gates || null;
}

function renderCandidate(candidate, result, model) {
  const article = document.createElement("article");
  article.className = `judge-battle-candidate ${candidate.label.toLowerCase()}`;
  const header = document.createElement("header");
  const title = document.createElement("strong");
  title.textContent = `候选 ${candidate.label}`;
  const identity = document.createElement("span");
  identity.className = `candidate-model candidate-model-${candidate.label.toLowerCase()}`;
  identity.textContent = model || "原模型已隐藏";
  header.append(title, identity);
  const content = document.createElement("pre");
  content.textContent = candidate.content;
  const meta = document.createElement("div");
  meta.className = "judge-battle-candidate-meta";
  const judgment = result.judgment;
  const lines = [
    `状态：${statusLabels[result.item.status] || result.item.status}`,
    judgment?.provider ? `渠道：${judgment.provider}` : "渠道：—",
    judgment
      ? `Token：${judgment.usage.prompt_tokens} 输入 / ${judgment.usage.completion_tokens} 输出`
      : "Token：—",
    judgment
      ? `费用：$${judgment.usage.cost_usd.toFixed(5)} · ${judgment.usage.latency_ms}ms`
      : "费用：—",
  ];
  lines.forEach((text) => {
    const span = document.createElement("span");
    span.textContent = text;
    meta.append(span);
  });
  const request = document.createElement("button");
  request.className = "ghost small";
  request.textContent = `查看 ${candidate.label} 请求`;
  request.addEventListener("click", () =>
    showRequest(candidate.seq, candidate.label),
  );
  meta.append(request);
  if (judgment?.error_message || result.item.error_message) {
    const error = document.createElement("p");
    error.className = "judge-battle-result-error";
    error.textContent = judgment?.error_message || result.item.error_message;
    meta.append(error);
  }
  if (judgment?.raw_output && Object.keys(judgment.raw_output).length) {
    const details = document.createElement("details");
    details.className = "judge-battle-raw-output";
    const summary = document.createElement("summary");
    summary.textContent = "查看模型原始输出（未校验）";
    const raw = document.createElement("pre");
    raw.textContent = JSON.stringify(judgment.raw_output, null, 2);
    details.append(summary, raw);
    meta.append(details);
  }
  article.append(header, content, meta);
  return article;
}

function renderGateComparison(a, b) {
  const root = el("judge-battle-gates");
  root.replaceChildren();
  const table = document.createElement("table");
  table.className = "judge-battle-compare-table";
  const head = document.createElement("thead");
  const headerRow = document.createElement("tr");
  ["Gate", "候选 A", "候选 B", "是否一致"].forEach((text) => {
    const th = document.createElement("th");
    th.textContent = text;
    headerRow.append(th);
  });
  head.append(headerRow);
  table.append(head);
  const body = document.createElement("tbody");
  const ga = candidateGates(a);
  const gb = candidateGates(b);
  const gateIds = state.rubric.gates.flatMap((group) =>
    group.items.map((item) => item.id),
  );
  gateIds.forEach((id) => {
    const row = document.createElement("tr");
    const va = scoreValue(ga?.[id]);
    const vb = scoreValue(gb?.[id]);
    [
      id,
      va ?? "—",
      vb ?? "—",
      va !== null && vb !== null ? (va === vb ? "一致" : "不同") : "不可比较",
    ].forEach((value) => {
      const td = document.createElement("td");
      td.textContent = String(value);
      row.append(td);
    });
    body.append(row);
  });
  table.append(body);
  root.append(table);
}

function pairedDimensionRows(a, b) {
  const sa = candidateScores(a);
  const sb = candidateScores(b);
  return state.rubric.dimensions.map((dimension) => {
    let sumA = 0;
    let sumB = 0;
    let common = 0;
    dimension.items.forEach((item) => {
      const va = scoreValue(sa?.[item.id]);
      const vb = scoreValue(sb?.[item.id]);
      if (typeof va === "number" && typeof vb === "number") {
        sumA += va;
        sumB += vb;
        common += 1;
      }
    });
    return {
      id: dimension.id,
      name: dimension.name,
      sumA,
      sumB,
      common,
      diff: sumA - sumB,
    };
  });
}

function renderDimensionSummary(a, b) {
  const root = el("judge-battle-dimensions");
  root.replaceChildren();
  const rows = pairedDimensionRows(a, b);
  const table = document.createElement("table");
  table.className = "judge-battle-compare-table";
  const head = document.createElement("thead");
  const headerRow = document.createElement("tr");
  ["维度", "共同可评分项", "A 合计", "B 合计", "A−B"].forEach((text) => {
    const th = document.createElement("th");
    th.textContent = text;
    headerRow.append(th);
  });
  head.append(headerRow);
  table.append(head);
  const body = document.createElement("tbody");
  rows.forEach((entry) => {
    const row = document.createElement("tr");
    [
      entry.name,
      entry.common,
      entry.common ? entry.sumA : "—",
      entry.common ? entry.sumB : "—",
      entry.common ? `${entry.diff > 0 ? "+" : ""}${entry.diff}` : "—",
    ].forEach((value) => {
      const td = document.createElement("td");
      td.textContent = String(value);
      row.append(td);
    });
    body.append(row);
  });
  table.append(body);
  root.append(table);
}

function scoreSide(label, record) {
  const side = document.createElement("div");
  side.className = "judge-battle-score-side";
  const heading = document.createElement("strong");
  heading.textContent = `候选 ${label}`;
  const value = document.createElement("span");
  value.className = "judge-battle-score-value";
  const normalized = scoreValue(record);
  value.textContent =
    normalized === null
      ? "缺失"
      : typeof normalized === "number"
        ? `${normalized} 分`
        : String(normalized);
  if (normalized === null) value.classList.add("judge-battle-missing");
  const tokenProbability = Number(record?.logprob_confidence);
  const hasTokenProbability =
    record?.logprob_confidence !== null &&
    record?.logprob_confidence !== undefined &&
    Number.isFinite(tokenProbability) &&
    tokenProbability >= 0 &&
    tokenProbability <= 1;
  const tokenConfidence = document.createElement("span");
  tokenConfidence.className = `judge-battle-confidence ${hasTokenProbability ? "confidence-token" : "confidence-missing"}`;
  tokenConfidence.textContent = hasTokenProbability
    ? `Token概率：${(tokenProbability * 100).toFixed(1)}%`
    : "Token概率：不可用";
  const distribution = document.createElement("span");
  distribution.className = "judge-battle-confidence confidence-token";
  const scoreDistribution = record?.logprob_distribution;
  distribution.textContent =
    scoreDistribution && typeof scoreDistribution === "object"
      ? `档位分布（${record?.logprob_distribution_complete === true ? "完整" : "部分，仅诊断"}）：${Object.entries(
          scoreDistribution,
        )
          .map(
            ([score, probability]) =>
              `${score} ${(Number(probability) * 100).toFixed(1)}%`,
          )
          .join(" · ")}`
      : "档位分布：不可用";
  const reason = document.createElement("p");
  reason.className = "judge-battle-score-reason";
  reason.textContent = record?.reason || "没有有效评分理由";
  side.append(
    heading,
    document.createElement("br"),
    value,
    tokenConfidence,
    distribution,
    reason,
  );
  (record?.evidence || []).forEach((quote) => {
    const evidence = document.createElement("div");
    evidence.className = "judge-battle-score-evidence";
    evidence.textContent = quote;
    side.append(evidence);
  });
  return side;
}

function renderScoreComparison(a, b) {
  const root = el("judge-battle-scores");
  root.replaceChildren();
  const sa = candidateScores(a);
  const sb = candidateScores(b);
  state.rubric.dimensions.forEach((dimension) => {
    dimension.items.forEach((item) => {
      const ra = sa?.[item.id] || null;
      const rb = sb?.[item.id] || null;
      const va = scoreValue(ra);
      const vb = scoreValue(rb);
      const card = document.createElement("article");
      card.className = "judge-battle-score-card";
      if (va !== vb) card.classList.add("difference");
      const header = document.createElement("header");
      header.className = "judge-battle-score-header";
      const title = document.createElement("strong");
      title.textContent = `${item.id} · ${item.name}`;
      const diff = document.createElement("span");
      diff.textContent =
        typeof va === "number" && typeof vb === "number"
          ? `A−B：${va - vb > 0 ? "+" : ""}${va - vb}`
          : va === vb && va !== null
            ? "一致"
            : "不可计算差值";
      header.append(title, diff);
      const grid = document.createElement("div");
      grid.className = "judge-battle-score-grid";
      grid.append(scoreSide("A", ra), scoreSide("B", rb));
      card.append(header, grid);
      root.append(card);
    });
  });
}

function renderWinner(a, b) {
  const box = el("judge-battle-winner");
  box.className = "judge-battle-winner";
  const statusA = a.item.status;
  const statusB = b.item.status;
  if (statusA === "completed" && statusB === "discarded") {
    box.textContent = "Judge 对比：A 胜出（B 被 Gate 淘汰）";
    box.classList.add("a");
    return;
  }
  if (statusB === "completed" && statusA === "discarded") {
    box.textContent = "Judge 对比：B 胜出（A 被 Gate 淘汰）";
    box.classList.add("b");
    return;
  }
  if (statusA === "discarded" && statusB === "discarded") {
    box.textContent = "Judge 对比：两边都被 Gate 淘汰";
    box.classList.add("tie");
    return;
  }
  if (statusA !== "completed" || statusB !== "completed") {
    box.textContent = "Judge 对比：等待 A、B 都得到可比较结果";
    return;
  }
  const rows = pairedDimensionRows(a, b);
  const common = rows.reduce((sum, row) => sum + row.common, 0);
  const diff = rows.reduce((sum, row) => sum + row.diff, 0);
  if (!common) {
    box.textContent = "Judge 对比：没有双方共同适用的数值评分项";
    return;
  }
  if (diff > 0) {
    box.textContent = `Judge 对比：A 高 ${diff} 分（${common} 个共同可评分项）`;
    box.classList.add("a");
  } else if (diff < 0) {
    box.textContent = `Judge 对比：B 高 ${-diff} 分（${common} 个共同可评分项）`;
    box.classList.add("b");
  } else {
    box.textContent = `Judge 对比：同分（${common} 个共同可评分项）`;
    box.classList.add("tie");
  }
}

function renderLearnedComparison(comparison) {
  const box = el("judge-battle-learned-comparison");
  box.replaceChildren();
  box.className = "judge-battle-learned-comparison";
  if (!comparison) {
    box.classList.add("hidden");
    return;
  }
  const title = document.createElement("strong");
  title.textContent = "拟合权重结果";
  const summary = document.createElement("p");
  if (!comparison.comparable) {
    summary.textContent = comparison.reason || "当前配对无法用拟合权重比较";
    box.append(title, summary);
    return;
  }
  const probability = percent(comparison.a_win_probability, 2);
  const winner =
    comparison.predicted_winner === "tie"
      ? "同分"
      : `${comparison.predicted_winner} 胜`;
  if (comparison.pair_kind === "gate") {
    summary.textContent = `Gate 硬判 ${comparison.filter_winner} 胜；拟合模型预测 ${winner}，A 胜率 ${probability}。`;
    const note = document.createElement("p");
    note.className = "fine-print";
    note.textContent =
      "Gate 系数只用于拟合线上选择；数据筛选仍按 Gate 硬淘汰。";
    box.append(title, summary, note);
    return;
  }
  const difference = Number(comparison.weighted_score_difference || 0);
  summary.textContent = `预测 ${winner} · A 胜率 ${probability} · 归一化加权分差 ${difference >= 0 ? "+" : ""}${difference.toFixed(4)}`;
  const detail = document.createElement("p");
  detail.className = "fine-print";
  detail.textContent = `共同数值项 ${comparison.comparable_items} 个 · 有效权重覆盖 ${percent(comparison.comparable_weight_coverage, 1)} · 双边 N/A ${comparison.both_na_items} 项 · 单边 N/A ${comparison.one_sided_na_items} 项`;
  box.append(title, summary, detail);
  if (comparison.dimension_contributions?.length) {
    const contributions = document.createElement("p");
    contributions.className = "fine-print";
    contributions.textContent = `维度分差贡献：${comparison.dimension_contributions
      .map((entry) => {
        const value = Number(entry.score_difference_contribution || 0);
        return `${entry.id} ${value >= 0 ? "+" : ""}${value.toFixed(4)}`;
      })
      .join(" · ")}`;
    box.append(contributions);
  }
}

function renderPair(payload) {
  const battle = payload.battle;
  const a = resultByLabel(payload, "A");
  const b = resultByLabel(payload, "B");
  el("judge-battle-pair-detail").classList.remove("hidden");
  el("judge-battle-pair-title").textContent = `Battle ${battle.pair_seq}`;
  el("judge-battle-online-choice").textContent =
    `线上用户选择：${battle.online_selected}`;
  el("judge-battle-context-warning").classList.toggle(
    "hidden",
    battle.context_match,
  );
  const context = el("judge-battle-context");
  context.replaceChildren();
  battle.messages.forEach((message, index) =>
    appendContextMessage(
      context,
      message.display_role || message.role,
      message.content,
      index,
    ),
  );
  const candidates = el("judge-battle-candidates");
  candidates.replaceChildren();
  candidates.append(
    renderCandidate(
      battle.candidates.find((entry) => entry.label === "A"),
      a,
      null,
    ),
    renderCandidate(
      battle.candidates.find((entry) => entry.label === "B"),
      b,
      null,
    ),
  );
  renderWinner(a, b);
  renderLearnedComparison(payload.learned_comparison);
  renderGateComparison(a, b);
  renderDimensionSummary(a, b);
  renderScoreComparison(a, b);
}

async function selectPair(pairSeq) {
  state.selectedPairSeq = pairSeq;
  await loadPair(pairSeq, true);
}

async function loadPair(pairSeq, scroll = false) {
  try {
    const payload = await api(
      `/api/llm-judge/battles/${state.selectedJobId}/pairs/${pairSeq}`,
    );
    renderPair(payload);
    if (scroll)
      el("judge-battle-pair-detail").scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
  } catch (error) {
    toast(error.message, true);
  }
}

async function revealModels() {
  if (state.selectedPairSeq === null) return;
  try {
    const payload = await api(`/api/battle/${state.selectedPairSeq}/models`);
    payload.models.forEach((entry) => {
      const target = document.querySelector(
        `.candidate-model-${entry.label.toLowerCase()}`,
      );
      if (target) target.textContent = entry.model_name;
    });
    el("judge-battle-reveal-models").disabled = true;
  } catch (error) {
    toast(error.message, true);
  }
}

async function createJob() {
  if (!state.config.enabled) return;
  const payload = {
    model: el("judge-battle-model").value,
    budget_limit_usd: Number(el("judge-battle-budget").value),
    ...scopePayload(),
  };
  const count = requestedPairCount(payload);
  if (
    !window.confirm(
      `确认使用 ${payload.model} 评分 ${count} 组 Battle（${count * 2} 个候选）？预算上限 $${payload.budget_limit_usd}。`,
    )
  )
    return;
  const button = el("judge-battle-create");
  button.disabled = true;
  try {
    const job = await api("/api/llm-judge/battles", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.selectedJobId = job.job_id;
    state.selectedPairSeq = null;
    state.pairOffset = 0;
    toast("Battle Judge 后台任务已创建");
    await loadJobs();
  } catch (error) {
    toast(error.message, true);
  } finally {
    button.disabled = !state.config.enabled;
  }
}

async function controlJob(action) {
  if (!state.selectedJobId) return;
  if (
    action === "cancel" &&
    !window.confirm(
      "确定停止所有尚未开始的候选吗？正在请求中的候选仍会保存结果。",
    )
  )
    return;
  try {
    await api(`/api/llm-judge/jobs/${state.selectedJobId}/${action}`, {
      method: "POST",
      body: "{}",
    });
    await loadJobs();
  } catch (error) {
    toast(error.message, true);
  }
}

async function showRequest(seq, label) {
  const selectedJob = state.jobs.find(
    (job) => job.job_id === state.selectedJobId,
  );
  const model = selectedJob?.model || el("judge-battle-model").value;
  try {
    const query = new URLSearchParams({ seq: String(seq), model });
    const preview = await api(`/api/llm-judge/request-preview?${query}`);
    el("judge-battle-request-title").textContent =
      `候选 ${label} · 样本 ${seq} · ${model}`;
    const context = el("judge-battle-request-context");
    context.replaceChildren();
    preview.sample_packet.context_messages.forEach((message) =>
      appendContextMessage(
        context,
        message.role,
        message.content,
        message.index,
      ),
    );
    if (preview.sample_packet.platform_output_requirements) {
      appendContextMessage(
        context,
        "platform_output_requirements（平台附加，不属于 user）",
        JSON.stringify(
          preview.sample_packet.platform_output_requirements,
          null,
          2,
        ),
      );
    }
    appendContextMessage(
      context,
      preview.sample_packet.candidate_response.role,
      preview.sample_packet.candidate_response.content,
    );
    el("judge-battle-request-json").textContent = JSON.stringify(
      {
        gate_policy: preview.gate_policy,
        endpoint: preview.endpoint,
        headers: preview.headers,
        phases: preview.phases,
      },
      null,
      2,
    );
    el("judge-battle-request-dialog").showModal();
  } catch (error) {
    toast(error.message, true);
  }
}

async function init() {
  try {
    const [session, config, rubric] = await Promise.all([
      api("/api/me"),
      api("/api/llm-judge/config"),
      api("/api/rubric"),
    ]);
    state.config = config;
    state.rubric = rubric;
    el("judge-battle-current-user").textContent = session.username;
    el("judge-battle-rubric-version").textContent =
      `规则 ${config.rubric_version}`;
    el("judge-battle-rubric-hash").textContent = config.rubric_sha256;
    el("judge-battle-view").classList.remove("hidden");
    renderModelCatalog();
    renderScopeFields();
    const total = config.battle_limits.pair_count;
    [
      "judge-battle-pair",
      "judge-battle-start",
      "judge-battle-end",
      "judge-battle-random-count",
    ].forEach((id) => {
      el(id).max = String(total);
    });
    el("judge-battle-limits").textContent =
      `共 ${total} 组 · 服务器并发 ${config.max_concurrency} · 单任务最多 ${config.battle_limits.max_pairs_per_job} 组 · 最高预算 $${config.max_job_budget_usd}`;
    if (!config.enabled) {
      el("judge-battle-disabled").textContent = config.disabled_reason;
      el("judge-battle-disabled").classList.remove("hidden");
      el("judge-battle-create").disabled = true;
    }
    await loadJobs(true);
    state.pollTimer = window.setInterval(async () => {
      if (!state.selectedJobId || document.hidden) return;
      try {
        await loadJobs();
      } catch (_) {
        /* next poll retries */
      }
    }, 3000);
  } catch (error) {
    if (error.status === 401) {
      window.location.replace("/");
      return;
    }
    el("judge-battle-view").classList.remove("hidden");
    toast(error.message, true);
  }
}

el("judge-battle-model-search").addEventListener("input", renderModelCatalog);
el("judge-battle-model").addEventListener("change", renderProviderRoute);
el("judge-battle-refresh-models").addEventListener(
  "click",
  refreshModelCatalog,
);
el("judge-battle-scope").addEventListener("change", renderScopeFields);
el("judge-battle-create").addEventListener("click", createJob);
el("judge-battle-refresh").addEventListener("click", () => loadJobs());
el("judge-battle-pause").addEventListener("click", () => controlJob("pause"));
el("judge-battle-resume").addEventListener("click", () => controlJob("resume"));
el("judge-battle-cancel").addEventListener("click", () => controlJob("cancel"));
el("judge-battle-fit-weights").addEventListener("click", fitWeights);
el("judge-battle-reveal-models").addEventListener("click", revealModels);
el("judge-battle-page-prev").addEventListener("click", async () => {
  state.pairOffset = Math.max(0, state.pairOffset - PAGE_SIZE);
  await loadSelectedJob();
});
el("judge-battle-page-next").addEventListener("click", async () => {
  state.pairOffset += PAGE_SIZE;
  await loadSelectedJob();
});
el("judge-battle-request-close").addEventListener("click", () =>
  el("judge-battle-request-dialog").close(),
);

init();
