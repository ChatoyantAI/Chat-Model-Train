"use strict";

const state = {
  config: null,
  comparisons: [],
  selectedModels: new Set(),
  selectedComparisonId: null,
  comparison: null,
  sampleDetail: null,
  selectedScoreDimension: "all",
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

function scopeDescription(scope) {
  if (scope.type === "current") return `样本 ${scope.seq}`;
  if (scope.type === "range") return `样本 ${scope.start_seq}–${scope.end_seq}`;
  if (scope.type === "random") return `随机 ${scope.count} 条`;
  return scope.type;
}

function formatRate(value) {
  return value === null || value === undefined
    ? "—"
    : `${(value * 100).toFixed(1)}%`;
}

function finiteNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function firstNumber(...values) {
  for (const value of values) {
    const number = finiteNumber(value);
    if (number !== null) return number;
  }
  return null;
}

function ratio(numerator, denominator) {
  return numerator !== null && denominator ? numerator / denominator : null;
}

function fraction(numerator, denominator, unit = "") {
  if (numerator === null || denominator === null) return "暂无计数";
  return `${numerator} / ${denominator}${unit}`;
}

function formatDistance(value) {
  const number = finiteNumber(value);
  return number === null ? "—" : number.toFixed(3);
}

function metricSummary(row = {}, legacyPairwise = null) {
  const coverage =
    row.coverage && typeof row.coverage === "object" ? row.coverage : {};
  const pairwise =
    row.pairwise && typeof row.pairwise === "object" ? row.pairwise : {};
  const confidence =
    row.confidence && typeof row.confidence === "object" ? row.confidence : {};
  const logprobConfidence =
    confidence.logprob && typeof confidence.logprob === "object"
      ? confidence.logprob
      : {};
  const logprobDistance =
    logprobConfidence.distance && typeof logprobConfidence.distance === "object"
      ? logprobConfidence.distance
      : {};
  const legacyComparablePairs = legacyPairwise
    ? legacyPairwise.reduce(
        (sum, item) => sum + (finiteNumber(item.comparable) || 0),
        0,
      )
    : null;
  const legacyExactPairs = legacyPairwise
    ? legacyPairwise.reduce(
        (sum, item) => sum + (finiteNumber(item.agreed) || 0),
        0,
      )
    : null;
  const expectedRatings = firstNumber(
    coverage.possible_ratings,
    row.expected_ratings,
    row.model_sample_slots,
  );
  const observedRatings = firstNumber(
    coverage.available_ratings,
    row.observed_ratings,
    row.participating_model_samples,
  );
  // Never treat legacy available-case agreement as strict all-model agreement.
  const strictComparable = firstNumber(
    row.full_coverage_comparable,
    row.strict_comparable,
  );
  const strictAgreed = firstNumber(row.full_coverage_agreed, row.strict_agreed);
  const allWithinOne = firstNumber(row.full_coverage_within_one);
  const comparablePairs = firstNumber(
    pairwise.comparable,
    row.comparable_pairs,
    legacyComparablePairs,
  );
  const exactPairs = firstNumber(
    pairwise.exact,
    row.exact_pairs,
    row.exact,
    legacyExactPairs,
  );
  const adjacentPairs = firstNumber(
    pairwise.adjacent,
    row.within_one_pairs,
    row.adjacent,
  );
  const flatAdjacentOnlyPairs = firstNumber(row.adjacent_only_pairs);
  const adjacentOnlyPairs =
    flatAdjacentOnlyPairs !== null
      ? flatAdjacentOnlyPairs
      : adjacentPairs !== null && exactPairs !== null
        ? Math.max(0, adjacentPairs - exactPairs)
        : null;
  const numericFarPairs = firstNumber(
    pairwise.numeric_far,
    row.numeric_far_pairs,
    row.numeric_far,
  );
  const naMismatchPairs = firstNumber(
    pairwise.na_mismatch,
    row.na_mismatch_pairs,
    row.na_mismatch,
  );
  const severePairs = firstNumber(
    pairwise.severe,
    row.severe_pairs,
    row.severe,
    numericFarPairs !== null || naMismatchPairs !== null
      ? (numericFarPairs || 0) + (naMismatchPairs || 0)
      : null,
  );
  const withinOnePairs =
    adjacentPairs !== null
      ? adjacentPairs
      : comparablePairs !== null &&
          exactPairs !== null &&
          adjacentOnlyPairs !== null
        ? exactPairs + adjacentOnlyPairs
        : null;
  return {
    coverageRate: firstNumber(
      coverage.rate,
      row.coverage_rate,
      ratio(observedRatings, expectedRatings),
    ),
    observedRatings,
    expectedRatings,
    strictRate: firstNumber(
      row.full_coverage_agreement_rate,
      row.strict_all_exact_rate,
      ratio(strictAgreed, strictComparable),
    ),
    strictAgreed,
    strictComparable,
    allWithinOne,
    allWithinOneRate: firstNumber(
      row.full_coverage_within_one_rate,
      ratio(allWithinOne, strictComparable),
    ),
    exactRate: firstNumber(
      pairwise.exact_rate,
      row.pairwise_exact_rate,
      row.exact_rate,
      ratio(exactPairs, comparablePairs),
    ),
    exactPairs,
    comparablePairs,
    adjacentRate: firstNumber(
      pairwise.adjacent_rate,
      row.pairwise_adjacent_rate,
      row.adjacent_rate,
      ratio(withinOnePairs, comparablePairs),
    ),
    adjacentOnlyPairs,
    withinOnePairs,
    severeRate: firstNumber(
      pairwise.severe_rate,
      row.pairwise_severe_rate,
      row.severe_rate,
      ratio(severePairs, comparablePairs),
    ),
    severePairs,
    numericFarPairs,
    naMismatchPairs,
    logprobRecordedRatings: firstNumber(logprobConfidence.recorded_ratings),
    logprobCoverageRate: firstNumber(logprobConfidence.coverage_rate),
    logprobDistanceComparable: firstNumber(logprobDistance.comparable),
    logprobRawDistanceMean: firstNumber(logprobDistance.raw_mean),
    logprobAdjustedDistanceMean: firstNumber(logprobDistance.adjusted_mean),
    logprobDistanceReduction: firstNumber(logprobDistance.reduction),
  };
}

function rateWithCount(rate, numerator, denominator) {
  const count =
    numerator === null || denominator === null
      ? ""
      : ` (${numerator}/${denominator})`;
  return `${formatRate(rate)}${count}`;
}

function renderProviderRoute() {
  const selected = state.config.models.filter((entry) =>
    state.selectedModels.has(entry.id),
  );
  const tatu = selected.filter((entry) => entry.gateway === "tatu");
  const openrouter = selected.filter((entry) => entry.gateway !== "tatu");
  const routing = state.config.provider_routing || {};
  const orderedCount = openrouter.filter(
    (entry) => (routing.preferred_order_by_model || {})[entry.id]?.length,
  ).length;
  const noTokenProbabilityCount = openrouter.filter(
    (entry) => entry.reports_token_logprobs !== true,
  ).length;
  const ignored = routing.ignore || [];
  const openrouterText = `${orderedCount}/${openrouter.length} 个模型使用历史成功率优先的独立渠道顺序；${routing.allow_fallbacks ? "不可用时继续 fallback" : "不兜底"}${ignored.length ? `；全局排除 ${ignored.join("、")}` : ""}。`;
  const parts = [];
  if (openrouter.length) {
    const tokenText = noTokenProbabilityCount
      ? `；其中 ${noTokenProbabilityCount} 个当前渠道不提供可信的评分 Token 概率`
      : "";
    parts.push(
      `OpenRouter ${openrouter.length} 个：${openrouterText}${tokenText}`,
    );
  }
  if (tatu.length)
    parts.push(
      `TatuCloud ${tatu.length} 个：按各模型精确 deployer 路由，Token 概率与美元费用不回传；长输出建议先单条试跑`,
    );
  el("compare-provider-route").textContent =
    `渠道：${parts.length ? parts.join(" · ") : "请先选择模型"}`;
}

function renderBudgetTotal() {
  const budget = Number(el("compare-budget").value) || 0;
  const count = state.selectedModels.size;
  el("compare-budget-total").textContent =
    `总预算理论上限：${count} × $${budget.toFixed(2)} = $${(count * budget).toFixed(2)}`;
}

function renderModelPicker() {
  const root = el("compare-model-picker");
  const query = el("compare-model-search").value.trim().toLocaleLowerCase();
  const maxModels = state.config.comparison_limits.max_models;
  const selectedEntries = state.config.models.filter((entry) =>
    state.selectedModels.has(entry.id),
  );
  const matched = state.config.models.filter((entry) => {
    const text =
      `${entry.id} ${entry.name || ""} ${entry.label || ""}`.toLocaleLowerCase();
    return !query || text.includes(query);
  });
  const visible = [];
  const seen = new Set();
  [...selectedEntries, ...matched].forEach((entry) => {
    if (!seen.has(entry.id) && visible.length < 120) {
      visible.push(entry);
      seen.add(entry.id);
    }
  });
  root.replaceChildren();
  visible.forEach((entry) => {
    const option = document.createElement("label");
    option.className = "compare-model-option";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = state.selectedModels.has(entry.id);
    input.disabled = !input.checked && state.selectedModels.size >= maxModels;
    input.addEventListener("change", () => {
      if (input.checked) state.selectedModels.add(entry.id);
      else state.selectedModels.delete(entry.id);
      renderModelPicker();
    });
    const copy = document.createElement("span");
    copy.className = "compare-model-copy";
    const id = document.createElement("strong");
    id.textContent = entry.id;
    const meta = document.createElement("span");
    meta.textContent = entry.label || entry.name || entry.id;
    copy.append(id, meta);
    option.append(input, copy);
    root.append(option);
  });
  if (!visible.length) {
    const empty = document.createElement("p");
    empty.className = "llm-help";
    empty.textContent = "没有匹配模型";
    root.append(empty);
  }
  el("compare-selected-count").textContent =
    `${state.selectedModels.size} / ${maxModels} 个`;
  renderBudgetTotal();
  renderProviderRoute();
}

async function refreshModels() {
  const button = el("compare-refresh-models");
  button.disabled = true;
  button.textContent = "刷新中…";
  try {
    const payload = await api("/api/llm-judge/models/refresh", {
      method: "POST",
    });
    state.config.models = payload.models;
    const { models: _, ...catalog } = payload;
    state.config.model_catalog = catalog;
    for (const model of [...state.selectedModels]) {
      if (!payload.models.some((entry) => entry.id === model))
        state.selectedModels.delete(model);
    }
    renderModelCatalogLabel();
    renderModelPicker();
    toast(
      payload.stale
        ? "刷新失败，继续使用备用目录"
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

function renderModelCatalogLabel() {
  const catalog = state.config.model_catalog;
  const source = catalog.source?.includes("tatu")
    ? `混合目录（${catalog.source}）`
    : catalog.source === "openrouter"
      ? "OpenRouter 实时目录"
      : "服务器备用目录";
  el("compare-model-catalog").textContent =
    `${source} · ${state.config.models.length} 个兼容模型${catalog.stale ? " · 缓存/备用" : ""}`;
}

function renderScopeFields() {
  const type = el("compare-scope").value;
  ["current", "range", "random"].forEach((name) => {
    el(`compare-scope-${name}`).classList.toggle("hidden", name !== type);
  });
}

function scopePayload() {
  const type = el("compare-scope").value;
  const payload = { scope_type: type };
  if (type === "current") payload.seq = Number(el("compare-seq").value);
  if (type === "range") {
    payload.start_seq = Number(el("compare-start").value);
    payload.end_seq = Number(el("compare-end").value);
  }
  if (type === "random")
    payload.random_count = Number(el("compare-random-count").value);
  return payload;
}

async function createComparison() {
  const limits = state.config.comparison_limits;
  if (state.selectedModels.size < limits.min_models) {
    toast(`至少选择 ${limits.min_models} 个模型`, true);
    return;
  }
  const budget = Number(el("compare-budget").value);
  const payload = {
    models: [...state.selectedModels],
    budget_per_model_usd: budget,
    ...scopePayload(),
  };
  const totalBudget = budget * payload.models.length;
  if (
    !window.confirm(
      `确认用 ${payload.models.length} 个模型评估同一批样本？每模型预算 $${budget.toFixed(2)}，总上限 $${totalBudget.toFixed(2)}。`,
    )
  )
    return;
  el("compare-create").disabled = true;
  try {
    const comparison = await api("/api/llm-judge/comparisons", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.selectedComparisonId = comparison.comparison_id;
    toast("多模型比较已在后台创建");
    await loadComparisons();
  } catch (error) {
    toast(error.message, true);
  } finally {
    el("compare-create").disabled = !state.config.enabled;
  }
}

function renderHistory() {
  const root = el("compare-history");
  root.replaceChildren();
  if (!state.comparisons.length) {
    const empty = document.createElement("p");
    empty.className = "llm-help";
    empty.textContent = "还没有多模型比较任务。";
    root.append(empty);
    return;
  }
  state.comparisons.forEach((comparison) => {
    const button = document.createElement("button");
    button.className = "llm-job";
    button.classList.toggle(
      "selected",
      comparison.comparison_id === state.selectedComparisonId,
    );
    const title = document.createElement("strong");
    title.textContent = `${scopeDescription(comparison.scope)} · ${statusLabels[comparison.status] || comparison.status}`;
    const models = document.createElement("span");
    models.textContent = comparison.models.join(" vs ");
    const cost = document.createElement("span");
    const rubric = comparison.rubric_version
      ? ` · ${comparison.rubric_version}`
      : "";
    cost.textContent = `${comparison.models.length} 模型 · ${comparison.total_count} 样本 · $${comparison.actual_cost_usd.toFixed(4)}${rubric}`;
    button.append(title, models, cost);
    button.addEventListener("click", () =>
      selectComparison(comparison.comparison_id),
    );
    root.append(button);
  });
}

async function loadComparisons(selectNewest = false) {
  const payload = await api("/api/llm-judge/comparisons");
  state.comparisons = payload.comparisons;
  if (selectNewest && !state.selectedComparisonId && state.comparisons.length) {
    state.selectedComparisonId = state.comparisons[0].comparison_id;
  }
  renderHistory();
  if (state.selectedComparisonId) await loadSelectedComparison();
}

async function selectComparison(comparisonId) {
  state.selectedComparisonId = comparisonId;
  state.sampleDetail = null;
  el("compare-sample-detail").classList.add("hidden");
  renderHistory();
  await loadSelectedComparison();
}

function renderModelJobs(jobs) {
  const root = el("compare-model-jobs");
  root.replaceChildren();
  jobs.forEach((job) => {
    const card = document.createElement("article");
    card.className = "compare-model-job";
    const model = document.createElement("strong");
    model.textContent = job.model;
    const progress = document.createElement("span");
    progress.textContent = `${statusLabels[job.status] || job.status} · ${job.finished_count}/${job.total_count} · $${job.actual_cost_usd.toFixed(4)}`;
    card.append(model, progress);
    root.append(card);
  });
}

function kpiCard(label, rate, detail) {
  const card = document.createElement("article");
  card.className = "compare-kpi";
  const title = document.createElement("span");
  title.textContent = label;
  const value = document.createElement("strong");
  value.textContent = formatRate(rate);
  const small = document.createElement("small");
  small.textContent = detail;
  card.append(title, value, small);
  return card;
}

function kpiValueCard(label, valueText, detail) {
  const card = document.createElement("article");
  card.className = "compare-kpi";
  const title = document.createElement("span");
  title.textContent = label;
  const value = document.createElement("strong");
  value.textContent = valueText;
  const small = document.createElement("small");
  small.textContent = detail;
  card.append(title, value, small);
  return card;
}

function renderSimpleTable(root, headers, rows) {
  root.replaceChildren();
  const table = document.createElement("table");
  table.className = "llm-table";
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  headers.forEach((header) => {
    const th = document.createElement("th");
    th.textContent = header;
    headerRow.append(th);
  });
  thead.append(headerRow);
  const tbody = document.createElement("tbody");
  rows.forEach((values) => {
    const row = document.createElement("tr");
    values.forEach((value) => {
      const td = document.createElement("td");
      td.textContent = String(value);
      row.append(td);
    });
    tbody.append(row);
  });
  table.append(thead, tbody);
  root.append(table);
}

function renderMetrics(metrics) {
  const atomic = metrics.atomic || {};
  const atomicPairs = metrics.atomic_pairwise || metrics.pairwise || [];
  const atomicSummary = metricSummary(atomic, atomicPairs);
  const severeDetail =
    atomicSummary.severePairs === null
      ? "暂无计数"
      : `${atomicSummary.severePairs} / ${atomicSummary.comparablePairs} 对 · 数字相差≥2 ${atomicSummary.numericFarPairs ?? "—"} · N/A↔数字 ${atomicSummary.naMismatchPairs ?? "—"}`;
  el("compare-atomic-kpis").replaceChildren(
    kpiCard(
      "评分覆盖率",
      atomicSummary.coverageRate,
      fraction(
        atomicSummary.observedRatings,
        atomicSummary.expectedRatings,
        " 个评分观测",
      ),
    ),
    kpiCard(
      "全体同档（严格）",
      atomicSummary.strictRate,
      fraction(
        atomicSummary.strictAgreed,
        atomicSummary.strictComparable,
        " 个完整覆盖单元",
      ),
    ),
    kpiCard(
      "全体最大档差≤1（严格）",
      atomicSummary.allWithinOneRate,
      fraction(
        atomicSummary.allWithinOne,
        atomicSummary.strictComparable,
        " 个完整覆盖单元",
      ),
    ),
    kpiCard(
      "两两同档",
      atomicSummary.exactRate,
      fraction(atomicSummary.exactPairs, atomicSummary.comparablePairs, " 对"),
    ),
    kpiCard(
      "两两相差≤1（含同档）",
      atomicSummary.adjacentRate,
      atomicSummary.withinOnePairs === null
        ? "暂无计数"
        : `${atomicSummary.withinOnePairs} / ${atomicSummary.comparablePairs} 对 · 仅相差一档 ${atomicSummary.adjacentOnlyPairs}`,
    ),
    kpiCard("严重分歧", atomicSummary.severeRate, severeDetail),
    kpiCard(
      "Token概率覆盖率",
      atomicSummary.logprobCoverageRate,
      fraction(
        atomicSummary.logprobRecordedRatings,
        atomicSummary.observedRatings,
        " 个评分",
      ),
    ),
    kpiValueCard(
      "Token可用子集原始距离",
      formatDistance(atomicSummary.logprobRawDistanceMean),
      `${atomicSummary.logprobDistanceComparable ?? "—"} 对 · 越低越一致`,
    ),
    kpiValueCard(
      "Token概率修正距离",
      formatDistance(atomicSummary.logprobAdjustedDistanceMean),
      `比同子集原始距离减少 ${formatDistance(atomicSummary.logprobDistanceReduction)}`,
    ),
  );

  const dimensionRows = metrics.dimensions || [];
  renderSimpleTable(
    el("compare-dimensions"),
    [
      "维度",
      "覆盖率",
      "全体同档（严格）",
      "全体最大差≤1（严格）",
      "两两同档",
      "相差≤1（含同档）",
      "严重分歧",
      "Token概率覆盖",
      "Token子集原始距离",
      "Token概率修正距离",
      "Token减少",
    ],
    dimensionRows.map((row) => {
      const summary = metricSummary(row);
      return [
        row.name || row.id,
        rateWithCount(
          summary.coverageRate,
          summary.observedRatings,
          summary.expectedRatings,
        ),
        rateWithCount(
          summary.strictRate,
          summary.strictAgreed,
          summary.strictComparable,
        ),
        rateWithCount(
          summary.allWithinOneRate,
          summary.allWithinOne,
          summary.strictComparable,
        ),
        rateWithCount(
          summary.exactRate,
          summary.exactPairs,
          summary.comparablePairs,
        ),
        rateWithCount(
          summary.adjacentRate,
          summary.withinOnePairs,
          summary.comparablePairs,
        ),
        `${rateWithCount(summary.severeRate, summary.severePairs, summary.comparablePairs)} · 数字相差≥2 ${summary.numericFarPairs ?? "—"} / N/A冲突 ${summary.naMismatchPairs ?? "—"}`,
        rateWithCount(
          summary.logprobCoverageRate,
          summary.logprobRecordedRatings,
          summary.observedRatings,
        ),
        `${formatDistance(summary.logprobRawDistanceMean)} (${summary.logprobDistanceComparable ?? "—"}对)`,
        `${formatDistance(summary.logprobAdjustedDistanceMean)} (${summary.logprobDistanceComparable ?? "—"}对)`,
        formatDistance(summary.logprobDistanceReduction),
      ];
    }),
  );

  const atomicItems = metrics.atomic_items || [];
  el("compare-atomic-items-heading").textContent =
    `按原子项（${atomicItems.length || "—"} 项）`;
  renderSimpleTable(
    el("compare-atomic-items"),
    [
      "原子项",
      "判定",
      "覆盖率",
      "全体同档（严格）",
      "全体最大差≤1（严格）",
      "两两同档",
      "相差≤1（含同档）",
      "严重分歧",
      "Token概率覆盖",
      "Token子集原始距离",
      "Token概率修正距离",
      "Token减少",
    ],
    atomicItems.map((row) => {
      const summary = metricSummary(row);
      return [
        row.id,
        row.name || "—",
        rateWithCount(
          summary.coverageRate,
          summary.observedRatings,
          summary.expectedRatings,
        ),
        rateWithCount(
          summary.strictRate,
          summary.strictAgreed,
          summary.strictComparable,
        ),
        rateWithCount(
          summary.allWithinOneRate,
          summary.allWithinOne,
          summary.strictComparable,
        ),
        rateWithCount(
          summary.exactRate,
          summary.exactPairs,
          summary.comparablePairs,
        ),
        rateWithCount(
          summary.adjacentRate,
          summary.withinOnePairs,
          summary.comparablePairs,
        ),
        `${rateWithCount(summary.severeRate, summary.severePairs, summary.comparablePairs)} · 数字相差≥2 ${summary.numericFarPairs ?? "—"} / N/A冲突 ${summary.naMismatchPairs ?? "—"}`,
        rateWithCount(
          summary.logprobCoverageRate,
          summary.logprobRecordedRatings,
          summary.observedRatings,
        ),
        `${formatDistance(summary.logprobRawDistanceMean)} (${summary.logprobDistanceComparable ?? "—"}对)`,
        `${formatDistance(summary.logprobAdjustedDistanceMean)} (${summary.logprobDistanceComparable ?? "—"}对)`,
        formatDistance(summary.logprobDistanceReduction),
      ];
    }),
  );

  const atomicPairHasDistance = atomicPairs.some(
    (row) =>
      row.pairwise_exact_rate !== undefined ||
      row.exact_pairs !== undefined ||
      row.exact !== undefined ||
      row.adjacent_only_pairs !== undefined ||
      row.adjacent !== undefined ||
      row.severe_pairs !== undefined ||
      row.severe !== undefined,
  );
  renderSimpleTable(
    el("compare-atomic-pairwise"),
    atomicPairHasDistance
      ? [
          "模型 A",
          "模型 B",
          "两两同档",
          "相差≤1（含同档）",
          "严重分歧",
          "可比较",
        ]
      : ["模型 A", "模型 B", "同档率", "同档", "分歧", "可比较"],
    atomicPairs.map((row) => {
      if (!atomicPairHasDistance) {
        return [
          row.left,
          row.right,
          formatRate(row.agreement_rate),
          row.agreed,
          row.disagreed,
          row.comparable,
        ];
      }
      const summary = metricSummary(row, [row]);
      return [
        row.left,
        row.right,
        rateWithCount(
          summary.exactRate,
          summary.exactPairs,
          summary.comparablePairs,
        ),
        rateWithCount(
          summary.adjacentRate,
          summary.withinOnePairs,
          summary.comparablePairs,
        ),
        `${rateWithCount(summary.severeRate, summary.severePairs, summary.comparablePairs)} · 数字相差≥2 ${summary.numericFarPairs ?? "—"} / N/A冲突 ${summary.naMismatchPairs ?? "—"}`,
        summary.comparablePairs ?? row.comparable ?? "—",
      ];
    }),
  );

  const gateSummary = metricSummary(metrics.gates || {});
  const gatePairs = metrics.gate_pairwise || [];
  const gatePairComparable = gatePairs.reduce(
    (sum, row) => sum + row.comparable,
    0,
  );
  const gatePairAgreed = gatePairs.reduce(
    (sum, row) => sum + (row.exact ?? row.agreed ?? 0),
    0,
  );
  el("compare-gate-kpis").replaceChildren(
    kpiCard(
      "Gate 覆盖率",
      gateSummary.coverageRate,
      fraction(
        gateSummary.observedRatings,
        gateSummary.expectedRatings,
        " 个判定观测",
      ),
    ),
    kpiCard(
      "全体同判（严格）",
      gateSummary.strictRate,
      fraction(
        gateSummary.strictAgreed,
        gateSummary.strictComparable,
        " 个完整覆盖单元",
      ),
    ),
    kpiCard(
      "两两同判",
      gateSummary.exactRate ??
        (gatePairComparable ? gatePairAgreed / gatePairComparable : null),
      `${gatePairAgreed} 同判 / ${gatePairComparable - gatePairAgreed} 分歧`,
    ),
  );
  renderSimpleTable(
    el("compare-gate-pairwise"),
    ["模型 A", "模型 B", "一致率", "一致", "分歧", "可比较"],
    gatePairs.map((row) => [
      row.left,
      row.right,
      formatRate(row.agreement_rate),
      row.agreed,
      row.disagreed,
      row.comparable,
    ]),
  );
  renderSimpleTable(
    el("compare-gate-items"),
    ["Gate", "判定", "覆盖率", "全体同判（严格）", "两两同判"],
    (metrics.gate_items || []).map((row) => {
      const summary = metricSummary(row);
      return [
        row.id,
        row.name,
        rateWithCount(
          summary.coverageRate,
          summary.observedRatings,
          summary.expectedRatings,
        ),
        rateWithCount(
          summary.strictRate,
          summary.strictAgreed,
          summary.strictComparable,
        ),
        rateWithCount(
          summary.exactRate,
          summary.exactPairs,
          summary.comparablePairs,
        ),
      ];
    }),
  );
}

function renderSamples(samples) {
  const body = el("compare-samples");
  body.replaceChildren();
  samples.forEach((sample) => {
    const row = document.createElement("tr");
    row.classList.toggle(
      "compare-samples-disagreement",
      sample.atomic_disagreed > 0 || sample.gate_disagreed > 0,
    );
    const values = [
      sample.position,
      sample.seq,
      `${sample.atomic_participant_count}/${sample.model_count}`,
      `${sample.gate_participant_count}/${sample.model_count}`,
      sample.atomic_comparable,
      sample.atomic_disagreed,
      sample.gate_comparable,
      sample.gate_disagreed,
    ];
    values.forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = String(value);
      row.append(cell);
    });
    const action = document.createElement("td");
    const button = document.createElement("button");
    button.className = "ghost small table-action";
    button.textContent = "逐项对比";
    button.addEventListener("click", () => loadComparisonSample(sample.seq));
    action.append(button);
    row.append(action);
    row.addEventListener("dblclick", () => loadComparisonSample(sample.seq));
    body.append(row);
  });
}

function renderComparison(comparison) {
  state.comparison = comparison;
  el("compare-empty").classList.add("hidden");
  el("compare-detail").classList.remove("hidden");
  el("compare-title").textContent =
    `${comparison.models.length} 个模型 × ${comparison.total_count} 条样本`;
  const rubricVersion =
    comparison.rubric_version || state.config.rubric_version || "—";
  const rubricHash =
    comparison.rubric_sha256 || state.config.rubric_sha256 || "—";
  el("compare-rubric-version").textContent = `规则 ${rubricVersion}`;
  el("compare-rubric-hash").textContent = rubricHash;
  el("compare-status").textContent =
    statusLabels[comparison.status] || comparison.status;
  el("compare-status").classList.toggle(
    "error",
    ["completed_with_errors", "budget_exhausted"].includes(comparison.status),
  );
  el("compare-meta").replaceChildren();
  [
    `范围：${scopeDescription(comparison.scope)}`,
    `费用：$${comparison.actual_cost_usd.toFixed(4)} / $${comparison.total_budget_limit_usd.toFixed(2)}`,
    `每模型预算：$${comparison.budget_per_model_usd.toFixed(2)}`,
    `规则：${rubricVersion}`,
    `规则哈希：${rubricHash}`,
  ].forEach((text) => {
    const span = document.createElement("span");
    span.textContent = text;
    el("compare-meta").append(span);
  });
  const rubricMetadata = comparison.metrics?.rubric_metadata;
  if (rubricMetadata && rubricMetadata.status !== "current") {
    const warning = document.createElement("span");
    warning.className = "compare-rubric-warning";
    warning.textContent =
      "历史规则快照不可用：下表只汇总任务已保存的评分值，项目名称与尺度应以该任务的规则版本为准。";
    el("compare-meta").append(warning);
  }
  const finished = comparison.jobs.reduce(
    (sum, job) => sum + job.finished_count,
    0,
  );
  const total = comparison.total_count * comparison.models.length;
  el("compare-progress-fill").style.width =
    `${total ? (finished / total) * 100 : 0}%`;
  el("compare-progress-text").textContent =
    `模型-样本完成 ${finished} / ${total}`;
  el("compare-pause").disabled = !comparison.jobs.some((job) =>
    ["queued", "running"].includes(job.status),
  );
  el("compare-resume").disabled = !comparison.jobs.some(
    (job) => job.status === "paused",
  );
  el("compare-cancel").disabled = !comparison.jobs.some((job) =>
    ["queued", "running", "paused"].includes(job.status),
  );
  renderModelJobs(comparison.jobs);
  renderMetrics(comparison.metrics);
  renderSamples(comparison.metrics.samples);
}

async function loadSelectedComparison() {
  if (!state.selectedComparisonId) return;
  try {
    const comparison = await api(
      `/api/llm-judge/comparisons/${state.selectedComparisonId}`,
    );
    const index = state.comparisons.findIndex(
      (item) => item.comparison_id === comparison.comparison_id,
    );
    if (index >= 0)
      state.comparisons[index] = { ...state.comparisons[index], ...comparison };
    renderHistory();
    renderComparison(comparison);
    if (state.sampleDetail)
      await loadComparisonSample(state.sampleDetail.seq, false);
  } catch (error) {
    toast(error.message, true);
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

function renderContext(sample) {
  const root = el("compare-context");
  root.replaceChildren();
  (sample.messages || []).forEach((message, index) => {
    appendContextMessage(
      root,
      message.display_role || message.role,
      message.content,
      index,
    );
  });
  if (sample.candidate)
    appendContextMessage(root, "assistant_candidate", sample.candidate.content);
}

function renderSampleModels(results) {
  const root = el("compare-sample-models");
  root.replaceChildren();
  results.forEach((result) => {
    const card = document.createElement("article");
    card.className = "compare-model-job";
    const model = document.createElement("strong");
    model.textContent = result.model;
    const status = document.createElement("span");
    const provider = result.provider ? ` · ${result.provider}` : "";
    status.textContent = `${statusLabels[result.status] || result.status}${provider} · $${result.usage.cost_usd.toFixed(5)} · ${result.usage.latency_ms}ms`;
    card.append(model, status);
    if (result.error_message) {
      const error = document.createElement("span");
      error.textContent = result.error_message;
      error.className = "status-failed";
      card.append(error);
    }
    if (!result.scores) {
      const excluded = document.createElement("span");
      excluded.className = "comparison-exclusion-note";
      excluded.textContent = "无原子评分：不参与本样本的原子评分比较";
      card.append(excluded);
    }
    if (!result.gates) {
      const excluded = document.createElement("span");
      excluded.className = "comparison-exclusion-note";
      excluded.textContent = "无 Gate 结果：不参与本样本的 Gate 比较";
      card.append(excluded);
    }
    root.append(card);
  });
}

function criterionCard(row, models) {
  const card = document.createElement("article");
  const cardState = row.agreement
    ? "agreement"
    : row.has_difference
      ? "difference"
      : "insufficient";
  card.className = `criterion-card ${cardState}`;
  const header = document.createElement("header");
  header.className = "criterion-header";
  const title = document.createElement("strong");
  title.textContent = `${row.id} · ${row.name}`;
  const tag = document.createElement("span");
  tag.className = "criterion-tag";
  if (row.agreement) {
    tag.textContent = row.excluded_model_count
      ? `有效结果一致 · ${row.participant_count} 模型参与`
      : "全部参与模型一致";
  } else if (row.has_difference) {
    tag.textContent = `存在分歧 · ${row.participant_count} 模型参与`;
  } else {
    tag.textContent = `参与不足 · ${row.participant_count} 模型`;
  }
  header.append(title, tag);
  const grid = document.createElement("div");
  grid.className = "criterion-model-grid";
  grid.style.setProperty("--model-count", String(models.length));
  models.forEach((model) => {
    const cell = document.createElement("section");
    cell.className = "criterion-cell";
    const name = document.createElement("div");
    name.className = "criterion-model";
    name.textContent = model;
    const record = row.models[model];
    const value = document.createElement("span");
    const hasValue = Boolean(
      record && record.value !== null && record.value !== undefined,
    );
    value.className = `criterion-value${hasValue ? "" : " missing"}`;
    value.textContent = hasValue ? String(record.value) : "无结果（不参与）";
    cell.append(name, value);
    if (record) {
      if (row.dimension !== "Gate") {
        const logprobConfidence = finiteNumber(record.logprob_confidence);
        const tokenConfidence = document.createElement("span");
        tokenConfidence.className = `criterion-confidence ${logprobConfidence === null ? "confidence-missing" : "confidence-token"}`;
        tokenConfidence.textContent =
          logprobConfidence === null
            ? "Token概率：不可用"
            : `Token概率：${(logprobConfidence * 100).toFixed(1)}%`;
        cell.append(tokenConfidence);
      }
      const reason = document.createElement("p");
      reason.className = "criterion-reason";
      reason.textContent = record.reason || "未提供理由";
      cell.append(reason);
      (record.evidence || []).forEach((text) => {
        const evidence = document.createElement("blockquote");
        evidence.className = "criterion-evidence";
        evidence.textContent = text;
        cell.append(evidence);
      });
    }
    grid.append(cell);
  });
  card.append(header, grid);
  return card;
}

function renderCriteria(root, rows, models) {
  root.replaceChildren();
  const onlyDifferences = el("compare-only-differences").checked;
  const visible = rows.filter((row) => !onlyDifferences || row.has_difference);
  visible.forEach((row) => root.append(criterionCard(row, models)));
  if (!visible.length) {
    const empty = document.createElement("p");
    empty.className = "compare-no-rows";
    empty.textContent = "这一部分实际参与比较的模型结果均一致。";
    root.append(empty);
  }
}

function scoreRecordValue(record) {
  return record && record.value !== null && record.value !== undefined
    ? String(record.value)
    : null;
}

function scoresWithinOne(values) {
  if (!values.length || values.some((value) => value === null)) return false;
  if (new Set(values).size === 1) return true;
  if (values.includes("N/A")) return false;
  const numeric = values.map((value) => Number(value));
  return (
    numeric.every(Number.isFinite) &&
    Math.max(...numeric) - Math.min(...numeric) <= 1
  );
}

function normalizedScoreDistance(leftValue, rightValue, scale = []) {
  if (leftValue === rightValue) return 0;
  if (leftValue === "N/A" || rightValue === "N/A") return 1;
  const left = Number(leftValue);
  const right = Number(rightValue);
  const numericScale = scale.map(Number).filter(Number.isFinite);
  if (!Number.isFinite(left) || !Number.isFinite(right)) return 1;
  const span =
    numericScale.length >= 2
      ? Math.max(...numericScale) - Math.min(...numericScale)
      : 1;
  return Math.min(1, Math.abs(left - right) / Math.max(1, span));
}

function renderScoreDimension(payload) {
  const select = el("compare-score-dimension");
  const dimensions = new Map();
  payload.scores.forEach((row) => {
    if (!dimensions.has(row.dimension)) dimensions.set(row.dimension, []);
    dimensions.get(row.dimension).push(row);
  });
  if (
    state.selectedScoreDimension !== "all" &&
    !dimensions.has(state.selectedScoreDimension)
  ) {
    state.selectedScoreDimension = "all";
  }
  select.replaceChildren();
  const allOption = document.createElement("option");
  allOption.value = "all";
  allOption.textContent = `全部维度（${payload.scores.length} 项）`;
  select.append(allOption);
  dimensions.forEach((rows, name) => {
    const option = document.createElement("option");
    const prefixes = [
      ...new Set(
        rows
          .map((row) => String(row.id).match(/^[A-Za-z]+/)?.[0])
          .filter(Boolean),
      ),
    ];
    option.value = name;
    option.textContent = `${prefixes.length === 1 ? `${prefixes[0]} · ` : ""}${name}（${rows.length} 项）`;
    select.append(option);
  });
  select.value = state.selectedScoreDimension;

  const selectedRows =
    state.selectedScoreDimension === "all"
      ? payload.scores
      : dimensions.get(state.selectedScoreDimension) || [];
  const selectedLabel =
    state.selectedScoreDimension === "all"
      ? "全部维度"
      : state.selectedScoreDimension;

  let observedRatings = 0;
  let fullUnits = 0;
  let fullExact = 0;
  let fullWithinOne = 0;
  let rawDistancePairs = 0;
  let rawDistanceSum = 0;
  let logprobRatings = 0;
  let logprobDistancePairs = 0;
  let logprobRawDistanceSum = 0;
  let logprobAdjustedDistanceSum = 0;
  selectedRows.forEach((row) => {
    const values = payload.models.map((model) =>
      scoreRecordValue(row.models[model]),
    );
    observedRatings += values.filter((value) => value !== null).length;
    logprobRatings += payload.models.filter(
      (model) =>
        scoreRecordValue(row.models[model]) !== null &&
        finiteNumber(row.models[model]?.logprob_confidence) !== null,
    ).length;
    if (values.every((value) => value !== null)) {
      fullUnits += 1;
      if (new Set(values).size === 1) fullExact += 1;
      if (scoresWithinOne(values)) fullWithinOne += 1;
    }
    payload.models.forEach((left, leftIndex) => {
      const leftRecord = row.models[left];
      const leftValue = scoreRecordValue(leftRecord);
      payload.models.slice(leftIndex + 1).forEach((right) => {
        const rightRecord = row.models[right];
        const rightValue = scoreRecordValue(rightRecord);
        if (leftValue === null || rightValue === null) return;
        const rawDistance = normalizedScoreDistance(
          leftValue,
          rightValue,
          row.scale || [],
        );
        rawDistancePairs += 1;
        rawDistanceSum += rawDistance;
        const leftLogprob = finiteNumber(leftRecord?.logprob_confidence);
        const rightLogprob = finiteNumber(rightRecord?.logprob_confidence);
        if (leftLogprob === null || rightLogprob === null) return;
        logprobDistancePairs += 1;
        logprobRawDistanceSum += rawDistance;
        logprobAdjustedDistanceSum +=
          rawDistance * Math.sqrt(leftLogprob * rightLogprob);
      });
    });
  });
  const possibleRatings = selectedRows.length * payload.models.length;
  el("compare-selected-dimension-kpis").replaceChildren(
    kpiCard(
      "评分覆盖率",
      ratio(observedRatings, possibleRatings),
      fraction(observedRatings, possibleRatings, " 个评分观测"),
    ),
    kpiCard(
      "全体同档（严格）",
      ratio(fullExact, fullUnits),
      fraction(fullExact, fullUnits, " 个完整覆盖原子项"),
    ),
    kpiCard(
      "全体最大档差≤1（严格）",
      ratio(fullWithinOne, fullUnits),
      fraction(fullWithinOne, fullUnits, " 个完整覆盖原子项"),
    ),
    kpiValueCard(
      "原始平均距离",
      formatDistance(ratio(rawDistanceSum, rawDistancePairs)),
      `${rawDistancePairs} 对有效评分 · 越低越一致`,
    ),
    kpiCard(
      "Token概率覆盖率",
      ratio(logprobRatings, observedRatings),
      fraction(logprobRatings, observedRatings, " 个评分"),
    ),
    kpiValueCard(
      "Token概率修正距离",
      formatDistance(ratio(logprobAdjustedDistanceSum, logprobDistancePairs)),
      `${logprobDistancePairs} 对 · 同子集原始距离 ${formatDistance(ratio(logprobRawDistanceSum, logprobDistancePairs))}`,
    ),
  );

  renderSimpleTable(
    el("compare-score-matrix"),
    ["模型", ...selectedRows.map((row) => row.id)],
    payload.models.map((model) => [
      model,
      ...selectedRows.map((row) => {
        const record = row.models[model];
        const value = scoreRecordValue(record);
        return value === null
          ? "—"
          : `${value} · Token${finiteNumber(record?.logprob_confidence) === null ? "—" : `${(Number(record.logprob_confidence) * 100).toFixed(1)}%`}`;
      }),
    ]),
  );
  renderCriteria(el("compare-score-detail"), selectedRows, payload.models);
  el("compare-score-detail-title").textContent =
    `${selectedLabel} · ${selectedRows.length} 项原子评分`;
}

function renderSampleDetail(payload) {
  state.sampleDetail = payload;
  el("compare-sample-detail").classList.remove("hidden");
  el("compare-sample-title").textContent = `样本 ${payload.seq}`;
  renderContext(payload.sample);
  renderSampleModels(payload.model_results);
  renderCriteria(el("compare-gate-detail"), payload.gates, payload.models);
  renderScoreDimension(payload);
}

async function loadComparisonSample(seq, scroll = true) {
  try {
    const payload = await api(
      `/api/llm-judge/comparisons/${state.selectedComparisonId}/samples/${seq}`,
    );
    renderSampleDetail(payload);
    if (scroll)
      el("compare-sample-detail").scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
  } catch (error) {
    toast(error.message, true);
  }
}

async function controlComparison(action) {
  if (!state.selectedComparisonId) return;
  if (
    action === "cancel" &&
    !window.confirm(
      "确定停止所有模型尚未开始的样本吗？正在请求中的结果仍会保存。",
    )
  )
    return;
  try {
    const comparison = await api(
      `/api/llm-judge/comparisons/${state.selectedComparisonId}/${action}`,
      { method: "POST", body: "{}" },
    );
    renderComparison(comparison);
    await loadComparisons();
  } catch (error) {
    toast(error.message, true);
  }
}

async function init() {
  try {
    const session = await api("/api/me");
    el("compare-current-user").textContent = session.username;
    state.config = await api("/api/llm-judge/config");
    el("compare-view").classList.remove("hidden");
    el("compare-rubric-version").textContent =
      `规则 ${state.config.rubric_version}`;
    el("compare-rubric-hash").textContent = state.config.rubric_sha256;
    renderModelCatalogLabel();
    renderModelPicker();
    renderProviderRoute();
    [
      "compare-seq",
      "compare-start",
      "compare-end",
      "compare-random-count",
    ].forEach((id) => {
      el(id).max = String(state.config.sample_count);
    });
    el("compare-random-count").max = String(
      state.config.comparison_limits.max_samples,
    );
    el("compare-limits").textContent =
      `${state.config.comparison_limits.min_models}–${state.config.comparison_limits.max_models} 个模型 · 最多 ${state.config.comparison_limits.max_samples} 条样本 · 全局并发 ${state.config.max_concurrency}`;
    if (!state.config.enabled) {
      el("compare-disabled").textContent = state.config.disabled_reason;
      el("compare-disabled").classList.remove("hidden");
      el("compare-create").disabled = true;
    }
    await loadComparisons(true);
    state.pollTimer = window.setInterval(async () => {
      if (!state.selectedComparisonId || document.hidden) return;
      if (
        state.comparison &&
        !["queued", "running", "paused"].includes(state.comparison.status)
      )
        return;
      try {
        await loadComparisons();
      } catch (_) {
        /* next poll will retry */
      }
    }, 3000);
  } catch (error) {
    if (error.status === 401) {
      window.location.replace("/");
      return;
    }
    el("compare-view").classList.remove("hidden");
    toast(error.message, true);
  }
}

el("compare-model-search").addEventListener("input", renderModelPicker);
el("compare-refresh-models").addEventListener("click", refreshModels);
el("compare-scope").addEventListener("change", renderScopeFields);
el("compare-budget").addEventListener("input", renderBudgetTotal);
el("compare-create").addEventListener("click", createComparison);
el("compare-refresh").addEventListener("click", () => loadComparisons());
el("compare-pause").addEventListener("click", () => controlComparison("pause"));
el("compare-resume").addEventListener("click", () =>
  controlComparison("resume"),
);
el("compare-cancel").addEventListener("click", () =>
  controlComparison("cancel"),
);
el("compare-only-differences").addEventListener("change", () => {
  if (state.sampleDetail) renderSampleDetail(state.sampleDetail);
});
el("compare-score-dimension").addEventListener("change", (event) => {
  state.selectedScoreDimension = event.target.value;
  if (state.sampleDetail) renderScoreDimension(state.sampleDetail);
});

init();
