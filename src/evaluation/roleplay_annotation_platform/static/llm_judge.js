"use strict";

const state = { config: null, jobs: [], selectedJobId: null, pollTimer: null };
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
  const model = el("llm-model").value;
  const entry = (state.config.models || []).find(
    (candidate) => candidate.id === model,
  );
  if (entry?.gateway === "tatu") {
    el("llm-provider-route").textContent =
      `渠道：TatuCloud · 精确路由 ${entry.deployer || "已配置部署商"} · Chat Completions · Token 概率与美元费用不回传 · 长输出的 Schema 遵循度与模型有关，建议先单条试跑`;
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
  if (preferred.length) {
    parts.push(`该模型支持时优先 ${preferred.join(" → ")}`);
  } else {
    parts.push("由 OpenRouter 自动选择渠道");
  }
  if (ignored.length) parts.push(`排除 ${ignored.join("、")}`);
  parts.push(
    routing.allow_fallbacks ? "不可用时自动切换其他渠道" : "不允许其他渠道兜底",
  );
  if (entry?.reports_token_logprobs !== true) {
    parts.push("当前渠道不提供可信的评分 Token 概率");
  }
  el("llm-provider-route").textContent = `渠道：${parts.join(" · ")}`;
}

function renderModelCatalog() {
  const modelSelect = el("llm-model");
  const current = modelSelect.value;
  const query = el("llm-model-search").value.trim().toLocaleLowerCase();
  const models = state.config.models.filter((entry) => {
    const searchable =
      `${entry.id} ${entry.name || ""} ${entry.label || ""}`.toLocaleLowerCase();
    return !query || searchable.includes(query);
  });
  modelSelect.replaceChildren();
  if (!models.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = state.config.models.length
      ? "没有匹配模型"
      : "服务器尚未配置模型";
    modelSelect.append(option);
    modelSelect.disabled = true;
  } else {
    models.forEach((entry) => {
      const option = document.createElement("option");
      option.value = entry.id;
      option.textContent = entry.label;
      modelSelect.append(option);
    });
    modelSelect.disabled = false;
    if (models.some((entry) => entry.id === current))
      modelSelect.value = current;
  }
  renderProviderRoute();

  const catalog = state.config.model_catalog;
  const source = catalog.source?.includes("tatu")
    ? `混合目录（${catalog.source}）`
    : catalog.source === "openrouter"
      ? "OpenRouter 实时目录"
      : "服务器备用目录";
  const stale = catalog.stale ? " · 当前为缓存/备用数据" : "";
  el("llm-model-catalog").textContent =
    `${source} · ${state.config.models.length} 个兼容模型${stale}`;
}

async function refreshModelCatalog() {
  const button = el("llm-refresh-models");
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

function selectedScopePayload(forceCurrent = false) {
  const type = forceCurrent ? "current" : el("llm-scope").value;
  const payload = { scope_type: type };
  if (type === "current") payload.seq = Number(el("llm-seq").value);
  if (type === "range") {
    payload.start_seq = Number(el("llm-start").value);
    payload.end_seq = Number(el("llm-end").value);
  }
  if (type === "random")
    payload.random_count = Number(el("llm-random-count").value);
  return payload;
}

function scopeDescription(scope) {
  if (scope.type === "current") return `样本 ${scope.seq}`;
  if (scope.type === "range") return `样本 ${scope.start_seq}–${scope.end_seq}`;
  if (scope.type === "random") return `随机 ${scope.count} 条`;
  if (scope.type === "all") return "全部样本";
  return scope.type;
}

function renderScopeFields() {
  const type = el("llm-scope").value;
  ["current", "range", "random"].forEach((name) => {
    el(`scope-${name}`).classList.toggle("hidden", name !== type);
  });
}

function renderJobs() {
  const root = el("llm-jobs");
  root.replaceChildren();
  if (!state.jobs.length) {
    const empty = document.createElement("p");
    empty.className = "llm-help";
    empty.textContent = "还没有 LLM Judge 任务。";
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
    progress.textContent = `${job.finished_count}/${job.total_count} · $${job.actual_cost_usd.toFixed(4)}`;
    button.append(title, model, progress);
    button.addEventListener("click", () => selectJob(job.job_id));
    root.append(button);
  });
}

async function loadJobs(selectNewest = false) {
  const payload = await api("/api/llm-judge/jobs");
  state.jobs = payload.jobs;
  if (selectNewest && state.jobs.length)
    state.selectedJobId = state.jobs[0].job_id;
  renderJobs();
  if (state.selectedJobId) await loadSelectedJob();
}

function renderJob(job) {
  el("llm-empty").classList.add("hidden");
  el("llm-job-detail").classList.remove("hidden");
  el("job-title").textContent = scopeDescription(job.scope);
  el("job-status").textContent = statusLabels[job.status] || job.status;
  el("job-status").classList.toggle(
    "error",
    ["completed_with_errors", "budget_exhausted"].includes(job.status),
  );
  el("job-model").textContent = `模型：${job.model}`;
  el("job-scope").textContent = `范围：${scopeDescription(job.scope)}`;
  el("job-cost").textContent =
    `费用：$${job.actual_cost_usd.toFixed(4)} / $${job.budget_limit_usd.toFixed(2)}`;
  const percent = job.total_count
    ? (job.finished_count / job.total_count) * 100
    : 0;
  el("job-progress-fill").style.width = `${percent}%`;
  const counts = job.counts;
  el("job-progress-text").textContent =
    `完成 ${counts.completed} · Gate淘汰 ${counts.discarded} · 运行 ${counts.running} · 等待 ${counts.pending} · 结果无效 ${counts.invalid} · 失败 ${counts.failed} · 取消 ${counts.cancelled}`;
  el("job-pause").disabled = !["queued", "running"].includes(job.status);
  el("job-resume").disabled = job.status !== "paused";
  el("job-cancel").disabled = !["queued", "running", "paused"].includes(
    job.status,
  );
}

function renderItems(items, job) {
  const body = el("job-items");
  body.replaceChildren();
  items.forEach((item) => {
    const row = document.createElement("tr");
    if (item.run_id) row.classList.add("clickable");
    const values = [
      item.position,
      item.seq,
      statusLabels[item.status] || item.status,
      item.attempt_count,
    ];
    values.forEach((value, index) => {
      const cell = document.createElement("td");
      cell.textContent = String(value);
      if (index === 2) cell.className = `status-${item.status}`;
      row.append(cell);
    });
    const note = document.createElement("td");
    note.textContent =
      item.error_message || (item.run_id ? "点击查看评分" : "");
    row.append(note);
    const action = document.createElement("td");
    if (item.run_id) {
      const result = document.createElement("button");
      result.className = "ghost small table-action";
      result.textContent = "结果";
      result.addEventListener("click", (event) => {
        event.stopPropagation();
        loadJudgment(item.run_id);
      });
      action.append(result);
    }
    const preview = document.createElement("button");
    preview.className = "ghost small table-action";
    preview.textContent = "请求";
    preview.addEventListener("click", (event) => {
      event.stopPropagation();
      showRequestPreview(item.seq, job.model);
    });
    action.append(preview);
    row.append(action);
    if (item.run_id)
      row.addEventListener("click", () => loadJudgment(item.run_id));
    body.append(row);
  });
}

async function selectJob(jobId) {
  state.selectedJobId = jobId;
  renderJobs();
  el("judgment-detail").classList.add("hidden");
  await loadSelectedJob();
}

async function loadSelectedJob() {
  if (!state.selectedJobId) return;
  const [job, items] = await Promise.all([
    api(`/api/llm-judge/jobs/${state.selectedJobId}`),
    api(`/api/llm-judge/jobs/${state.selectedJobId}/items?limit=500`),
  ]);
  const index = state.jobs.findIndex((entry) => entry.job_id === job.job_id);
  if (index >= 0) state.jobs[index] = job;
  renderJobs();
  renderJob(job);
  renderItems(items.items, job);
}

function judgmentCard(id, record) {
  const card = document.createElement("article");
  card.className = "judgment-item";
  const header = document.createElement("header");
  const name = document.createElement("span");
  name.textContent = `${id}：`;
  const value = document.createElement("span");
  value.textContent =
    !id.startsWith("G-") && record.value !== "N/A"
      ? `${record.value} 分`
      : String(record.value);
  const outcome = document.createElement("span");
  outcome.className = "judgment-outcome";
  outcome.append(value);
  if (!id.startsWith("G-")) {
    const tokenProbability = Number(record.logprob_confidence);
    const hasTokenProbability =
      record.logprob_confidence !== null &&
      record.logprob_confidence !== undefined &&
      Number.isFinite(tokenProbability) &&
      tokenProbability >= 0 &&
      tokenProbability <= 1;
    const tokenConfidence = document.createElement("span");
    tokenConfidence.className = `judge-confidence ${hasTokenProbability ? "confidence-token" : "confidence-missing"}`;
    tokenConfidence.textContent = hasTokenProbability
      ? `Token概率：${(tokenProbability * 100).toFixed(1)}%`
      : "Token概率：不可用";
    outcome.append(tokenConfidence);
    const distribution = record.logprob_distribution;
    if (distribution && typeof distribution === "object") {
      const distributionLabel = document.createElement("span");
      distributionLabel.className = "judge-confidence confidence-token";
      const distributionScope =
        record.logprob_distribution_complete === true ? "完整" : "部分，仅诊断";
      distributionLabel.textContent = `档位分布（${distributionScope}）：${Object.entries(
        distribution,
      )
        .map(
          ([score, probability]) =>
            `${score} ${(Number(probability) * 100).toFixed(1)}%`,
        )
        .join(" · ")}`;
      outcome.append(distributionLabel);
    }
  }
  header.append(name, outcome);
  const reason = document.createElement("p");
  reason.textContent = record.reason;
  card.append(header, reason);
  (record.evidence || []).forEach((quote) => {
    const evidence = document.createElement("blockquote");
    evidence.textContent = quote;
    card.append(evidence);
  });
  return card;
}

function renderJudgmentGroup(
  root,
  title,
  records,
  emptyText = "Gate 未通过，未发起维度评分请求。",
) {
  root.replaceChildren();
  if (!records || !Object.keys(records).length) {
    const empty = document.createElement("p");
    empty.className = "llm-help";
    empty.textContent = emptyText;
    root.append(empty);
    return;
  }
  const section = document.createElement("section");
  section.className = "judgment-group";
  const heading = document.createElement("h3");
  heading.textContent = title;
  const grid = document.createElement("div");
  grid.className = "judgment-grid";
  Object.entries(records).forEach(([id, record]) =>
    grid.append(judgmentCard(id, record)),
  );
  section.append(heading, grid);
  root.append(section);
}

async function loadJudgment(runId) {
  try {
    const payload = await api(
      `/api/llm-judge/jobs/${state.selectedJobId}/judgments/${runId}`,
    );
    el("judgment-detail").classList.remove("hidden");
    el("judgment-title").textContent =
      `样本 ${payload.seq} · ${payload.status}`;
    const provider = payload.provider ? ` · 渠道 ${payload.provider}` : "";
    el("judgment-usage").textContent =
      `${payload.usage.prompt_tokens} 输入 / ${payload.usage.completion_tokens} 输出 Token · $${payload.usage.cost_usd.toFixed(5)} · ${payload.usage.latency_ms}ms${provider}`;
    const errorBox = el("judgment-error");
    errorBox.textContent = payload.error_message || "";
    errorBox.classList.toggle("hidden", !payload.error_message);
    if (payload.judgment) {
      renderJudgmentGroup(el("judgment-gates"), "Gate", payload.judgment.gates);
      renderJudgmentGroup(
        el("judgment-scores"),
        `${state.rubric.dimensions.reduce((sum, dimension) => sum + dimension.items.length, 0)} 项原子评分`,
        payload.judgment.scores,
        payload.status === "invalid"
          ? "维度阶段输出未通过校验；未校验的原始结果见下方。"
          : "Gate 未通过，未发起维度评分请求。",
      );
    } else {
      el("judgment-gates").textContent = payload.error_message
        ? "没有通过校验的评分结果；模型原始输出见下方。"
        : "没有有效评分结果";
      el("judgment-scores").replaceChildren();
    }
    const rawDetails = el("judgment-raw");
    const hasRawOutput =
      payload.raw_output && Object.keys(payload.raw_output).length > 0;
    rawDetails.classList.toggle("hidden", !hasRawOutput);
    rawDetails.open =
      hasRawOutput && ["invalid", "failed"].includes(payload.status);
    el("judgment-raw-content").textContent = hasRawOutput
      ? JSON.stringify(payload.raw_output, null, 2)
      : "";
    el("judgment-detail").scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  } catch (error) {
    toast(error.message, true);
  }
}

async function createJob(forceCurrent = false) {
  if (!state.config.enabled) return;
  const payload = {
    model: el("llm-model").value,
    budget_limit_usd: Number(el("llm-budget").value),
    ...selectedScopePayload(forceCurrent),
  };
  const description = forceCurrent
    ? `样本 ${payload.seq}`
    : scopeDescription({
        type: payload.scope_type,
        seq: payload.seq,
        start_seq: payload.start_seq,
        end_seq: payload.end_seq,
        count: payload.random_count,
      });
  if (
    !window.confirm(
      `确认使用 ${payload.model} 评分 ${description}？预算上限 $${payload.budget_limit_usd}。`,
    )
  )
    return;
  el("llm-create").disabled = true;
  el("llm-test-one").disabled = true;
  try {
    const job = await api("/api/llm-judge/jobs", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.selectedJobId = job.job_id;
    toast("后台任务已创建");
    await loadJobs();
  } catch (error) {
    toast(error.message, true);
  } finally {
    el("llm-create").disabled = !state.config.enabled;
    el("llm-test-one").disabled = !state.config.enabled;
  }
}

async function controlJob(action) {
  if (!state.selectedJobId) return;
  if (
    action === "cancel" &&
    !window.confirm(
      "确定停止所有尚未开始的样本吗？正在请求中的样本仍会保存结果。",
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

function activateInspectorTab(name) {
  document.querySelectorAll(".inspector-tab").forEach((button) => {
    const selected = button.dataset.tab === name;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-selected", String(selected));
  });
  document.querySelectorAll(".inspector-panel").forEach((panel) => {
    panel.classList.toggle("hidden", panel.dataset.panel !== name);
  });
}

function openInspector(tabName) {
  activateInspectorTab(tabName);
  const dialog = el("inspector-dialog");
  if (!dialog.open) dialog.showModal();
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

function renderPreviewContext(packet) {
  const root = el("inspector-context");
  root.replaceChildren();
  const trigger = document.createElement("p");
  trigger.className = "preview-trigger";
  trigger.textContent = `trigger_type: ${packet.trigger_type}`;
  root.append(trigger);
  packet.context_messages.forEach((message) => {
    appendContextMessage(root, message.role, message.content, message.index);
  });
  if (packet.platform_output_requirements) {
    appendContextMessage(
      root,
      "platform_output_requirements（平台附加，不属于 user）",
      JSON.stringify(packet.platform_output_requirements, null, 2),
    );
  }
  appendContextMessage(
    root,
    packet.candidate_response.role,
    packet.candidate_response.content,
  );
}

async function loadSchemaIntoInspector() {
  const payload = await api("/api/llm-judge/schema");
  el("schema-content").textContent = JSON.stringify(payload.schema, null, 2);
  return payload;
}

async function showSchema() {
  try {
    const payload = await loadSchemaIntoInspector();
    el("inspector-identity").textContent =
      `规则 ${payload.rubric_version} · Schema SHA-256 ${payload.schema_sha256}`;
    openInspector("schema");
  } catch (error) {
    toast(error.message, true);
  }
}

async function showRequestPreview(seq = null, model = null) {
  const selectedSeq = seq ?? Number(el("llm-seq").value);
  const selectedModel = model || el("llm-model").value;
  if (!selectedModel) {
    toast("服务器尚未配置可预览的 Judge 模型", true);
    return;
  }
  try {
    const query = new URLSearchParams({
      seq: String(selectedSeq),
      model: selectedModel,
    });
    const [preview, schema] = await Promise.all([
      api(`/api/llm-judge/request-preview?${query}`),
      loadSchemaIntoInspector(),
    ]);
    renderPreviewContext(preview.sample_packet);
    el("request-content").textContent = JSON.stringify(
      {
        pipeline: {
          phase_1: "gate（始终执行）",
          phase_2: "dimensions（仅 Gate 全部通过后执行）",
        },
        gate_policy: preview.gate_policy,
        method: preview.method,
        endpoint: preview.endpoint,
        headers: preview.headers,
        phases: preview.phases,
      },
      null,
      2,
    );
    el("inspector-identity").textContent =
      `样本 ${preview.seq} · ${preview.model} · Gate ${preview.request_hashes.gate} · Dimensions ${preview.request_hashes.dimensions} · Schema ${schema.schema_sha256}`;
    openInspector("context");
  } catch (error) {
    toast(error.message, true);
  }
}

async function init() {
  try {
    const session = await api("/api/me");
    el("llm-current-user").textContent = session.username;
    state.config = await api("/api/llm-judge/config");
    el("llm-view").classList.remove("hidden");
    el("llm-rubric-version").textContent =
      `规则 ${state.config.rubric_version}`;
    el("llm-rubric-hash").textContent = state.config.rubric_sha256;
    renderModelCatalog();
    ["llm-seq", "llm-start", "llm-end", "llm-random-count"].forEach((id) => {
      el(id).max = String(state.config.sample_count);
    });
    el("llm-limits").textContent =
      `服务器并发 ${state.config.max_concurrency} · 单任务最多 ${state.config.max_job_samples} 条 · 最高预算 $${state.config.max_job_budget_usd}`;
    if (!state.config.enabled) {
      el("llm-disabled").textContent = state.config.disabled_reason;
      el("llm-disabled").classList.remove("hidden");
      el("llm-create").disabled = true;
      el("llm-test-one").disabled = true;
    }
    await loadJobs(true);
    state.pollTimer = window.setInterval(async () => {
      if (!state.selectedJobId || document.hidden) return;
      try {
        await loadJobs();
      } catch (_) {
        /* next poll will retry */
      }
    }, 3000);
  } catch (error) {
    if (error.status === 401) {
      window.location.replace("/");
      return;
    }
    el("llm-view").classList.remove("hidden");
    toast(error.message, true);
  }
}

el("llm-scope").addEventListener("change", renderScopeFields);
el("llm-model-search").addEventListener("input", renderModelCatalog);
el("llm-model").addEventListener("change", renderProviderRoute);
el("llm-refresh-models").addEventListener("click", refreshModelCatalog);
el("llm-create").addEventListener("click", () => createJob(false));
el("llm-test-one").addEventListener("click", () => createJob(true));
el("llm-refresh").addEventListener("click", () => loadJobs());
el("job-pause").addEventListener("click", () => controlJob("pause"));
el("job-resume").addEventListener("click", () => controlJob("resume"));
el("job-cancel").addEventListener("click", () => controlJob("cancel"));
el("llm-show-request").addEventListener("click", () => showRequestPreview());
el("llm-show-schema").addEventListener("click", showSchema);
el("job-schema").addEventListener("click", showSchema);
document.querySelectorAll(".inspector-tab").forEach((button) => {
  button.addEventListener("click", () =>
    activateInspectorTab(button.dataset.tab),
  );
});
el("inspector-close").addEventListener("click", () =>
  el("inspector-dialog").close(),
);

init();
