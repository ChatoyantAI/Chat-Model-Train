"use strict";
const $ = (id) => document.getElementById(id);
const state = {
  me: null,
  rubric: null,
  models: [],
  seq: null,
  conversation: null,
  result: null,
  job: null,
  jobId: "",
  page: 1,
  total: 0,
  tab: "result",
  nonce: "",
  loading: false,
};
const labels = {
  queued: "等待 / 待重试",
  running: "运行中",
  paused: "已暂停",
  completed: "已完成",
  completed_errors: "完成但有错误",
  failed: "调用失败",
  invalid: "结果无效",
  interrupted: "服务中断",
  cancelled: "已取消",
};
const esc = (v) =>
  String(v ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const money = (v) => (v == null ? "费用未返回" : `$${Number(v).toFixed(4)}`);
const number = (v) => Number(v || 0).toLocaleString();
const badge = (text, cls = "") =>
  `<span class="badge ${cls}">${esc(text)}</span>`;
function notice(message) {
  $("notice").hidden = !message;
  $("notice").textContent = message || "";
}
async function api(path, options = {}) {
  const response = await fetch(path, {
    credentials: "same-origin",
    ...options,
    headers: {
      "Content-Type": "application/json",
      "X-Requested-With": "conversation-filter",
      ...(options.headers || {}),
    },
  });
  let body;
  try {
    body = await response.json();
  } catch {
    throw Error(`服务返回异常（${response.status}）`);
  }
  if (!response.ok) {
    if (response.status === 401 && state.me) {
      state.me = null;
      $("workbench").hidden = true;
      $("login-panel").hidden = false;
    }
    throw Error(
      typeof body.detail === "string"
        ? body.detail
        : JSON.stringify(body.detail || body),
    );
  }
  return body;
}
const post = (path, body) =>
  api(path, { method: "POST", body: JSON.stringify(body) });
function run(fn) {
  return async (event) => {
    try {
      await fn(event);
    } catch (error) {
      notice(error.message);
    }
  };
}
function textBlock(text) {
  return `<div class="textblock">${String(text || "")
    .split(/\n\s*\n/)
    .map((p) => `<p>${esc(p)}</p>`)
    .join("")}</div>`;
}
function references(ids) {
  return (ids || [])
    .map(
      (id) => `<button class="ref" data-jump="${esc(id)}">${esc(id)}</button>`,
    )
    .join("");
}
function filters() {
  return {
    q: $("search").value.trim(),
    language: $("language").value,
    character: $("character-filter").value.trim(),
    user: $("user-filter").value.trim(),
  };
}
function resultQuery() {
  return {
    minimum: $("minimum").value,
    exclude_major: $("exclude-major").checked,
  };
}
function listQuery() {
  return new URLSearchParams({
    ...filters(),
    ...resultQuery(),
    page: state.page,
    limit: 30,
    job_id: state.jobId,
    result_status: state.jobId ? $("result-status").value : "",
  });
}
async function loadList() {
  const position = $("conversation-list").scrollTop;
  const data = await api("/api/conversations?" + listQuery());
  state.total = data.total;
  $("list-count").textContent = number(data.total) + " 段";
  $("page-label").textContent =
    `${state.page} / ${Math.max(1, Math.ceil(data.total / 30))}`;
  $("prev-page").disabled = state.page <= 1;
  $("next-page").disabled = state.page * 30 >= data.total;
  $("conversation-list").innerHTML = data.rows.length
    ? data.rows
        .map(
          (row) =>
            `<button class="conversation-item ${state.seq === row.seq ? "selected" : ""}" data-seq="${row.seq}"><strong>#${row.seq} · ${esc(row.character_name)}</strong><span>${badge(row.language)} ${row.status ? badge(labels[row.status] || row.status, ["failed", "invalid", "interrupted"].includes(row.status) ? "bad" : row.status === "completed" ? "good" : "") : ""}</span><div class="item-meta">${row.turn_count} 轮 · ${row.candidate_count} 条候选<br>用户 ${esc(row.user_key.slice(0, 10))}${row.status === "completed" ? ` · 最低 ${row.min_score} 分 · N/A ${row.na_count}` : ""}</div></button>`,
        )
        .join("")
    : '<div class="empty">当前范围没有对话。</div>';
  $("conversation-list").scrollTop = position;
  if (!state.seq && data.rows.length)
    await selectConversation(data.rows[0].seq);
}
async function selectConversation(seq) {
  state.seq = seq;
  state.conversation = null;
  state.result = null;
  $("conversation-title").textContent = "正在读取对话…";
  $("conversation-content").innerHTML =
    '<div class="empty">正在读取完整上下文…</div>';
  const packet = await api("/api/conversations/" + seq);
  if (state.seq !== seq) return;
  state.conversation = packet;
  $("conversation-title").textContent = `#${seq} · ${packet.character_name}`;
  $("score-current").disabled = false;
  $("conversation-meta").innerHTML =
    `${badge(packet.language)} ${packet.turn_count} 轮 / ${packet.candidate_count} 条候选 · ${esc(packet.conversation_id)}<br>用户 ${esc(packet.user_key.slice(0, 12))} <button data-copy-user>用此用户筛选</button> · 角色 <button data-copy-role>只看此角色</button>`;
  renderConversation();
  document
    .querySelectorAll(".conversation-item")
    .forEach((el) =>
      el.classList.toggle("selected", Number(el.dataset.seq) === seq),
    );
  await loadResult();
}
function renderConversation() {
  const p = state.conversation;
  if (!p) return;
  const basis = p.context_basis;
  const report = state.me?.data?.report || {};
  $("greeting-coverage").textContent =
    `Greeting 来自角色资料，按平台规则替换占位符 · ${number(report["character_descriptions.greeting"])} 段已关联；${number(report.role_card_language_fallback)} 段使用已有英文资料；${number(report.missing)} 段仍缺失。未生成或补写，可展开查看原文。`;
  let html = `<div class="context-note"><strong>离线上下文快照</strong> · 不是原始请求逐字回放。${esc((basis.known_gaps || []).join("；"))}<details><summary>查看来源说明</summary>${basis.notes.map((n) => `<p>${esc(n)}</p>`).join("")}<p>正文差异：User ${p.provenance.query_text_differences} / Assistant ${p.provenance.target_text_differences}；候选以粗筛文件为准。</p><p>角色资料来源：${esc(p.provenance.profile_source)} · ${esc(p.provenance.profile_language)}</p></details></div>`;
  for (const s of p.system_contexts || [])
    html += `<details class="context-card" id="${esc(s.id)}"><summary>SYSTEM · ${esc(s.template)} · ${esc(s.response_language)}</summary><p>平台模板拼接，适用于 ${esc(s.applies_to.join("、"))}。不是数据库存储的 prompt。</p>${textBlock(s.content)}</details>`;
  if (p.platform_assembly)
    html += `<p class="context-note">与单条 Judge / Battle 共用平台拼接代码 · ${esc(p.platform_assembly.version)}。配置为当前快照，不是历史逐轮请求回放；目标语言按候选数据，不跟随用户正文。</p>`;
  html += `<details class="context-card" id="role_profile"><summary>角色基础资料（不含数据库 prompt）</summary>${
    Object.entries(p.role_profile)
      .map(
        ([k, v]) =>
          `<div class="field"><h4>${esc(k)}</h4>${textBlock(typeof v === "object" ? JSON.stringify(v) : v)}</div>`,
      )
      .join("") || "<p>未找到角色详细资料。</p>"
  }</details>`;
  if (p.persona)
    html += `<details class="context-card" id="persona"><summary>玩家设定 Persona</summary>${Object.entries(
      p.persona,
    )
      .map(
        ([k, v]) =>
          `<div class="field"><h4>${esc(k)}</h4>${textBlock(v)}</div>`,
      )
      .join("")}</details>`;
  const source = p.provenance.greeting_source;
  const title =
    source === "character_descriptions.greeting"
      ? "Greeting · 角色资料原文"
      : source === "conversations.initial_greeting" ||
          source === "initial_greeting"
        ? "Greeting · 会话保存原文"
        : source === "messages.first_assistant"
          ? "开场 · 最早历史 Assistant"
          : "Greeting · 记录缺失";
  html += `<article class="message greeting" id="msg-greeting"><div class="message-header"><strong>${p.platform_assembly ? "Greeting · 按平台规则替换占位符与移除图片链接" : title}</strong>${badge(source)} ${badge(p.provenance.greeting_language || "")}</div>${textBlock(p.greeting.content || "资料中未找到开场正文，不自行生成补写。")}${p.greeting.source_content ? `<details><summary>角色资料中的开场原文（未生成或改写）</summary>${textBlock(p.greeting.source_content)}</details>` : ""}</article>`;
  html += p.messages
    .map(
      (m) =>
        `<article class="message ${m.scope === "candidate" ? "candidate" : "context-only"} ${m.role}" id="msg-${esc(m.id)}"><div class="message-header"><strong>${esc(m.id)} · ${m.role === "user" ? "USER" : "ASSISTANT"}</strong>${badge(m.scope === "candidate" ? "待评候选" : "仅上下文", m.scope === "candidate" ? "good" : "")}</div>${textBlock(m.content)}${
          m.platform_output_requirements
            ? `<details class="requirements"><summary>平台输出要求（不属于用户原文）</summary>${Object.entries(
                m.platform_output_requirements,
              )
                .map(
                  ([k, v]) =>
                    `<p>${esc(k)}：${esc(typeof v === "object" ? JSON.stringify(v) : v)}</p>`,
                )
                .join("")}</details>`
            : ""
        }</article>`,
    )
    .join("");
  $("conversation-content").innerHTML = html;
  $("conversation-content").scrollTop = 0;
}
async function loadResult() {
  const seq = state.seq,
    job = state.jobId;
  let result = null;
  if (job && seq) {
    try {
      result = await api(
        `/api/jobs/${job}/result/${seq}?` + new URLSearchParams(resultQuery()),
      );
    } catch (error) {
      if (!error.message.includes("不在任务")) throw error;
    }
  }
  if (seq !== state.seq || job !== state.jobId) return;
  state.result = result;
  renderReview();
}
function renderReview() {
  const content = $("review-content"),
    position = content.scrollTop;
  if (state.tab === "rubric") {
    renderRubric();
    return;
  }
  if (state.tab === "request") {
    content.innerHTML = `<div class="request-actions"><h3>发送前检查</h3><p>这里展示后台实际使用的请求与 Schema。已运行项目优先显示实际发送记录，不另行总结。</p><label>预览模型<input id="preview-model" list="model-options" value="${esc(state.job?.model || $("model-input").value)}"></label><button id="show-request" ${!state.seq ? "disabled" : ""}>查看完整请求</button><button id="show-schema">查看期望输出 Schema</button><p class="muted">20 项均含 value、reason、turn_ids；另外记录局部问题和数据限制。不要求自报置信度，不发送 logprobs、temperature 或输出 Token 上限。DeepInfra 选项直连 DeepInfra，不经过 OpenRouter；其他模型沿用后台路由。Key 不会返回。</p></div>`;
    return;
  }
  const r = state.result;
  if (!state.jobId) {
    content.innerHTML =
      '<div class="empty">当前为数据浏览模式。<br>选择一个任务查看结果，或点击“新建评分任务”。<br>不会自动发起付费调用。</div>';
    return;
  }
  if (!r) {
    content.innerHTML = '<div class="empty">当前对话不在所选任务中。</div>';
    return;
  }
  let html = `<h3>对话 #${r.seq} ${badge(labels[r.status] || r.status, r.status === "completed" ? "good" : "")}</h3>`;
  if (r.error) html += `<p class="warning">${esc(r.error)}</p>`;
  if (r.output) {
    const screen = r.screening;
    html += `<div class="summary-card ${screen.label === "eligible" ? "" : "review"}"><strong>${screen.label === "eligible" ? "满足当前筛选条件" : "有低分或问题，建议复核"}</strong><br>${screen.numeric_items} 项数字分 / ${screen.not_applicable.length} 项 N/A${screen.below_threshold.length ? "<br>低于阈值：" + esc(screen.below_threshold.join("、")) : ""}<p class="small">整段候选结论，不是每条回复的通过证明。</p></div>`;
    for (const issue of r.output.issues)
      html += `<div class="issue ${issue.severity === "major" ? "major" : ""}"><strong>${issue.severity === "major" ? "重要问题" : "局部问题"} · ${esc(issue.item_ids.join("、"))}</strong><p>${esc(issue.reason)}</p>${issue.affects_later ? "<p>可能影响后文，拆条前需检查相关片段。</p>" : ""}${references(issue.turn_ids)}</div>`;
    for (const warning of [...r.output.data_limitations, ...r.output.warnings])
      html += `<p class="warning">${esc(warning)}</p>`;
    for (const [group, name] of Object.entries(state.rubric.groups)) {
      html += `<h3 class="review-group">${esc(name)}</h3>`;
      for (const item of state.rubric.items.filter((i) => i.group === group)) {
        const rec = r.output.scores[item.id];
        html += `<div class="score-card"><div class="score-title"><strong>${item.id} · ${esc(item.name)}</strong><span class="score ${Number.isInteger(rec.value) && rec.value < 2 ? "low" : ""}">${esc(rec.value)}</span></div><p>${esc(rec.reason || "模型没有给出理由")}</p>${references(rec.turn_ids)}</div>`;
      }
    }
  } else
    html +=
      '<div class="empty">尚无通过基本结构校验的评分。<br>失败或无效结果的原始输出仍保留在下方。</div>';
  html += '<h3 class="review-group">请求尝试与费用</h3>';
  for (const a of r.attempts) {
    html += `<div class="attempt">尝试 ${a.attempt} · ${esc(labels[a.status] || a.status)} · ${money(a.cost)}${a.raw.cost_basis === "usage.estimated_cost" ? "（渠道估算）" : ""}<br>${number(a.prompt_tokens)} 输入 / ${number(a.completion_tokens)} 输出 Token · ${(a.latency_ms / 1000).toFixed(1)} 秒<br>${esc(a.raw.provider || "渠道未返回")} ${esc(a.raw.model || "")}<br>${esc(a.error)}<br><button data-raw-attempt="${a.id}">查看原始输出</button></div>`;
  }
  content.innerHTML = html;
  content.scrollTop = position;
}
function renderRubric() {
  const r = state.rubric;
  if (!r) return;
  let html = `<p class="muted">${esc(r.version)} · ${esc(r.sha256.slice(0, 16))}…</p><p><a href="/rubric.md" target="_blank" rel="noopener">查看完整标准原文</a></p><details class="rule-card" open><summary>统一判断方式与职责边界</summary><div class="level rule-intro">${esc(r.rules)}</div></details>`;
  for (const [group, name] of Object.entries(r.groups)) {
    html += `<h3 class="review-group">${esc(name)}</h3>`;
    for (const item of r.items.filter((i) => i.group === group)) {
      html += `<details class="rule-card"><summary>${item.id} · ${esc(item.name)}</summary>${Object.entries(
        item.levels,
      )
        .map(
          ([value, text]) =>
            `<div class="level"><strong>${value} 分</strong>${esc(text)}</div>`,
        )
        .join(
          "",
        )}<div class="level na"><strong>N/A</strong>${esc(item.na || "不允许，必须给 0、1、2 或 3。")}</div></details>`;
    }
  }
  html += '<h3 class="review-group">对话校准案例</h3>';
  for (const e of r.examples)
    html += `<details class="rule-card"><summary>${esc(e.title)}</summary><div class="level"><p>${esc(e.context)}</p>${textBlock(e.dialogue)}<p><strong>判定说明：</strong>${esc(e.judgment)}</p></div></details>`;
  $("review-content").innerHTML = html;
}
async function loadJobs() {
  const current = state.jobId,
    data = await api("/api/jobs");
  $("job-select").innerHTML =
    '<option value="">浏览全部对话（不运行评分）</option>' +
    data.jobs
      .map(
        (j) =>
          `<option value="${j.id}">${esc(j.model)} · ${j.total} 段 · ${esc(labels[j.status] || j.status)} · ${money(j.spent)}</option>`,
      )
      .join("");
  $("job-select").value = current;
}
async function loadJob() {
  if (!state.jobId) {
    state.job = null;
    $("job-controls").hidden = true;
    $("result-filters").hidden = true;
    $("job-progress").hidden = true;
    $("job-summary").textContent = "数据浏览中，没有启动任务";
    $("job-note").textContent = "";
    return;
  }
  const j = await api(
    "/api/jobs/" + state.jobId + "?" + new URLSearchParams(resultQuery()),
  );
  state.job = j;
  $("job-controls").hidden = false;
  $("result-filters").hidden = false;
  $("job-progress").hidden = false;
  const complete = Object.entries(j.counts)
    .filter(([s]) => !["queued", "running"].includes(s))
    .reduce((sum, [, n]) => sum + n, 0);
  $("job-progress").max = j.total;
  $("job-progress").value = complete;
  $("job-summary").textContent =
    `${labels[j.status] || j.status} · ${complete}/${j.total} · ${money(j.spent)}${j.model.startsWith("deepinfra::") ? "（估算）" : ""} / $${j.budget} · 条件满足 ${j.eligible} 段`;
  $("job-note").textContent =
    Object.entries(j.counts)
      .map(([s, n]) => `${labels[s] || s} ${n}`)
      .join(" · ") +
    (j.unknown_cost_attempts
      ? ` · ${j.unknown_cost_attempts} 次费用未知`
      : "") +
    (j.message ? " · " + j.message : "");
}
async function refreshModels(force = false) {
  const data = force
    ? await post("/api/models/refresh", {})
    : await api("/api/models");
  state.models = data.models.filter((m) =>
    ["openrouter", "deepinfra"].includes(m.gateway),
  );
  $("model-options").innerHTML = state.models
    .map((m) => `<option value="${esc(m.id)}">${esc(m.label)}</option>`)
    .join("");
  if (!$("model-input").value)
    $("model-input").value = data.default_model || state.models[0]?.id || "";
  $("model-note").textContent =
    `${state.models.length} 个模型${data.stale ? " · 使用缓存目录" : ""}${data.error ? " · " + data.error : ""}`;
}
const loadJobBase = loadJob;
loadJob = async function () {
  await loadJobBase();
  if (!state.job) return;
  const j = state.job;
  $("job-note").textContent +=
    ` · 当前在途 ${j.counts.running || 0} / ${j.concurrency} 个槽位 · 每段最多 ${j.max_attempts} 次尝试` +
    (j.rate_limit_wait_seconds
      ? ` · 渠道限流，${j.rate_limit_wait_seconds} 秒后恢复派发`
      : "");
};
function openJob(mode = "random") {
  if (!state.me?.api_configured) {
    notice("服务器尚未配置所选渠道的 Key，不能启动评分。");
    return;
  }
  $("scope-mode").value = mode;
  $("selection-summary").textContent = "";
  $("ack-snapshot").checked = false;
  state.nonce = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  $("job-dialog").showModal();
}
function getScope() {
  return {
    ...filters(),
    mode: $("scope-mode").value,
    seq: state.seq,
    count: Number($("scope-count").value),
    start: Number($("scope-start").value),
    end: Number($("scope-end").value),
    seed: Number($("scope-seed").value),
    character_cap: Number($("scope-role-cap").value),
    user_cap: Number($("scope-user-cap").value),
  };
}
function showRaw(title, value) {
  $("raw-title").textContent = title;
  $("raw-content").textContent =
    typeof value === "string" ? value : JSON.stringify(value, null, 2);
  $("raw-dialog").showModal();
}
async function enter() {
  state.me = await api("/api/me");
  $("login-panel").hidden = true;
  $("workbench").hidden = false;
  $("logout").hidden = false;
  $("username").textContent = state.me.username;
  $("version").textContent = state.me.version;
  $("new-job").disabled = !state.me.data.ready || !state.me.api_configured;
  const r = state.me.data.report || {};
  $("dataset-stats").textContent =
    `${number(r.conversations)} 段对话 · ${number(r.candidates)} 条候选回复 · ${number(r.characters)} 个角色 · ${number(r.users)} 个用户`;
  $("language").innerHTML =
    '<option value="">全部语言</option>' +
    Object.entries(state.me.data.languages || {})
      .map(
        ([lang, n]) =>
          `<option value="${esc(lang)}">${esc(lang)} · ${number(n)} 段</option>`,
      )
      .join("");
  state.rubric = await api("/api/rubric");
  await Promise.all([refreshModels(), loadJobs()]);
  await loadList();
  if (!state.me.api_configured)
    notice("数据可查看，但服务器尚未配置所选渠道的 Key。");
}
$("login-form").addEventListener(
  "submit",
  run(async (e) => {
    e.preventDefault();
    await post("/api/login", { username: $("login-name").value });
    notice("");
    await enter();
  }),
);
$("logout").addEventListener(
  "click",
  run(async () => {
    await post("/api/logout", {});
    location.reload();
  }),
);
$("filters").addEventListener(
  "submit",
  run(async (e) => {
    e.preventDefault();
    state.page = 1;
    await loadList();
  }),
);
$("prev-page").addEventListener(
  "click",
  run(async () => {
    state.page--;
    await loadList();
  }),
);
$("next-page").addEventListener(
  "click",
  run(async () => {
    state.page++;
    await loadList();
  }),
);
$("conversation-list").addEventListener(
  "click",
  run(async (e) => {
    const b = e.target.closest("[data-seq]");
    if (b) await selectConversation(Number(b.dataset.seq));
  }),
);
$("job-select").addEventListener(
  "change",
  run(async () => {
    state.jobId = $("job-select").value;
    state.page = 1;
    await loadJob();
    await loadList();
    await loadResult();
  }),
);
$("refresh-jobs").addEventListener(
  "click",
  run(async () => {
    await loadJobs();
    await loadJob();
    await loadList();
    await loadResult();
  }),
);
for (const id of ["result-status", "minimum", "exclude-major"])
  $(id).addEventListener(
    "change",
    run(async () => {
      state.page = 1;
      await loadJob();
      await loadList();
      await loadResult();
    }),
  );
$("new-job").addEventListener("click", () => openJob());
$("score-current").addEventListener("click", () => openJob("current"));
$("refresh-models").addEventListener(
  "click",
  run(() => refreshModels(true)),
);
$("selection-preview").addEventListener(
  "click",
  run(async () => {
    const p = await post("/api/selection-preview", getScope()),
      s = p.summary;
    $("selection-summary").textContent =
      `${s.conversations} 段 / ${s.candidate_replies} 条候选 / ${s.characters} 角色 / ${s.users} 用户${s.shortfall ? " · 约束下缺额 " + s.shortfall + " 段" : ""}`;
  }),
);
$("job-form").addEventListener(
  "submit",
  run(async (e) => {
    e.preventDefault();
    const button = $("start-job");
    button.disabled = true;
    try {
      if (!state.models.some((m) => m.id === $("model-input").value))
        throw Error("请从目录中选择准确的模型 ID");
      const payload = {
        model: $("model-input").value,
        scope: getScope(),
        budget: Number($("scope-budget").value),
        concurrency: Number($("scope-concurrency").value),
        acknowledge_snapshot: $("ack-snapshot").checked,
        nonce: state.nonce,
      };
      const result = await post("/api/jobs", payload);
      state.jobId = result.job_id;
      state.page = 1;
      $("job-dialog").close();
      notice("任务已启动。关闭页面不会停止后台任务。");
      await loadJobs();
      await loadJob();
      await loadList();
      await loadResult();
    } finally {
      button.disabled = false;
    }
  }),
);
$("job-controls").addEventListener(
  "click",
  run(async (e) => {
    const b = e.target.closest("[data-control]");
    if (!b || !state.jobId) return;
    const action = b.dataset.control;
    if (
      action === "cancel" &&
      !confirm("取消尚未发出的请求？已在途请求仍会保存和计费。")
    )
      return;
    let budget = state.job.budget;
    if (["retry", "resume"].includes(action)) {
      const input = prompt(
        "费用停止阈值（美元）。继续 / 重试可能产生新的模型费用。",
        String(budget),
      );
      if (input === null) return;
      budget = Number(input);
    }
    await post(`/api/jobs/${state.jobId}/control`, { action, budget });
    await loadJob();
    await loadList();
  }),
);
for (const [id, only] of [
  ["export-all", false],
  ["export-eligible", true],
])
  $(id).addEventListener("click", () => {
    if (state.jobId)
      location.href =
        `/api/jobs/${state.jobId}/export?` +
        new URLSearchParams({ ...resultQuery(), eligible_only: only });
  });

// The server has the authoritative cap; no browser-side batching.
$("scope-concurrency").max = "100";
$("scope-concurrency").value = "100";
const concurrencyNote = document.createElement("p");
concurrencyNote.className = "muted small";
concurrencyNote.textContent =
  "固定并发槽位：每完成一段立即补下一段，不等整批结束。遇到 429 按渠道要求等待重试；额度、未知费用或服务中断可能暂停。";
$("scope-concurrency").closest(".two").after(concurrencyNote);
const rankingButton = document.createElement("button");
rankingButton.id = "show-ranking";
rankingButton.textContent = "分数排名与汇总";
$("job-controls").append(rankingButton);
const rankingDialog = document.createElement("dialog");
rankingDialog.id = "ranking-dialog";
rankingDialog.innerHTML =
  '<div class="dialog-title"><h2>会话分数排名与汇总</h2><button data-close="ranking-dialog">关闭</button></div><div id="ranking-body"></div>';
document.body.append(rankingDialog);
let rankingPage = 1;
async function showRanking(page = 1) {
  if (!state.jobId) return;
  const jobId = state.jobId;
  const report = await api(`/api/jobs/${jobId}/ranking?page=${page}`);
  if (state.jobId !== jobId) return;
  rankingPage = page;
  $("ranking-body").innerHTML =
    `<p class="warning">${esc(report.note)}</p><p>任务 ${esc(labels[report.job_status] || report.job_status)} · 可排名 ${report.ranked_count} / ${report.total} · 未排名 ${report.excluded_count}（含疑似拒评 ${report.suspected_refusal_count}） · 排名中有复核标记 ${report.review_flagged_ranked_count}</p><p>均分范围：${report.mean_score_range ? report.mean_score_range.map((x) => Number(x).toFixed(3)).join("–") : "暂无"} · 分项数量：${Object.entries(
      report.score_counts,
    )
      .map(([s, n]) => `${esc(s)} 分 ${n}`)
      .join(
        " / ",
      )}</p><p><a href="/api/jobs/${jobId}/ranking?format=csv">下载完整排名 CSV（含未排名条目和全部分项）</a></p><div class="ranking-scroll"><table><thead><tr><th>排名</th><th>对话</th><th>均分</th><th>最低</th><th>低于 2</th><th>3 分项</th><th>适用 / N/A</th><th>复核提示</th></tr></thead><tbody>${report.rows.map((r) => `<tr><td>${r.rank}</td><td><button data-ranking-seq="${r.seq}">#${r.seq} ${esc(r.character_name)}</button></td><td>${Number(r.mean_score).toFixed(3)}</td><td>${r.min_score}</td><td>${r.low_count}</td><td>${r.three_count}</td><td>${r.numeric_count} / ${r.na_count}</td><td>${esc(r.review_notes.join("；") || "无自动标记，尚非人工验证")}</td></tr>`).join("") || '<tr><td colspan="8">暂无可排名结果；已准备的任务不会自动开始。</td></tr>'}</tbody></table></div><div class="row"><button data-ranking-page="${page - 1}" ${page <= 1 ? "disabled" : ""}>上一页</button><span>${page} / ${Math.max(1, Math.ceil(report.ranked_count / report.limit))}</span><button data-ranking-page="${page + 1}" ${page * report.limit >= report.ranked_count ? "disabled" : ""}>下一页</button></div><details><summary>未排名条目示例（最多 20 条，完整列表在 CSV）</summary>${report.excluded.map((r) => `<p>#${r.seq} · ${esc(labels[r.status] || r.status)} · ${esc(r.exclusion_reason)}</p>`).join("")}</details>`;
  if (!rankingDialog.open) rankingDialog.showModal();
}
rankingButton.addEventListener(
  "click",
  run(() => showRanking()),
);
rankingDialog.addEventListener(
  "click",
  run(async (e) => {
    const page = e.target.closest("[data-ranking-page]");
    if (page) await showRanking(Number(page.dataset.rankingPage));
    const seq = e.target.closest("[data-ranking-seq]");
    if (seq) {
      rankingDialog.close();
      await selectConversation(Number(seq.dataset.rankingSeq));
    }
  }),
);
document.querySelectorAll("[data-tab]").forEach((button) =>
  button.addEventListener("click", () => {
    state.tab = button.dataset.tab;
    document.querySelectorAll("[data-tab]").forEach((b) => {
      b.classList.toggle("active", b === button);
      b.setAttribute("aria-selected", String(b === button));
    });
    $("review-content").scrollTop = 0;
    renderReview();
  }),
);
document.addEventListener(
  "click",
  run(async (e) => {
    const close = e.target.closest("[data-close]");
    if (close) $(close.dataset.close).close();
    const jump = e.target.closest("[data-jump]");
    if (jump) {
      const id = jump.dataset.jump,
        target = $(id) || $("msg-" + id);
      if (target) {
        if (target.tagName === "DETAILS") target.open = true;
        target.scrollIntoView({ block: "center", behavior: "smooth" });
        target.classList.add("highlight");
        setTimeout(() => target.classList.remove("highlight"), 1800);
      } else notice("模型引用的位置未识别，请结合理由人工核对。");
    }
    if (e.target.closest("[data-copy-user]")) {
      $("user-filter").value = state.conversation.user_key;
      state.page = 1;
      await loadList();
    }
    if (e.target.closest("[data-copy-role]")) {
      $("character-filter").value = state.conversation.character_id;
      state.page = 1;
      await loadList();
    }
    const a = e.target.closest("[data-raw-attempt]");
    if (a) {
      const record = state.result.attempts.find(
        (x) => x.id === Number(a.dataset.rawAttempt),
      );
      showRaw("原始输出与完成原因（不展示推理正文）", record);
    }
    if (e.target.id === "show-request") {
      const args = state.jobId
        ? { job_id: state.jobId }
        : { model: $("preview-model").value };
      showRaw(
        "实际请求",
        await api(
          `/api/request-preview/${state.seq}?` + new URLSearchParams(args),
        ),
      );
    }
    if (e.target.id === "show-schema") {
      const args = state.jobId
        ? { job_id: state.jobId }
        : { model: $("preview-model").value };
      const preview = await api(
        `/api/request-preview/${state.seq}?` + new URLSearchParams(args),
      );
      showRaw("期望输出 Schema", preview.schema);
    }
  }),
);
const old = new URL(location.href);
old.port = "8766";
old.pathname = "/";
old.search = "";
old.hash = "";
$("old-site").href = old.href;
(async () => {
  try {
    await enter();
  } catch (error) {
    if (error.message !== "请先输入用户名") notice(error.message);
  }
})();
const renderReviewBody = renderReview;
renderReview = function () {
  const content = $("review-content"),
    key =
      state.tab === "rubric"
        ? "rubric"
        : state.tab === "request"
          ? `request:${state.seq}:${state.jobId}`
          : "";
  if (key && content.dataset.renderKey === key) return;
  renderReviewBody();
  content.dataset.renderKey = key;
};
setInterval(async () => {
  if (!state.me || state.loading || !state.jobId || $("job-dialog").open)
    return;
  state.loading = true;
  try {
    const previous = state.job?.status;
    await loadJob();
    if (
      previous !== state.job?.status ||
      ["queued", "running", "paused"].includes(state.job?.status) ||
      state.result?.status === "running" ||
      state.result?.status === "queued"
    ) {
      await loadList();
      await loadResult();
      await loadJobs();
    }
  } catch (error) {
    notice(error.message);
  } finally {
    state.loading = false;
  }
}, 5000);
