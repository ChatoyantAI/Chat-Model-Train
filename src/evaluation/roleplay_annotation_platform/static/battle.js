"use strict";

const battleState = { pairSeq: 1, total: 0, loading: false, modelsVisible: false };
const el = (id) => document.getElementById(id);

async function api(path) {
  const response = await fetch(path, {
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
  });
  let payload = {};
  try { payload = await response.json(); } catch (_) { payload = {}; }
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
  toast.timer = window.setTimeout(() => box.classList.add("hidden"), 3200);
}

function renderMessages(messages) {
  const root = el("battle-messages");
  root.replaceChildren();
  const labels = {
    system: "SYSTEM",
    greeting: "GREETING",
    user: "USER",
    assistant: "ASSISTANT HISTORY",
  };
  for (const message of messages) {
    const article = document.createElement("article");
    article.className = `message message-${message.display_role}`;
    const label = document.createElement("div");
    label.className = "message-label";
    label.textContent = labels[message.display_role] || message.display_role;
    const content = document.createElement("pre");
    content.textContent = message.content;
    article.append(label, content);
    root.append(article);
  }
}

function renderCandidate(candidate) {
  const card = el(`battle-candidate-${candidate.label.toLowerCase()}`);
  card.classList.toggle("winner", candidate.online_selected);
  card.classList.toggle("not-selected", !candidate.online_selected);
  card.querySelector(".winner-badge").classList.toggle("hidden", !candidate.online_selected);
  card.querySelector("pre").textContent = candidate.content;
}

function hideModels() {
  battleState.modelsVisible = false;
  document.querySelectorAll(".model-name").forEach((node) => {
    node.textContent = "";
    node.classList.add("hidden");
  });
  el("model-reveal-button").textContent = "查看模型名称";
}

async function toggleModels() {
  if (battleState.modelsVisible) {
    hideModels();
    return;
  }
  try {
    const payload = await api(`/api/battle/${battleState.pairSeq}/models`);
    for (const model of payload.models) {
      const card = el(`battle-candidate-${model.label.toLowerCase()}`);
      const name = card.querySelector(".model-name");
      name.textContent = model.model_name;
      name.classList.remove("hidden");
    }
    battleState.modelsVisible = true;
    el("model-reveal-button").textContent = "隐藏模型名称";
  } catch (error) {
    toast(error.message, true);
  }
}

async function loadBattle(pairSeq) {
  if (battleState.loading) return;
  battleState.loading = true;
  try {
    const payload = await api(`/api/battle/${pairSeq}`);
    battleState.pairSeq = payload.pair_seq;
    battleState.total = payload.total;
    el("battle-jump-input").value = String(payload.pair_seq);
    el("battle-jump-input").max = String(payload.total);
    el("battle-total").textContent = `/ ${payload.total}`;
    el("battle-prev").disabled = payload.pair_seq <= 1;
    el("battle-next").disabled = payload.pair_seq >= payload.total;
    el("battle-trigger").textContent = `触发：${payload.trigger_type}`;
    el("battle-time").textContent = payload.source_completed_at
      ? `投票时间：${payload.source_completed_at}`
      : "投票时间：未知";
    el("online-selected").textContent = `候选 ${payload.online_selected}`;
    el("battle-context-warning").classList.toggle("hidden", payload.context_match);
    hideModels();
    renderMessages(payload.messages);
    payload.candidates.forEach(renderCandidate);
    window.scrollTo({ top: 0, behavior: "instant" });
  } catch (error) {
    if (error.status === 401) {
      window.location.replace("/");
      return;
    }
    toast(error.message, true);
  } finally {
    battleState.loading = false;
  }
}

async function init() {
  try {
    const session = await api("/api/me");
    el("battle-current-user").textContent = session.username;
    el("battle-view").classList.remove("hidden");
    await loadBattle(1);
  } catch (error) {
    if (error.status === 401) {
      window.location.replace("/");
      return;
    }
    el("battle-view").classList.remove("hidden");
    toast(error.message, true);
  }
}

el("battle-prev").addEventListener("click", () => loadBattle(battleState.pairSeq - 1));
el("battle-next").addEventListener("click", () => loadBattle(battleState.pairSeq + 1));
el("battle-jump").addEventListener("click", () => {
  const target = Number(el("battle-jump-input").value);
  if (!Number.isInteger(target) || target < 1 || target > battleState.total) {
    toast(`请输入 1–${battleState.total} 之间的投票组序号`, true);
    return;
  }
  loadBattle(target);
});
el("battle-jump-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter") el("battle-jump").click();
});
el("model-reveal-button").addEventListener("click", toggleModels);
window.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement) return;
  if (event.key === "ArrowLeft" && battleState.pairSeq > 1) loadBattle(battleState.pairSeq - 1);
  if (event.key === "ArrowRight" && battleState.pairSeq < battleState.total) loadBattle(battleState.pairSeq + 1);
});

init();
