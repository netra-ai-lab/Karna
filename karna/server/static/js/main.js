// Karna web app: plain ES module, no build step. Talks to the REST API only.
import { recordWav } from "./record.js";

const $ = (sel) => document.querySelector(sel);
const state = { info: null, jobs: [], polling: new Set(), recorder: null };

function keyHeader() {
  let key = "";
  try { key = localStorage.getItem("karna.key") || ""; } catch { /* storage blocked */ }
  return key ? { "X-API-Key": key } : {};
}

async function api(path, options = {}) {
  const res = await fetch(path, { ...options, headers: { ...keyHeader(), ...(options.headers || {}) } });
  if (res.status === 204) return null;
  const body = res.headers.get("content-type")?.includes("json") ? await res.json() : await res.text();
  if (!res.ok) throw new Error(body?.detail || body || res.statusText);
  return body;
}

function toast(text) {
  const el = $("#toast");
  el.textContent = text;
  el.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { el.hidden = true; }, 2600);
}

// ── engine status ──────────────────────────────────────────────────────
async function refreshInfo() {
  try {
    const info = await api("/v1/info");
    state.info = info;
    $("#version").textContent = `Karna ${info.version}`;
    const el = $("#engine");
    if (info.error) { el.className = "status status-error"; el.textContent = "Model failed to load"; el.title = info.error; }
    else if (info.models_ready) { el.className = "status status-done"; el.textContent = `Ready · ${(info.device || "cpu").toUpperCase()}`; }
    else { el.className = "status status-running"; el.textContent = "Loading models…"; setTimeout(refreshInfo, 1500); }
    const select = $("#model");
    if (!select.options.length) {
      for (const m of info.models) {
        const opt = new Option(`${m.name} (${m.params})`, m.name, false, m.name === info.default_model);
        opt.title = m.about;
        select.add(opt);
      }
    }
    $("#drop-meta").textContent = `Khmer or English speech · wav, flac, ogg, mp3 · up to ${info.max_seconds} s each`;
    $("#key-field").hidden = !info.auth;
    renderApi();
  } catch (e) {
    $("#engine").className = "status status-error";
    $("#engine").textContent = "Server unreachable";
    setTimeout(refreshInfo, 3000);
  }
}

// ── jobs ───────────────────────────────────────────────────────────────
async function refreshJobs() {
  try { state.jobs = await api("/v1/jobs"); } catch (e) { showError(e.message); return; }
  renderJobs();
  for (const job of state.jobs) if (job.status === "queued" || job.status === "running") poll(job.id);
}

function poll(id) {
  if (state.polling.has(id)) return;
  state.polling.add(id);
  const tick = async () => {
    let job;
    try { job = await api(`/v1/jobs/${id}`); } catch { state.polling.delete(id); return; }
    const i = state.jobs.findIndex((j) => j.id === id);
    if (i >= 0) state.jobs[i] = job; else state.jobs.unshift(job);
    renderJobs();
    if (job.status === "queued" || job.status === "running") setTimeout(tick, 400);
    else state.polling.delete(id);
  };
  tick();
}

const STATUS = { queued: "Queued", running: "Transcribing…", done: "Done", failed: "Failed" };

function renderJobs() {
  const list = $("#jobs");
  list.replaceChildren(...state.jobs.map(renderJob));
  $("#empty").hidden = state.jobs.length > 0;
}

function renderJob(job) {
  const li = document.createElement("li");
  li.className = "job";
  const head = document.createElement("div");
  head.className = "job-head";
  const name = Object.assign(document.createElement("span"), { className: "job-name", textContent: job.filename });
  const status = Object.assign(document.createElement("span"), { className: `status status-${job.status}`, textContent: STATUS[job.status] || job.status });
  const meta = Object.assign(document.createElement("span"), { className: "meta",
    textContent: `${job.model} · ${job.duration_s.toFixed(1)} s audio` + (job.result ? ` · ${job.result.seconds.toFixed(2)} s on ${job.result.device}` : "") });
  const actions = document.createElement("span");
  actions.className = "job-actions";
  if (job.status === "done") {
    actions.append(button("Copy", () => navigator.clipboard.writeText(job.result.text).then(() => toast("Copied"))));
    actions.append(button("Download", () => download(job)));
  }
  actions.append(deleteButton(job));
  head.append(name, status, meta, actions);
  li.append(head);
  if (job.status === "done") {
    const text = Object.assign(document.createElement("p"), { className: "job-text", textContent: job.result.text || "(no speech recognised)" });
    li.append(text);
  } else if (job.status === "failed") {
    li.append(Object.assign(document.createElement("p"), { className: "job-error", textContent: job.error }));
  }
  return li;
}

function button(label, onClick) {
  const b = Object.assign(document.createElement("button"), { type: "button", className: "link-btn", textContent: label });
  b.addEventListener("click", onClick);
  return b;
}

// Two clicks, no dialog: the first arms the button for a few seconds.
function deleteButton(job) {
  const b = button("Delete", async () => {
    if (!b.classList.contains("armed")) {
      b.classList.add("armed");
      b.textContent = "Delete?";
      setTimeout(() => { b.classList.remove("armed"); b.textContent = "Delete"; }, 3000);
      return;
    }
    try { await api(`/v1/jobs/${job.id}`, { method: "DELETE" }); } catch (e) { showError(e.message); return; }
    state.jobs = state.jobs.filter((j) => j.id !== job.id);
    renderJobs();
  });
  return b;
}

async function download(job) {
  const res = await fetch(`/v1/jobs/${job.id}/export?format=txt`, { headers: keyHeader() });
  const url = URL.createObjectURL(await res.blob());
  const a = Object.assign(document.createElement("a"), { href: url, download: `${job.filename.replace(/\.[^.]+$/, "") || job.id}.txt` });
  a.click();
  URL.revokeObjectURL(url);
}

// ── upload ─────────────────────────────────────────────────────────────
function showError(text) { $("#upload-error").textContent = text || ""; }

async function upload(blob, filename) {
  showError("");
  const form = new FormData();
  form.append("file", blob, filename);
  form.append("model", $("#model").value);
  try {
    const job = await api("/v1/transcribe", { method: "POST", body: form });
    state.jobs.unshift(job);
    renderJobs();
    poll(job.id);
  } catch (e) {
    showError(`${filename}: ${e.message}`);
  }
}

function uploadFiles(files) {
  for (const f of files) upload(f, f.name);
}

$("#file").addEventListener("change", (e) => { uploadFiles(e.target.files); e.target.value = ""; });
// Files can be dropped anywhere on the page.
let depth = 0;
addEventListener("dragenter", (e) => { e.preventDefault(); depth++; document.body.classList.add("dragging"); });
addEventListener("dragleave", () => { if (--depth <= 0) { depth = 0; document.body.classList.remove("dragging"); } });
addEventListener("dragover", (e) => e.preventDefault());
addEventListener("drop", (e) => {
  e.preventDefault();
  depth = 0;
  document.body.classList.remove("dragging");
  if (e.dataTransfer?.files?.length) uploadFiles(e.dataTransfer.files);
});

$("#key").addEventListener("change", (e) => {
  try { localStorage.setItem("karna.key", e.target.value); } catch { /* storage blocked */ }
  refreshJobs();
});

// ── recording ──────────────────────────────────────────────────────────
$("#record").addEventListener("click", async () => {
  const btn = $("#record"), label = $("#record-label");
  if (state.recorder) { state.recorder.stop(); return; }
  showError("");
  try {
    const maxSeconds = state.info?.max_seconds || 30;
    state.recorder = await recordWav({
      maxSeconds,
      onTick: (s) => { label.textContent = `Stop · ${s.toFixed(0)} s`; },
    });
    btn.setAttribute("aria-pressed", "true");
    label.textContent = "Stop · 0 s";
    const wav = await state.recorder.done;
    const stamp = new Date().toISOString().slice(0, 19).replace("T", " ");
    await upload(wav, `Recording ${stamp}.wav`);
  } catch (e) {
    showError(e.name === "NotAllowedError" ? "Microphone access was refused." : e.message);
  } finally {
    state.recorder = null;
    btn.setAttribute("aria-pressed", "false");
    label.textContent = "Record";
  }
});

// ── API tab ────────────────────────────────────────────────────────────
function renderApi() {
  const origin = location.origin;
  const auth = state.info?.auth ? ' -H "X-API-Key: $KARNA_API_KEY"' : "";
  const routes = [
    ["Transcribe a file and wait for the text", "POST /v1/transcribe?wait=true returns the finished job.",
      `curl${auth} -F file=@speech.wav -F model=karna-base "${origin}/v1/transcribe?wait=true"`],
    ["Queue a file", "Returns the job at once; poll it until status is done.",
      `curl${auth} -F file=@speech.wav ${origin}/v1/transcribe\ncurl${auth} ${origin}/v1/jobs/<id>`],
    ["Download the text", "format=txt or json.",
      `curl${auth} -OJ "${origin}/v1/jobs/<id>/export?format=txt"`],
    ["List and delete jobs", "", `curl${auth} ${origin}/v1/jobs\ncurl${auth} -X DELETE ${origin}/v1/jobs/<id>`],
    ["Server and models", "Version, device, whether the models are loaded, and the limits.", `curl ${origin}/v1/info`],
  ];
  const root = $("#api-examples");
  root.replaceChildren(...routes.map(([title, text, code]) => {
    const section = Object.assign(document.createElement("section"), { className: "route" });
    section.append(Object.assign(document.createElement("h2"), { textContent: title }));
    if (text) section.append(Object.assign(document.createElement("p"), { textContent: text }));
    const wrap = Object.assign(document.createElement("div"), { className: "code-wrap" });
    wrap.append(Object.assign(document.createElement("pre"), { className: "code", textContent: code }));
    wrap.append(button("Copy", () => navigator.clipboard.writeText(code).then(() => toast("Copied"))));
    section.append(wrap);
    return section;
  }));
}

// ── routing ────────────────────────────────────────────────────────────
function route() {
  const view = location.hash === "#/api" ? "api" : "home";
  for (const el of document.querySelectorAll("[data-view]")) el.hidden = el.dataset.view !== view;
  for (const a of document.querySelectorAll("[data-tab]")) {
    if (a.dataset.tab === view) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  }
}
addEventListener("hashchange", route);
route();
refreshInfo();
refreshJobs();
