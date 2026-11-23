// Karna web app: plain ES module, no build step. Talks to the REST API only.
import { recordWav } from "./record.js";
import { streamMic } from "./stream.js";

const $ = (sel) => document.querySelector(sel);
const state = { info: null, jobs: [], polling: new Set(), recorder: null, live: null, fresh: null };
const LANGUAGE_NAMES = { auto: "Auto", km: "Khmer", en: "English" };

function stored(key, fallback = "") {
  try { return localStorage.getItem(key) || fallback; } catch { return fallback; /* storage blocked */ }
}

function store(key, value) {
  try { localStorage.setItem(key, value); } catch { /* storage blocked */ }
}

function apiKey() { return stored("karna.key"); }

function keyHeader() {
  const key = apiKey();
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

// ── model and language choices (segmented controls) ───────────────────
const model = () => document.querySelector('input[name="model"]:checked')?.value || state.info?.default_model || "karna-base";
const language = () => document.querySelector('input[name="language"]:checked')?.value || "auto";

function lockChoices(locked) {
  for (const input of document.querySelectorAll('input[name="model"], input[name="language"]')) input.disabled = locked;
}

function renderModels(info) {
  const root = $("#model");
  if (root.children.length) return;
  const chosen = info.models.some((m) => m.name === stored("karna.model")) ? stored("karna.model") : info.default_model;
  for (const m of info.models) {
    const label = document.createElement("label");
    label.title = m.about;
    const input = Object.assign(document.createElement("input"), { type: "radio", name: "model", value: m.name, checked: m.name === chosen });
    input.addEventListener("change", () => store("karna.model", m.name));
    const text = document.createElement("span");
    text.append(m.name.replace(/^karna-/, ""), Object.assign(document.createElement("small"), { textContent: m.params }));
    label.append(input, text);
    root.append(label);
  }
}

{
  const saved = stored("karna.language", "auto");
  const input = document.querySelector(`input[name="language"][value="${saved}"]`) || document.querySelector('input[name="language"][value="auto"]');
  input.checked = true;
  for (const el of document.querySelectorAll('input[name="language"]')) el.addEventListener("change", () => store("karna.language", el.value));
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
    renderModels(info);
    $("#drop-meta").textContent = `or drop audio anywhere · up to ${info.max_seconds} s per file`;
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
const opened = new Set();  // jobs whose full text is shown

function renderJobs() {
  const list = $("#jobs");
  list.replaceChildren(...state.jobs.map(renderJob));
  $("#empty").hidden = state.jobs.length > 0;
  // "More" only where the four-line clamp actually cut something
  requestAnimationFrame(() => {
    for (const more of list.querySelectorAll(".job-more")) {
      const text = more.previousElementSibling;
      more.hidden = !text.classList.contains("open") && text.scrollHeight <= text.clientHeight + 2;
    }
  });
}

function ago(seconds) {
  const s = Date.now() / 1000 - seconds;
  if (s < 45) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return new Date(seconds * 1000).toLocaleDateString(undefined, { day: "numeric", month: "short" });
}

function kind(job) {
  if (job.live) return ["live", "Live"];
  if (/^Recording /.test(job.filename)) return ["recording", "Recording"];
  return ["upload", "File"];
}

function el(tag, className, text) {
  return Object.assign(document.createElement(tag), { className, ...(text !== undefined ? { textContent: text } : {}) });
}

function renderJob(job) {
  const li = el("li", job.id === state.fresh ? "job fresh" : "job");
  const head = el("div", "job-head");
  const [k, label] = kind(job);
  head.append(el("span", `badge badge-${k}`, label));
  const when = el("span", "job-time", ago(job.created));
  when.title = new Date(job.created * 1000).toLocaleString();
  head.append(when);
  if (k === "upload") head.append(el("span", "job-name", job.filename));
  if (job.status !== "done") head.append(el("span", `status status-${job.status}`, STATUS[job.status] || job.status));
  const bits = [job.model.replace(/^karna-/, ""), job.language && job.language !== "auto" ? LANGUAGE_NAMES[job.language] : null,
    `${job.duration_s.toFixed(1)} s`, job.result ? `took ${job.result.seconds.toFixed(1)} s` : null].filter(Boolean);
  head.append(el("span", "meta", bits.join(" · ")));
  const actions = el("span", "job-actions");
  if (job.status === "done") {
    actions.append(button("Copy", () => navigator.clipboard.writeText(job.result.text).then(() => toast("Copied"))));
    actions.append(button("Download", () => download(job)));
  }
  actions.append(deleteButton(job));
  head.append(actions);
  li.append(head);
  if (job.status === "done") {
    const text = el("p", opened.has(job.id) ? "job-text open" : "job-text", job.result.text || "(no speech recognised)");
    li.append(text);
    if ((job.result.text || "").length > 120) {
      const more = button(opened.has(job.id) ? "Less" : "More", () => {
        if (opened.has(job.id)) opened.delete(job.id); else opened.add(job.id);
        text.classList.toggle("open");
        more.textContent = text.classList.contains("open") ? "Less" : "More";
      });
      more.classList.add("job-more");
      li.append(more);
    }
  } else if (job.status === "failed") {
    li.append(el("p", "job-error", job.error));
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

// relative times stay fresh
setInterval(() => { if (state.jobs.length && !document.hidden) renderJobs(); }, 60000);

// ── upload ─────────────────────────────────────────────────────────────
function showError(text) { $("#upload-error").textContent = text || ""; }

async function upload(blob, filename) {
  showError("");
  const form = new FormData();
  form.append("file", blob, filename);
  form.append("model", model());
  form.append("language", language());
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
  store("karna.key", e.target.value);
  refreshJobs();
});

// ── recording ──────────────────────────────────────────────────────────
$("#record").addEventListener("click", async () => {
  const btn = $("#record"), label = $("#record-label");
  if (state.recorder) { state.recorder.stop(); return; }
  if (state.live) return;
  showError("");
  $("#live-btn").disabled = true;
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
    $("#live-btn").disabled = false;
  }
});

// ── live captions ──────────────────────────────────────────────────────
// Each utterance (segment) is a block of word spans in one of three states: "draft" (partials, may
// still change), "finishing" (a newer utterance has started, so this one is closed and its final
// decode is running) and "final" (never changes).
//
// An utterance can run 15 s before its final, so drafts settle as they go, from the front only: the
// white text is locked and only grows, and grey is just the unsettled end. Each draft is aligned
// with the locked text (`resume`) to find where it ends; after that, words the previous draft also
// had (aligned, not just a common prefix) are locked, up to the first that differs (unless the
// SETTLE_AHEAD words after it agree) and short of the newest SETTLE_TAIL words. Later drafts never
// change locked words; the final replaces the whole line. Replaying a real-time karna-large session
// of a 50 s Khmer recording (drafts every 2-3 s): 29 % of the open line white, the grey end a median
// 35 characters (about the last 4 s of speech), and the final changed 3.3 % of the white characters.
// (Settling any agreed word wherever it was mixed grey and white words in the middle of the line,
// which was hard to read.) Words found in both the old and new
// text keep their element, so only the changes pop in.
//
// The text sits at the bottom of a few-line window: when it grows a line, the block glides up
// (FLIP), the oldest line fades out under the top mask, and segments that have left the window are
// dropped from the page. The whole text is kept in `segs` for Copy and "Show all".
const wordSplitter = typeof Intl.Segmenter === "function" ? new Intl.Segmenter("km", { granularity: "word" }) : null;
const MAX_STAGGER = 12;  // words; keeps a long correction from trickling in for seconds
const SETTLE_TAIL = 2;   // the newest words of a draft are never white: the audio may end mid-word
const SETTLE_AHEAD = 3;  // agreed words after a changed one that settle it anyway

// For each piece of `b`, the index of the same piece in `a` (longest common subsequence), or -1.
function align(a, b) {
  const n = a.length, m = b.length, L = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) {
    L[i][j] = a[i] === b[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
  }
  const out = new Array(m).fill(-1);
  for (let i = 0, j = 0; i < n && j < m;) {
    if (a[i] === b[j]) { out[j] = i; i++; j++; } else if (L[i + 1][j] >= L[i][j + 1]) i++; else j++;
  }
  return out;
}

function pieces(text) {
  if (wordSplitter) return Array.from(wordSplitter.segment(text), (s) => s.segment);
  return text.split(/(\s+)/).filter(Boolean);
}

// Where the locked words end in a new draft: the index of the first piece of `next` that follows
// them. Aligned by character, because a draft may split the same text into words differently.
// Locked pieces at the end that the draft no longer has, or that end inside one of its words, are
// unlocked (removed from `locked`) so nothing is shown twice.
function resume(locked, next) {
  if (!locked.length) return 0;
  const a = locked.join(""), at = align(next.join("").split(""), a.split(""));  // per locked char
  const spans = [];   // per locked piece: [first, last] matched draft char, or null
  let off = 0;
  for (const p of locked) {
    let lo = -1, hi = -1;
    for (let c = off; c < off + p.length; c++) if (at[c] >= 0) { if (lo < 0) lo = at[c]; hi = at[c]; }
    spans.push(lo < 0 ? null : [lo, hi]);
    off += p.length;
  }
  const starts = [];  // draft piece start offsets
  off = 0;
  for (const p of next) { starts.push(off); off += p.length; }
  starts.push(off);
  while (locked.length) {
    const span = spans[locked.length - 1];
    if (!span) { locked.pop(); continue; }            // gone from the draft
    const cut = span[1] + 1;
    const s = starts.findLastIndex((o) => o <= cut);
    if (starts[s] === cut) return s;                 // ends on a piece boundary
    locked.pop();                                    // ends inside a draft word: let that word be grey
    while (locked.length && spans[locked.length - 1] && spans[locked.length - 1][1] >= starts[s]) locked.pop();
    if (!locked.length || !spans[locked.length - 1]) continue;
    return starts.findLastIndex((o) => o <= spans[locked.length - 1][1] + 1) === s ? s : resume(locked, next);
  }
  return 0;
}

class Captions {
  constructor(view, el) { this.view = view; this.el = el; this.segs = new Map(); this.expanded = false; }

  clear() {
    this.el.replaceChildren();
    this.segs.clear();
    this.setExpanded(false);
  }

  get text() {
    return [...this.segs.entries()].sort((a, b) => a[0] - b[0])
      .filter(([, s]) => s.final).map(([, s]) => s.pieces.join("")).filter(Boolean).join(" ");
  }

  get empty() { return this.segs.size === 0; }

  // Stop was pressed: every open draft is now being finalised.
  finishing() {
    for (const seg of this.segs.values()) if (!seg.final && seg.el) seg.el.className = "seg finishing";
  }

  update(id, text, final) {
    let seg = this.segs.get(id);
    if (seg?.final) return;                      // finals never change; ignore a late draft
    if (!seg) {
      if (!text) return;
      seg = { el: null, pieces: [], locked: [], shown: [], nodes: [], final: false };
      this.segs.set(id, seg);
      for (const [other, s] of this.segs) {      // the speaker moved on: older drafts are being finalised
        if (other < id && !s.final && s.el) s.el.className = "seg finishing";
      }
    }
    const before = this.el.offsetHeight;
    if (final && !text) {                        // too little speech after all: fade the draft out
      this.segs.delete(id);
      if (seg.el) {
        const gone = seg.el;
        gone.classList.add("leaving");
        gone.addEventListener("animationend", () => gone.remove(), { once: true });
      }
      return;
    }
    if (!seg.el) {
      seg.el = el("span", "seg");
      seg.el.dataset.id = id;
      const after = [...this.el.children].find((c) => Number(c.dataset.id) > id);
      this.el.insertBefore(seg.el, after || null);
    }
    const next = pieces(text);
    let shown = next, settled = next.length;
    if (!final) {
      const { locked } = seg;
      const start = resume(locked, next);
      // then lock what the previous draft agrees on, up to the first change or the newest words
      const agreed = align(seg.pieces, next);
      const words = next.reduce((k, p, j) => (p.trim() ? k.concat(j) : k), []);
      const stop = words.length > SETTLE_TAIL ? words[words.length - SETTLE_TAIL] : 0;
      // a word that keeps changing doesn't hold back the rest: if the SETTLE_AHEAD words after it
      // agree, it is locked as this draft spells it (the final can still correct it)
      const ahead = (k) => {
        let w = 0;
        for (let j = k + 1; j < stop && w < SETTLE_AHEAD; j++) {
          if (!next[j].trim()) continue;
          if (agreed[j] < 0) return false;
          w++;
        }
        return w === SETTLE_AHEAD;
      };
      let k = start;
      while (seg.pieces.length && k < stop && (agreed[k] >= 0 || ahead(k))) locked.push(next[k++]);
      shown = locked.concat(next.slice(k));
      settled = locked.length;
    }
    const from = align(seg.shown, shown);
    let n = 0;
    const nodes = shown.map((p, j) => {
      const old = (from[j] >= 0 && seg.nodes[from[j]]) || null;
      if (!p.trim()) return old || document.createTextNode(p);   // spaces stay text, so lines wrap there
      const w = old || el("span", "w pop", p);
      if (!old) {
        w.style.setProperty("--i", Math.min(n++, MAX_STAGGER));
        w.addEventListener("animationend", () => w.classList.remove("pop"), { once: true });  // not again when moved
      }
      w.classList.toggle("stable", j < settled);
      return w;
    });
    seg.el.replaceChildren(...nodes);
    seg.pieces = next;
    seg.shown = shown;
    seg.nodes = nodes;
    seg.final = final;
    seg.el.className = `seg ${final ? "final" : "draft"}`;
    if (this.expanded) { this.view.scrollTop = this.view.scrollHeight; return; }
    this.glide(this.el.offsetHeight - before);
    this.prune();
  }

  // The text is anchored to the bottom: when it grows by `dy`, start it `dy` lower and slide it up.
  glide(dy) {
    if (dy <= 0 || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const s = this.el.style;
    s.transition = "none";
    s.transform = `translateY(${dy}px)`;
    void this.el.offsetHeight;                   // apply the start position before transitioning
    s.transition = "transform .38s cubic-bezier(.2, .7, .2, 1)";
    s.transform = "";
  }

  // Remove segments that are wholly above the window (finals only; the full text stays in `segs`).
  prune() {
    const top = this.el.offsetHeight - this.view.clientHeight;
    if (top <= 0) return;
    const box = this.el.getBoundingClientRect();
    for (const seg of this.segs.values()) {
      if (!seg.el || !seg.final) continue;
      const rects = seg.el.getClientRects();
      const bottom = rects.length ? rects[rects.length - 1].bottom - box.top : 0;
      if (bottom < top - 4) { seg.el.remove(); seg.el = null; } else break;
    }
  }

  // "Show all": every final at once, scrollable; back to the rolling window when a new session starts.
  setExpanded(on) {
    this.expanded = on;
    this.view.classList.toggle("expanded", on);
    this.el.style.transform = "";
    if (!on) return;
    this.el.replaceChildren();
    for (const [id, seg] of [...this.segs.entries()].sort((a, b) => a[0] - b[0])) {
      seg.el = el("span", `seg ${seg.final ? "final" : "draft"}`, seg.shown.join(""));
      seg.nodes = [];                            // plain text now: a late draft or final starts afresh
      seg.el.dataset.id = id;
      this.el.append(seg.el);
    }
    this.view.scrollTop = this.view.scrollHeight;
  }
}

const captions = new Captions($("#captions"), $("#live-text"));
function showDrafts(on) {
  $("#captions").classList.toggle("settled-only", !on);
  $("#live").classList.toggle("settled-only", !on);
  $("#live-drafts").setAttribute("aria-pressed", String(!on));
}
showDrafts(stored("karna.drafts", "show") !== "hide");
$("#live-drafts").addEventListener("click", () => {
  const on = $("#live-drafts").getAttribute("aria-pressed") === "true";   // was hiding: show again
  showDrafts(on);
  store("karna.drafts", on ? "show" : "hide");
});
$("#live-copy").addEventListener("click", () => navigator.clipboard.writeText(captions.text).then(() => toast("Copied")));
$("#live-all").addEventListener("click", () => {
  captions.setExpanded(!captions.expanded);
  $("#live-all").textContent = captions.expanded ? "Show less" : "Show all";
});

// Seven bars, tallest in the middle, plus a line along the bottom of the stage, following the
// microphone level (smoothed per animation frame).
const meter = {
  bars: [...document.querySelectorAll("#live-meter i")], shape: [.45, .7, .9, 1, .9, .7, .45],
  target: 0, level: 0, frame: 0,
  set(rms) { this.target = Math.max(0, Math.min(1, (20 * Math.log10(rms + 1e-9) + 58) / 46)); },
  start() {
    const step = (t) => {
      this.level += (this.target - this.level) * (this.target > this.level ? .55 : .14);
      this.bars.forEach((b, i) => {
        const wobble = .82 + .18 * Math.sin(t / 90 + i * 1.7);
        b.style.setProperty("--h", (.12 + .88 * this.level * this.shape[i] * wobble).toFixed(3));
      });
      $("#live").style.setProperty("--level", this.level.toFixed(3));
      this.frame = requestAnimationFrame(step);
    };
    this.frame = requestAnimationFrame(step);
  },
  stop() { cancelAnimationFrame(this.frame); this.target = this.level = 0; $("#live").style.setProperty("--level", "0"); },
};

function clock(s) { return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`; }

function setLive(stateName, text) {
  $("#live").dataset.state = stateName;
  $("#live-status").textContent = text;
}

function markFresh(id) {
  state.fresh = id;
  clearTimeout(markFresh.timer);
  markFresh.timer = setTimeout(() => { state.fresh = null; }, 2800);
}

$("#live-btn").addEventListener("click", async () => {
  const btn = $("#live-btn"), label = $("#live-label");
  if (state.live) { state.live.stop(); captions.finishing(); setLive("finishing", "Finishing…"); label.textContent = "Finishing…"; btn.disabled = true; return; }
  if (state.recorder) return;
  showError("");
  const chosen = model(), lang = language();
  captions.clear();
  setLive("connecting", "Connecting…");
  $("#live-time").textContent = "0:00";
  $("#live-model").textContent = chosen;
  $("#live-language").textContent = lang === "auto" ? "" : LANGUAGE_NAMES[lang];
  $("#live-copy").hidden = true;
  $("#live-all").hidden = true;
  $("#live-all").textContent = "Show all";
  $("#live-empty").hidden = false;
  $("#live").hidden = false;
  $("#record").disabled = true;
  lockChoices(true);
  try {
    state.live = await streamMic({
      model: chosen,
      language: lang,
      apiKey: apiKey(),
      maxSeconds: state.info?.max_stream_seconds || 600,
      onTick: (s) => { $("#live-time").textContent = clock(s); },
      onLevel: (rms) => meter.set(rms),
      onMessage: (msg) => {
        if (msg.type === "ready") {
          setLive("listening", "Listening");
          // karna-large: drafts come from karna-base, and karna-large's finals replace them
          if (msg.partial_model && msg.partial_model !== msg.model) {
            $("#live-model").textContent = `${msg.model} · drafts ${msg.partial_model.replace(/^karna-/, "")}`;
          }
        } else if (msg.type === "partial") captions.update(msg.segment, msg.text, false);
        else if (msg.type === "final") captions.update(msg.segment, msg.text, true);
        $("#live-empty").hidden = !captions.empty;
      },
    });
    meter.start();
    btn.setAttribute("aria-pressed", "true");
    label.textContent = "Stop";
    const done = await state.live.done;
    if (done.job) {
      markFresh(done.job.id);
      state.jobs = [done.job, ...state.jobs.filter((j) => j.id !== done.job.id)];
      renderJobs();
      setLive("saved", done.reason === "limit" ? `Stopped at the ${state.info?.max_stream_seconds || 600} s limit · saved to Recent` : "Saved to Recent");
      $("#live-copy").hidden = false;
      $("#live-all").hidden = false;
    } else {
      setLive("idle", "No speech heard");
    }
  } catch (e) {
    showError(e.name === "NotAllowedError" ? "Microphone access was refused." : e.message);
    setLive("error", "Stopped");
  } finally {
    meter.stop();
    $("#live-empty").hidden = true;
    state.live = null;
    btn.disabled = false;
    btn.setAttribute("aria-pressed", "false");
    label.textContent = "Start live captions";
    $("#record").disabled = false;
    lockChoices(false);
  }
});

// ── API tab ────────────────────────────────────────────────────────────
function renderApi() {
  const origin = location.origin;
  const auth = state.info?.auth ? ' -H "X-API-Key: $KARNA_API_KEY"' : "";
  const routes = [
    ["Transcribe a file and wait for the text", "POST /v1/transcribe?wait=true returns the finished job. language is auto, km (no Latin letters) or en.",
      `curl${auth} -F file=@speech.wav -F model=karna-large -F language=km "${origin}/v1/transcribe?wait=true"`],
    ["Queue a file", "Returns the job at once; poll it until status is done.",
      `curl${auth} -F file=@speech.wav ${origin}/v1/transcribe\ncurl${auth} ${origin}/v1/jobs/<id>`],
    ["Download the text", "format=txt or json.",
      `curl${auth} -OJ "${origin}/v1/jobs/<id>/export?format=txt"`],
    ["List and delete jobs", "", `curl${auth} ${origin}/v1/jobs\ncurl${auth} -X DELETE ${origin}/v1/jobs/<id>`],
    ["Server and models", "Version, device, whether the models are loaded, and the limits.", `curl ${origin}/v1/info`],
    ["Live transcription (WebSocket)",
      "Send a start message, then binary frames of 16 kHz mono int16 PCM, then a stop message. " +
      "You get partial text (may still change), final text per utterance, and a done message with the saved job.",
      `ws = new WebSocket("${origin.replace(/^http/, "ws")}/v1/stream")\n` +
      `ws.send(JSON.stringify({type: "start", model: "karna-large", language: "km"${state.info?.auth ? ', api_key: KEY' : ""}}))\n` +
      `ws.send(int16Pcm.buffer)            // repeat, about 100 ms each\n` +
      `ws.send(JSON.stringify({type: "stop"}))\n` +
      `// <- {"type":"partial","segment":0,"text":"..."}  {"type":"final",...}  {"type":"done","job":{...}}`],
  ];
  const root = $("#api-examples");
  root.replaceChildren(...routes.map(([title, text, code]) => {
    const section = el("section", "route");
    section.append(el("h2", "", title));
    if (text) section.append(el("p", "", text));
    const wrap = el("div", "code-wrap");
    wrap.append(el("pre", "code", code));
    wrap.append(button("Copy", () => navigator.clipboard.writeText(code).then(() => toast("Copied"))));
    section.append(wrap);
    return section;
  }));
}

// ── routing ────────────────────────────────────────────────────────────
function route() {
  const view = location.hash === "#/api" ? "api" : "home";
  for (const v of document.querySelectorAll("[data-view]")) v.hidden = v.dataset.view !== view;
  for (const a of document.querySelectorAll("[data-tab]")) {
    if (a.dataset.tab === view) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  }
}
addEventListener("hashchange", route);
route();
refreshInfo();
refreshJobs();
