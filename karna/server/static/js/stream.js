// Live transcription: microphone -> 16 kHz mono int16 PCM -> WebSocket /v1/stream.
// The server answers with "ready", then "partial" and "final" text, then "done" with the saved job.

const RATE = 16000;
const SEND_SAMPLES = RATE / 10; // about 100 ms per frame

// Streaming windowed-sinc resampler (Blackman window, 7 kHz cutoff), for when the browser can't
// run the AudioContext at 16 kHz itself (Firefox). Any ratio, so 44.1 kHz works too. A plain
// box filter here let 8-16 kHz noise alias into the speech band at about -6 dB.
export class Resampler {
  constructor(inRate, outRate = RATE) {
    this.step = inRate / outRate;              // input samples per output sample
    this.passthrough = inRate === outRate;
    const fc = 7000 / inRate;                  // cutoff, cycles per input sample
    this.W = Math.ceil(24 * this.step);        // half-width of the kernel, input samples
    this.R = 64;                               // table points per input sample
    const n = 2 * this.W * this.R + 1;
    this.table = new Float32Array(n + 1);
    for (let i = 0; i < n; i++) {
      const x = i / this.R - this.W;
      const sinc = x === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * x) / (Math.PI * x);
      const u = i / (n - 1);
      this.table[i] = sinc * (0.42 - 0.5 * Math.cos(2 * Math.PI * u) + 0.08 * Math.cos(4 * Math.PI * u));
    }
    this.buf = new Float32Array(this.W);       // history, starting as silence
    this.t = this.W;                           // next output's centre, in buf coordinates
  }

  push(input) {
    if (this.passthrough) return input;
    const { W, R, table, step } = this;
    const buf = new Float32Array(this.buf.length + input.length);
    buf.set(this.buf);
    buf.set(input, this.buf.length);
    const out = new Float32Array(Math.max(0, Math.ceil((buf.length - W - this.t) / step) + 1));
    let t = this.t, n = 0;
    for (let c = Math.floor(t); c + W < buf.length; c = Math.floor(t)) {
      let acc = 0;
      for (let j = c - W + 1; j <= c + W; j++) {
        const p = (j - t + W) * R, i = p | 0;
        acc += buf[j] * (table[i] + (table[i + 1] - table[i]) * (p - i));
      }
      out[n++] = acc;
      t += step;
    }
    const keep = Math.max(0, Math.floor(t) - W);
    this.buf = buf.slice(keep);
    this.t = t - keep;
    return out.subarray(0, n);
  }
}

function rms(samples) {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  return Math.sqrt(sum / Math.max(1, samples.length));
}

// A 16 kHz context lets the browser resample properly (as OfflineAudioContext does for Record).
// Firefox refuses to connect a microphone to a context at another rate: fall back to the native
// rate and the Resampler above.
function openContext(media) {
  try {
    const ctx = new AudioContext({ sampleRate: RATE });
    try { return { ctx, source: ctx.createMediaStreamSource(media) }; } catch { ctx.close(); }
  } catch { /* sampleRate option unsupported */ }
  const ctx = new AudioContext();
  return { ctx, source: ctx.createMediaStreamSource(media) };
}

function toInt16(samples) {
  const out = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

// Returns { stop, done }: stop() ends the session; done resolves with the server's "done" message.
// onLevel(rms) gets the microphone level about 20 times a second, for a meter.
export async function streamMic({ model, language = "auto", apiKey = "", maxSeconds = 600, onMessage = () => {},
                                  onTick = () => {}, onLevel = () => {}, noiseSuppression = true } = {}) {
  // Automatic gain is off: it changes the gain within an utterance, and the model normalises
  // each utterance itself. Noise suppression is the user's choice (the "Microphone" setting).
  const media = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression, autoGainControl: false },
  });
  let ctx, source;
  try {
    ({ ctx, source } = openContext(media));
    await ctx.audioWorklet.addModule("/static/js/pcm-worklet.js");
  } catch (e) {
    media.getTracks().forEach((t) => t.stop());
    ctx?.close();
    throw e;
  }
  const node = new AudioWorkletNode(ctx, "pcm-capture");
  const mute = ctx.createGain();
  mute.gain.value = 0;
  source.connect(node).connect(mute).connect(ctx.destination); // keeps the worklet pulled

  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/v1/stream`);
  ws.binaryType = "arraybuffer";
  const down = new Resampler(ctx.sampleRate);
  let pending = [], pendingLen = 0, ready = false, stopped = false, finished = false;
  const backlog = []; // frames recorded before the server said "ready"

  function sendFrame(frame) {
    if (ready && ws.readyState === WebSocket.OPEN) ws.send(frame.buffer);
    else backlog.push(frame);
  }
  function flush() {
    if (!pendingLen) return;
    const all = new Float32Array(pendingLen);
    let o = 0;
    for (const p of pending) { all.set(p, o); o += p.length; }
    pending = []; pendingLen = 0;
    sendFrame(toInt16(all));
  }
  node.port.onmessage = (e) => {
    if (stopped) return;
    onLevel(rms(e.data));
    const samples = down.push(e.data);
    pending.push(samples); pendingLen += samples.length;
    if (pendingLen >= SEND_SAMPLES) flush();
  };

  const started = performance.now();
  const timer = setInterval(() => {
    const s = (performance.now() - started) / 1000;
    onTick(s);
    if (s >= maxSeconds) stop();
  }, 250);

  function release() {
    clearInterval(timer);
    media.getTracks().forEach((t) => t.stop());
    node.port.onmessage = null;
    source.disconnect();
    ctx.close();
  }

  function stop() {
    if (stopped) return;
    stopped = true;
    flush();
    release();
    if (ws.readyState === WebSocket.OPEN && ready) ws.send(JSON.stringify({ type: "stop" }));
  }

  const done = new Promise((resolve, reject) => {
    ws.onopen = () => ws.send(JSON.stringify({ type: "start", model, language, api_key: apiKey }));
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.type === "ready") {
        ready = true;
        for (const frame of backlog.splice(0)) ws.send(frame.buffer);
        if (stopped) ws.send(JSON.stringify({ type: "stop" }));
      } else if (msg.type === "done") {
        finished = true;
        resolve(msg);
      } else if (msg.type === "error") {
        finished = true;
        reject(new Error(msg.detail));
      }
      onMessage(msg);
    };
    ws.onclose = (e) => {
      stop();
      if (!finished) {
        const reasons = { 1008: "The API key is missing or wrong.", 1013: "The server is busy with other live sessions; try again shortly." };
        reject(new Error(reasons[e.code] || e.reason || "The live connection closed."));
      }
    };
  });
  done.catch(() => {}).finally(() => { stop(); if (ws.readyState === WebSocket.OPEN) ws.close(); });
  return { stop, done };
}
