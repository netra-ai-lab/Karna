// Live transcription: microphone -> 16 kHz mono int16 PCM -> WebSocket /v1/stream.
// The server answers with "ready", then "partial" and "final" text, then "done" with the saved job.

const RATE = 16000;
const SEND_SAMPLES = RATE / 10; // about 100 ms per frame

// Box-filter decimation from the AudioContext rate (often 48 or 44.1 kHz) to 16 kHz: enough for speech.
class Downsampler {
  constructor(inRate) { this.ratio = inRate / RATE; this.sum = 0; this.count = 0; this.t = 0; }
  push(input) {
    if (this.ratio === 1) return input;
    const out = new Float32Array(Math.ceil(input.length / this.ratio) + 1);
    let n = 0;
    for (let i = 0; i < input.length; i++) {
      this.sum += input[i]; this.count++; this.t += 1;
      if (this.t >= this.ratio) {
        out[n++] = this.sum / this.count;
        this.sum = 0; this.count = 0; this.t -= this.ratio;
      }
    }
    return out.subarray(0, n);
  }
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
export async function streamMic({ model, apiKey = "", maxSeconds = 600, onMessage = () => {}, onTick = () => {} } = {}) {
  const media = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  const ctx = new AudioContext();
  try {
    await ctx.audioWorklet.addModule("/static/js/pcm-worklet.js");
  } catch (e) {
    media.getTracks().forEach((t) => t.stop());
    ctx.close();
    throw e;
  }
  const source = ctx.createMediaStreamSource(media);
  const node = new AudioWorkletNode(ctx, "pcm-capture");
  const mute = ctx.createGain();
  mute.gain.value = 0;
  source.connect(node).connect(mute).connect(ctx.destination); // keeps the worklet pulled

  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/v1/stream`);
  ws.binaryType = "arraybuffer";
  const down = new Downsampler(ctx.sampleRate);
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
    ws.onopen = () => ws.send(JSON.stringify({ type: "start", model, api_key: apiKey }));
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
