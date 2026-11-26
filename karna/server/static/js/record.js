// Microphone -> 16 kHz mono 16-bit WAV, in the browser, so the server needs no ffmpeg:
// MediaRecorder (webm/opus or mp4), then decodeAudioData, then OfflineAudioContext resampling.

const RATE = 16000;

export async function recordWav({ maxSeconds = 30, onTick = () => {}, noiseSuppression = true } = {}) {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression } });
  const recorder = new MediaRecorder(stream);
  const chunks = [];
  recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  const started = performance.now();
  const timer = setInterval(() => {
    const s = (performance.now() - started) / 1000;
    onTick(s);
    if (s >= maxSeconds) stop();
  }, 250);
  const stopped = new Promise((resolve) => { recorder.onstop = resolve; });
  function stop() {
    clearInterval(timer);
    if (recorder.state !== "inactive") recorder.stop();
    stream.getTracks().forEach((t) => t.stop());
  }
  recorder.start(250);
  const done = stopped.then(async () => {
    const blob = new Blob(chunks, { type: recorder.mimeType });
    const ctx = new AudioContext();
    const decoded = await ctx.decodeAudioData(await blob.arrayBuffer());
    ctx.close();
    const length = Math.max(1, Math.ceil(decoded.duration * RATE));
    const offline = new OfflineAudioContext(1, length, RATE);
    const src = offline.createBufferSource();
    src.buffer = decoded;
    src.connect(offline.destination);
    src.start();
    const rendered = await offline.startRendering();
    return encodeWav(rendered.getChannelData(0), RATE);
  });
  return { stop, done };
}

export function encodeWav(samples, rate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const text = (offset, s) => { for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i)); };
  text(0, "RIFF"); view.setUint32(4, 36 + samples.length * 2, true); text(8, "WAVE");
  text(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, rate, true); view.setUint32(28, rate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  text(36, "data"); view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}
