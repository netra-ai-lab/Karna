"""The ``karna`` command: transcribe, models, download, serve."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

from .models import DEFAULT_MODEL, MODELS


def _transcribe(args: argparse.Namespace) -> None:
    from .asr import load_model

    from .asr import SINGLE_PASS_S
    from .audio import load_audio

    asr = load_model(args.model, args.device)
    if args.lm:
        from .models import lm_settings

        os.environ["KARNA_LM"] = args.lm  # so lm_settings reads this LM's tuned settings
        asr.set_lm(args.lm, **lm_settings(asr.name)[1])
    wavs = [load_audio(path, asr.sample_rate) for path in args.audio]
    short = [i for i, w in enumerate(wavs) if len(w) / asr.sample_rate <= SINGLE_PASS_S]
    texts: list = [None] * len(wavs)
    segments: dict[int, list] = {}
    for i, text in zip(short, asr.transcribe([wavs[i] for i in short], batch_size=args.batch_size,
                                             language=args.language, decoder=args.decoder)):
        texts[i] = text
    for i, w in enumerate(wavs):  # longer recordings: split at pauses
        if texts[i] is None:
            long = asr.transcribe_long(w, language=args.language, decoder=args.decoder, batch_size=args.batch_size)
            texts[i], segments[i] = long["text"], long["segments"]
    if args.format == "json":
        rows = [{"file": path, "text": text, **({"segments": segments[i]} if i in segments else {})}
                for i, (path, text) in enumerate(zip(args.audio, texts))]
        print(json.dumps({"model": asr.name, "device": asr.device, "results": rows}, ensure_ascii=False, indent=2))
        return
    for path, text in zip(args.audio, texts):
        print(text if args.format == "txt" else f"{path}\t{text}")
        sys.stdout.flush()


def _models(args: argparse.Namespace) -> None:
    for name, m in MODELS.items():
        print(f"{name}\t{m['params']}\thttps://huggingface.co/{m['repo']}\t{m['about']}")


def _download(args: argparse.Namespace) -> None:
    from .inference.hub import fetch
    from .models import source

    for name in args.models or [DEFAULT_MODEL]:
        print(f"{name}\t{fetch(source(name), args.device)}")


def _serve(args: argparse.Namespace) -> None:
    missing = [m for m in ("fastapi", "uvicorn", "multipart") if importlib.util.find_spec(m) is None]
    if missing:
        sys.exit(f"`karna serve` needs the server extra (missing: {', '.join(missing)}):\n"
                 "  pip install 'netra-karna[cpu,server]'")
    from .server import run

    run(host=args.host, port=args.port, model=args.model, device=args.device)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="karna", description="Khmer and English speech recognition.")
    ap.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    sub = ap.add_subparsers(dest="command", required=True)

    t = sub.add_parser("transcribe", help="transcribe audio files (over 30 s: split at pauses)")
    t.add_argument("audio", nargs="+", help="audio files (wav, flac, ogg, mp3, ...)")
    t.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"{' or '.join(MODELS)}, a Hugging Face id or a model folder (default {DEFAULT_MODEL})")
    t.add_argument("--device", help="auto, cpu, cuda, tensorrt or openvino (default: $KARNA_DEVICE, else auto)")
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--language", choices=["auto", "km", "en"], default="auto",
                   help="auto: Khmer or English; km: never output Latin letters; en: English only")
    t.add_argument("--decoder", choices=["auto", "greedy", "beam"], default="auto",
                   help="auto (default): beam search when a language model is set (--lm or $KARNA_LM), "
                        "else greedy; greedy: the per-frame argmax; beam: CTC beam search")
    t.add_argument("--lm", help="character language model (.npz) for beam search (default: $KARNA_LM)")
    t.add_argument("--format", choices=["tsv", "txt", "json"], default="tsv",
                   help="tsv: <path><TAB><text> per file (default); txt: the text only; json: one document")
    t.set_defaults(func=_transcribe)

    sub.add_parser("models", help="list the released models").set_defaults(func=_models)

    d = sub.add_parser("download", help="download models now, for offline use later")
    d.add_argument("models", nargs="*", help=f"default: {DEFAULT_MODEL}")
    d.add_argument("--device", default="cpu", help="fetch only the files this device needs (default cpu)")
    d.set_defaults(func=_download)

    s = sub.add_parser("serve", help="start the web app and REST API (needs the [server] extra)")
    s.add_argument("--host", help="default: $KARNA_HOST, else 127.0.0.1")
    s.add_argument("--port", type=int, help="default: $KARNA_PORT, else 8000")
    s.add_argument("--model", help="model loaded at start (default: $KARNA_MODEL, else karna-base)")
    s.add_argument("--device", help="default: $KARNA_DEVICE, else auto")
    s.set_defaults(func=_serve)

    args = ap.parse_args(argv)
    args.func(args)


def _version() -> str:
    from . import __version__

    return __version__


if __name__ == "__main__":
    main()
