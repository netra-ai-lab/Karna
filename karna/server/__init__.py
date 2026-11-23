"""The Karna web app and REST API (``karna serve``; needs the ``[server]`` extra).

    app.py        routes (create_app)
    jobs.py       jobs as directories, and the one worker thread that owns the models
    settings.py   KARNA_* environment settings
    static/       the browser UI: plain ES modules, no build step
"""

from __future__ import annotations


def run(host: str | None = None, port: int | None = None, model: str | None = None, device: str | None = None) -> None:
    import uvicorn

    from .app import create_app
    from .settings import Settings

    settings = Settings()
    for name, value in (("host", host), ("port", port), ("model", model), ("device", device)):
        if value is not None:
            setattr(settings, name, value)
    print(f"Karna on http://{settings.host}:{settings.port}  (API docs: /docs, jobs in {settings.data_dir})")
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="warning")
