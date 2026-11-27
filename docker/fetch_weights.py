"""Download Karna's ONNX model folders into /models (KARNA_BUNDLES), one subfolder per model,
and the language model into /models/karna-lm.

Run at image build time so the container works offline. Only the files the CPU runs are
fetched, at the revisions pinned in karna/models.py.

    python docker/fetch_weights.py /models [karna-base karna-large]
"""

import shutil
import sys
from pathlib import Path

from karna.inference import hub
from karna.models import DEFAULT_LM, LMS, MODELS, lm_path, source

target = Path(sys.argv[1] if len(sys.argv) > 1 else "/models")
for name in sys.argv[2:] or list(MODELS):
    # Copied out of the Hugging Face cache, whose files are symlinks to blobs.
    shutil.copytree(hub.fetch(source(name), "cpu"), target / name, dirs_exist_ok=True)
    print(target / name)
lm = lm_path(DEFAULT_LM)
(target / DEFAULT_LM).mkdir(parents=True, exist_ok=True)
for file in (lm, lm.with_suffix(".json")):
    shutil.copyfile(file, target / DEFAULT_LM / file.name)  # resolves the cache's symlinks
print(target / DEFAULT_LM / LMS[DEFAULT_LM]["file"])
