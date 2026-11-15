"""One Kaggle script file per run that needs gliner2: the code joined, with the run's arguments.

    python kaggle/encoder/make_kernel.py            # every kernel in KERNELS -> kaggle/push/<folder>/<file>

A script kernel is a single file, so the modules are joined: one ``from __future__`` line first, then a
pip install of gliner2 that keeps the image's torch (a newer torch from pip would not match its CUDA),
the run's arguments, then the code. In the GLiNER2 kernel the encoder module's own main is switched off and
the GLiNER2 code's ``from nts_encoder import`` finds nothing to import and uses the names defined above it.

* ``train-gliner2-v2``: GLiNER2 fine-tuned on encoder-intents-v2 (v1 + more withdraw wordings);
* ``train-encoder-v2``: the DeBERTa reader (our heads) started from GLiNER2's encoder and its trained
  ``[P]``/``[L]``/``[E]`` tokens, on encoder-intents-v2.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
PUSH = HERE.parent / "push"

PRELUDE = '''from __future__ import annotations

import os
import subprocess
import sys

try:
    import gliner2  # noqa: F401
except ImportError:
    import torch as _torch
    pin = f"torch=={_torch.__version__.split('+')[0]}"
    open("/tmp/constraints.txt", "w").write(pin + "\\n")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gliner2", "-c", "/tmp/constraints.txt"], check=False)
    import importlib.metadata as _md
    print({p: _md.version(p) for p in ("gliner2", "torch", "transformers")}, flush=True)
'''

# folder -> (file, modules, environment, arguments)
KERNELS = {
    "train-gliner2-v2": ("nts_gliner2_kernel.py", ["nts_encoder.py", "nts_gliner2.py"], {"NTS_NO_MAIN": "1"}, []),
    "train-encoder-v2": ("nts_encoder_kernel.py", ["nts_encoder.py"], {},
                         ["--model", "gliner2:fastino/gliner2-base-v1"]),
}


def body(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(x for x in lines if x.strip() != "from __future__ import annotations")


def build(folder: str) -> Path:
    file, modules, env, argv = KERNELS[folder]
    text = PRELUDE
    text += "".join(f"os.environ[{k!r}] = {v!r}\n" for k, v in env.items())
    if argv:
        text += f"sys.argv += {argv!r}\n"
    for m in modules:
        text += f"\n\n# ===== {m} =====\n" + body(HERE / m)
    text += "\n"
    out = PUSH / folder / file
    compile(text, str(out), "exec")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(out, len(text.splitlines()), "lines")
    return out


def main() -> None:
    for folder in KERNELS:
        build(folder)


if __name__ == "__main__":
    main()
