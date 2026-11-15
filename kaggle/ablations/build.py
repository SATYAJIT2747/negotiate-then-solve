"""Build the Kaggle push folders for the paired ablation rerun (ours-llm, A1, A2, A3 x 3 seeds, 30 scenarios).

Two accounts, two GPU sessions each (T4 x2, a llama-server per GPU). The 12 (config, seed) jobs are split 3 per session.

    python kaggle/ablations/build.py            # writes kaggle/ablations/push/<account>/{data,kernel}
Then push the dataset and the kernel per account (see push.sh).
"""

import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "push"
ACCOUNTS = ["seronic2001", "shubhadeepmandal"]
CONFIGS = ["ours-llm", "A1", "A2", "A3"]
SEEDS = [0, 1, 2]
JOBS = [(c, s) for s in SEEDS for c in CONFIGS]  # 12
PARTS = {(a, part): JOBS[2 * i + k::4] for i, a in enumerate(ACCOUNTS) for k, part in enumerate("ab")}  # 4 x 3
SKIP_DATA = {"distill", "encoder-intents", "encoder-intents-v2", "lookup-train", "lookup-train-v2", "lookup-train-v3",
             "lookup-train-v3_2", "lookup-train-v4", "memory-train", "memory-train-v1_1", "multitask",
             "multitask-readable", "multitask.zip", "multitask-readable.zip", "compiler.zip"}


def code_zip(dest: Path) -> None:
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ["pyproject.toml", "uv.lock", "adapters-gpu.toml", "adapters.toml"]:
            z.write(ROOT / f, f)
        for top in ["src", "data"]:
            for p in (ROOT / top).rglob("*"):
                rel = p.relative_to(ROOT)
                if p.is_dir() or "__pycache__" in rel.parts:
                    continue
                if top == "data" and rel.parts[1] in SKIP_DATA:
                    continue
                z.write(p, rel.as_posix())


def keys() -> str:
    lines = (ROOT / ".env").read_text(encoding="utf-8").splitlines()
    return "\n".join(l for l in lines if l.startswith("GEMINI_API_KEY")) + "\n"


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    for acct in ACCOUNTS:
        data, kern = OUT / acct / "data", OUT / acct / "kernel"
        data.mkdir(parents=True)
        kern.mkdir(parents=True)
        code_zip(data / "nts-code.zip")
        shutil.copy2(ROOT.parent / "nts-read100.gguf", data / "nts-read100.gguf")
        (data / "keys.env").write_text(keys(), encoding="utf-8")
        for part in "ab":
            (data / f"jobs-{part}.json").write_text(json.dumps([{"config": c, "seed": s} for c, s in PARTS[acct, part]]))
        (data / "dataset-metadata.json").write_text(json.dumps(
            {"title": "nts-ablation-code", "id": f"{acct}/nts-ablation-code", "licenses": [{"name": "unknown"}]}))
        for part in "ab":
            k = kern / part
            k.mkdir()
            (k / "nts_ablation_run.py").write_text(
                (HERE / "nts_ablation_run.py").read_text(encoding="utf-8").replace("__PART__", part), encoding="utf-8")
            (k / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{acct}/nts-ablations-v2{part}", "title": f"nts-ablations-v2{part}", "code_file": "nts_ablation_run.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
            "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [f"{acct}/nts-ablation-code"], "kernel_sources": [f"{acct}/nts-eval-tools"],
            "competition_sources": []}, indent=1))
            print(acct, part, [f"{c}/s{s}" for c, s in PARTS[acct, part]])


if __name__ == "__main__":
    main()
