"""Paired ablation rerun on Kaggle (GPU T4 x2): the 4B + nts-read100 agent on a llama-server per GPU, Gemini
simulators (key from the private dataset), one `evaluation.negotiation` process per (config, seed) job.

Writes /kaggle/working/results/: <config>-s<seed>.json (+ .rows.jsonl), logs/, servers.json, summary.json.
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import tomllib
import urllib.request
import zipfile
from pathlib import Path

T0 = time.time()
INP, WORK = Path("/kaggle/input"), Path("/kaggle/working")
RES = WORK / "results"
(RES / "logs").mkdir(parents=True, exist_ok=True)


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)


def find(name):
    return sorted(INP.glob(f"**/{name}"), key=lambda p: len(p.parts))


def sh(cmd):
    log("$", cmd)
    subprocess.run(cmd, shell=True, check=True)


# -- code, keys, jobs -----------------------------------------------------------------------------------------------
CODE = Path("/tmp/nts")
z = find("nts-code.zip")
if z and z[0].is_file():
    with zipfile.ZipFile(z[0]) as f:
        f.extractall(CODE)
else:  # Kaggle unpacked the zip into a folder of the same name
    src = min((p.parent for p in find("pyproject.toml")), key=lambda p: len(p.parts))
    shutil.copytree(src, CODE, dirs_exist_ok=True)
jobs = json.loads(find("jobs-__PART__.json")[0].read_text())
env = dict(os.environ)
for line in find("keys.env")[0].read_text().splitlines():
    if "=" in line:
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
log(f"{len(jobs)} jobs:", [f"{j['config']}/s{j['seed']}" for j in jobs], "| keys:",
    sum(k.startswith("GEMINI_API_KEY") for k in env))

deps = tomllib.loads((CODE / "pyproject.toml").read_text())["project"]["dependencies"]
sh(f"{sys.executable} -m pip install -q uv || {sys.executable} -m pip install -q --break-system-packages uv")
sh(f"{sys.executable} -m uv venv -q --python {sys.executable} /tmp/venv")
sh(f"{sys.executable} -m uv pip install -q --python /tmp/venv/bin/python " + " ".join(f"'{d}'" for d in deps))
PY = "/tmp/venv/bin/python"

# -- a llama-server per GPU ---------------------------------------------------------------------------------------
tools = [p for p in find("llama-server") if p.parent.name == "bin"][0].parent.parent
BIN = Path("/tmp/bin")
BIN.mkdir(exist_ok=True)
shutil.copy2(tools / "bin" / "llama-server", BIN / "llama-server")
os.chmod(BIN / "llama-server", 0o755)
base = next(iter(tools.glob("*.gguf")))
lora = find("nts-read100.gguf")[0]
log("base", base, "| lora", lora)
NP, CTX = 4, 32768
ports = [8080, 8081]
for gpu, port in enumerate(ports):
    out = open(RES / "logs" / f"server-{gpu}.log", "w")
    subprocess.Popen([str(BIN / "llama-server"), "-m", str(base), "--jinja", "-ngl", "99", "-c", str(CTX),
                      "-np", str(NP), "--host", "127.0.0.1", "--port", str(port), "--lora-init-without-apply",
                      "--lora", str(lora)], stdout=out, stderr=subprocess.STDOUT,
                     env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})


def get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.load(r)


servers = {}
for port in ports:
    for _ in range(600):
        try:
            if get(f"http://127.0.0.1:{port}/health").get("status") == "ok":
                break
        except Exception:  # noqa: BLE001 - loading
            pass
        time.sleep(2)
    servers[port] = {"loras": get(f"http://127.0.0.1:{port}/lora-adapters"),
                     "model": get(f"http://127.0.0.1:{port}/v1/models")["data"][0]["id"]}
(RES / "servers.json").write_text(json.dumps(servers, indent=1, default=str))
log("servers up")


# -- jobs: one process each, spread over the two servers ----------------------------------------------------------
summary = {}
lock = threading.Lock()


def run(job, port):
    name = f"{job['config']}-s{job['seed']}"
    cmd = [PY, "-m", "evaluation.negotiation", "--configs", job["config"], "--seeds", str(job["seed"]),
           "--ablation-all-seeds", "--subset", "30", "--local", "--local-url", f"http://127.0.0.1:{port}/v1",
           "--local-adapter", "nts-read100", "--out", str(RES / f"{name}.json")]
    t = time.time()
    rc = None
    for attempt in range(3):  # a quota stop or crash resumes from the rows already logged
        with open(RES / "logs" / f"{name}.log", "a") as out:
            rc = subprocess.run(cmd + (["--resume"] if attempt else []), cwd=CODE, stdout=out,
                                stderr=subprocess.STDOUT, env={**env, "PYTHONPATH": "src"}).returncode
        if rc == 0:
            break
        log(f"{name}: rc {rc}, retry in 60 s")
        time.sleep(60)
    with lock:
        summary[name] = {"rc": rc, "minutes": round((time.time() - t) / 60, 1), "port": port}
        (RES / "summary.json").write_text(json.dumps(summary, indent=1))
    log(f"done {name} rc={rc} in {summary[name]['minutes']} min")


threads = [threading.Thread(target=run, args=(j, ports[i % 2])) for i, j in enumerate(jobs)]
for th in threads:
    th.start()
for th in threads:
    th.join()
log("all done", summary)
