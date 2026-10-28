#!/usr/bin/env bash
# Training episodes for nts-agent-v6: the v6 families (nts.eval.compositional.V6) with coordinator-v6, best of 3,
# on the four training departments, then Gemini rewordings ("--rewrite train") on train-d1 and train-big.
#   bash scripts/collect_v6.sh gemini|gemini31|llama
# gemini: gemini-3.5-flash-lite (seeds 901-904, 911/914); gemini31: gemini-3.1-flash-lite (921-924, 931/934);
# llama: the served nts-agent-v4-read-100 (941-944, 951/954), which waits for the server. A finished report is
# skipped, so the script resumes; a run the daily quota stopped leaves no report and is run again the next day.
# The compositional suite (eval.jsonl, F-103) stays held out: no family runs on the eval stores.
cd "$(dirname "$0")/.." || exit 1
PY="../negotiate-then-solve/.venv/Scripts/python.exe"
URL=http://10.1.25.191:8080
FAMILIES="window_move or_windows ranked_move day_cap days_cap free_gap after_lecture courses_apart even_weeks
          except_week two_windows_off move_apart"
case "$1" in
  gemini) MODEL=(--model gemini --gemini-model gemini-3.5-flash-lite); BASE=900 ;;
  gemini31) MODEL=(--model gemini --gemini-model gemini-3.1-flash-lite); BASE=920 ;;
  llama) MODEL=(--model llama --adapter nts-agent-v4-read-100); BASE=940 ;;
  *) echo "usage: $0 gemini|gemini31|llama" >&2; exit 2 ;;
esac
up() { [ "$1" != llama ] && return; until curl -s -m 8 -o /dev/null "$URL/health"; do sleep 60; done; }
run() {  # store seed [extra...]
  local store=$1 seed=$2; shift 2
  local stem="runs/eval/collect-v6-${WHO}-${store}-s${seed}"
  [ "${1:-}" = "--rewrite" ] && stem="${stem}-train"
  [ -f "$stem.json" ] && return
  up "$WHO"
  # shellcheck disable=SC2086
  PYTHONPATH=src "$PY" -m nts eval "${MODEL[@]}" --agent coordinator-v6 --env ../negotiate-then-solve/.env \
    --store "runs/agentic/${store}.jsonl" --families $FAMILIES --per-family 2 --best-of 3 --seed "$seed" \
    --max-seconds 600 "$@" --out "$stem.json" > "$stem.log" 2>&1
  echo "$(date +%H:%M) $WHO $store s$seed: $(tail -n 1 "$stem.log")"
}
WHO=$1
run train-d1 $((BASE + 1))
run train-d2 $((BASE + 2))
run train-d3 $((BASE + 3))
run train-big $((BASE + 4))
run train-d1 $((BASE + 11)) --rewrite train
run train-big $((BASE + 14)) --rewrite train
echo "$(date +%H:%M) $WHO done"
