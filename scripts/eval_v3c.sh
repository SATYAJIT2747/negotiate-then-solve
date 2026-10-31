#!/usr/bin/env bash
# v3 adapters (dpo, read-050, read-100): s0, heldout rewrites, 20-teacher dept, seeds 1-2. Waits for the server before
# each run; a run with connection errors (model_error) is set aside as *.unreachable and tried again (twice at most).
cd "C:/Users/Shubh/Desktop/LMA Major Project/negotiate-then-solve-agentic" || exit 1
PY="../negotiate-then-solve/.venv/Scripts/python.exe"
URL=http://10.1.25.191:8080
up() { until curl -s -m 8 -o /dev/null "$URL/health"; do sleep 60; done; }
run() {  # adapter store seed tag [extra...]
  local a=$1 store=$2 seed=$3 tag=$4; shift 4
  local stem="runs/eval/served_${a}-${tag}" try
  [ -f "$stem.json" ] && return
  for try in 1 2 3; do
    up
    PYTHONPATH=src "$PY" -m nts eval --model llama --adapter "$a" --agent coordinator-v3 --env ../negotiate-then-solve/.env \
      --store "runs/agentic/${store}.jsonl" --seed "$seed" --families q_free q_classes away preference move room_closed not_allowed impossible ambiguous clarify negotiate negotiate_refused rules_question weekend clarify_negotiate hedged_yes no_reply two_changes away_days \
      --max-seconds 900 "$@" --out "$stem.json" > "$stem.log" 2>&1
    echo "$(date +%H:%M) $a $tag (try $try): $(tail -n 1 "$stem.log")"
    grep -q "model_error" "$stem.log" || return
    mv "$stem.log" "$stem.unreachable$try.log"; [ -f "$stem.json" ] && mv "$stem.json" "$stem.unreachable$try.json"
  done
}
run nts-agent-v3-dpo eval 0 s0
for a in nts-agent-v3-dpo nts-agent-v3-read-050 nts-agent-v3-read-100; do
  run $a eval 0 s0-heldout --rewrite heldout
  run $a eval-big 0 big-s0
  run $a eval 1 s1
  run $a eval 2 s2
done
echo "$(date +%H:%M) done"
