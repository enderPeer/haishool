#!/usr/bin/env bash
# Distribute a life8 many-world search over hosts by ssh (one tmux session per host), and collect the results.
#
#   scripts/life8_search.sh start   TAG START_SEED "SEARCH ARGS" HOST:WORKERS:WORLDS[:CAP_MB] ...
#   scripts/life8_search.sh resume  TAG "SEARCH ARGS" HOST:WORKERS[:WORLDS[:CAP_MB]] ...
#   scripts/life8_search.sh status  TAG HOST ...
#   scripts/life8_search.sh stop    TAG HOST ...
#   scripts/life8_search.sh collect TAG HOST ...
#
# Example (run from the Haishool checkout; ssh aliases adler40 knecht24 specht32 falke64):
#   scripts/life8_search.sh start r8a 0 "--preset living --source bridged --rounds 3 --ticks 500 --keep 0.25 --reps 6" \
#       adler40:22:400 knecht24:22:300:1500 specht32:10:150 falke64:10:250
#   scripts/life8_search.sh status r8a adler40 knecht24 specht32 falke64
#   scripts/life8_search.sh collect r8a adler40 knecht24 specht32 falke64
#
# What it does
# - start: ONE candidate list (plan) of sum(WORLDS) worlds from START_SEED is built HERE
#   (bridged and world7 worlds need haishool.evo and numpy, which the hosts lack; one plan
#   means no world7 world is searched twice), then cut into consecutive parts in host
#   order (search split). The hosts only need python3 >= 3.11, tmux and the standard library.
#   --preset and --option values in SEARCH ARGS shape the plan (use --preset living: plain
#   Config worlds have no kin credit, predators or blooms, so signalling cannot pay there).
#   The code (haishool/__init__.py and haishool/life8, no caches), the host's plan part and a
#   launcher run.sh are copied with tar/cat over ssh into ~/life8-search/TAG/ on each host;
#   nothing else on the host is touched (no GPU, no services, no other folders or sessions).
#   The search runs under nice 10 in a detached tmux session named life8-search (it
#   survives disconnects), output in ~/life8-search/TAG/out, log in ~/life8-search/TAG/search.log.
#   A host that already has a life8-search tmux session or a TAG/out folder is refused.
# - Output cap: CAP_MB (default 1500) is passed as --max-output-mb. Before each round the
#   search projects the folder size (current files plus one end state per running world,
#   sized by the largest checkpoint so far, at least 1 MB) and stops with
#   stop_reason "output_cap" instead of filling the disk; results.jsonl keeps every row.
#   The engine writes no per-tick logs in a search; only results.jsonl, search.json and the
#   gzipped checkpoints of kept worlds are stored. Give knecht24 a hard cap.
# - resume: re-launches the same search in a new life8-search session (it skips finished
#   rounds and worlds); a new CAP_MB replaces the saved cap.
# - status: the tmux session, the last log lines, search.json status, results.jsonl lines, disk.
# - stop: kills only the life8-search tmux session (the folder stays; resume continues it).
# - collect: copies results.jsonl, search.json and search.log of each host into
#   runs/life8-search/TAG/HOST/ and merges them with
#   python -m haishool.life8.search collect (runs/life8-search/TAG/merged/report.json).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL="${LIFE8_SEARCH_LOCAL:-$ROOT/runs/life8-search}"
PY="${PYTHON:-python}"
SSH="${LIFE8_SSH:-ssh -o BatchMode=yes -o ConnectTimeout=15}"  # LIFE8_SSH: a stand-in for local dry runs
SESSION="${LIFE8_SESSION:-life8-search}"

usage() { sed -n '2,8p' "$0"; exit 2; }

hostspec() {  # HOST:WORKERS[:WORLDS[:CAP_MB]] -> HOST WORKERS WORLDS CAP
    local host workers worlds cap
    IFS=: read -r host workers worlds cap <<<"$1"
    [[ -n "$host" && "$workers" =~ ^[0-9]+$ && "${worlds:-0}" =~ ^[0-9]+$ ]] \
        || { echo "bad host spec '$1' (HOST:WORKERS:WORLDS[:CAP_MB])" >&2; exit 2; }
    echo "$host" "$workers" "${worlds:-0}" "${cap:-1500}"
}

check_tag() { [[ "$1" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "TAG must be [A-Za-z0-9._-]+" >&2; exit 2; }; }

run_args() {  # SEARCH ARGS without --source (the plan carries it)
    sed 's/--source[ =][a-z0-9]*//' <<<"$1"
}

launch() {  # HOST TAG WORKERS CAP "SEARCH ARGS": write run.sh, start it in a detached tmux session
    local host="$1" tag="$2" workers="$3" cap="$4" args="$5"
    # shellcheck disable=SC2029
    $SSH "$host" "cat > ~/life8-search/$tag/run.sh" <<EOF
#!/usr/bin/env bash
# life8 search $tag on $host (written by scripts/life8_search.sh; started in tmux session $SESSION)
cd "\$HOME/life8-search/$tag/code" || exit 1
echo "start \$(date -u +%Y-%m-%dT%H:%M:%SZ) on \$(hostname), workers $workers, cap ${cap} MB" >> ../search.log
nice -n 10 python3 -m haishool.life8.search run --out ../out --plan ../plan.json --workers $workers \\
    --max-output-mb $cap $args >> ../search.log 2>&1 < /dev/null
echo "exit \$? \$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> ../search.log
EOF
    # shellcheck disable=SC2029
    $SSH "$host" "python3 -c 'import sys; assert sys.version_info >= (3, 11), sys.version' \
        && ! tmux has-session -t $SESSION 2>/dev/null \
        && tmux new-session -d -s $SESSION 'bash ~/life8-search/$tag/run.sh' \
        && echo started $host tmux session $SESSION" \
        || { echo "$host: not started (python3 < 3.11, or a tmux session $SESSION exists)" >&2; return 1; }
}

cmd="${1:-}"; shift || true
case "$cmd" in
start)
    [[ $# -ge 4 ]] || usage
    tag="$1" start="$2" args="$3"; shift 3
    check_tag "$tag"
    stage="$LOCAL/$tag/stage"; mkdir -p "$stage"
    source_kind="$(sed -n 's/.*--source[ =]\([a-z0-9]*\).*/\1/p' <<<"$args")"; source_kind="${source_kind:-plain}"
    plan_args=(); read -r -a words <<<"$args"
    for ((k = 0; k < ${#words[@]}; k++)); do  # --option key=value pairs and --preset also shape the plan
        [[ "${words[k]}" == "--option" || "${words[k]}" == "--preset" ]] && plan_args+=("${words[k]}" "${words[k+1]}")
    done
    hosts=() parts=() total=0
    for spec in "$@"; do
        read -r host workers worlds cap <<<"$(hostspec "$spec")"
        [[ "$worlds" -ge 1 ]] || { echo "$spec: give WORLDS for start" >&2; exit 2; }
        hosts+=("$spec"); parts+=("$host=$worlds"); total=$((total + worlds))
        $SSH "$host" "test ! -e ~/life8-search/$tag/out && ! tmux has-session -t $SESSION 2>/dev/null" \
            || { echo "$host: ~/life8-search/$tag/out or tmux session $SESSION exists; use resume or a new TAG" >&2; exit 1; }
    done
    "$PY" -m haishool.life8.search plan --out "$stage/plan.json" --worlds "$total" --source "$source_kind" \
        --start-seed "$start" --workers "${LIFE8_PLAN_WORKERS:-8}" ${plan_args[@]+"${plan_args[@]}"}
    "$PY" -m haishool.life8.search split --plan "$stage/plan.json" --out-dir "$stage" "${parts[@]}"
    for spec in "${hosts[@]}"; do
        read -r host workers worlds cap <<<"$(hostspec "$spec")"
        $SSH "$host" "mkdir -p ~/life8-search/$tag/code"
        tar -C "$ROOT" --exclude=__pycache__ -czf - haishool/__init__.py haishool/life8 \
            | $SSH "$host" "tar -xzf - -C ~/life8-search/$tag/code"
        $SSH "$host" "cat > ~/life8-search/$tag/plan.json" < "$stage/$host.plan.json"
        echo "$host: $worlds worlds, workers $workers, cap ${cap}MB"
        launch "$host" "$tag" "$workers" "$cap" "$(run_args "$args")"
    done
    ;;
resume)
    [[ $# -ge 3 ]] || usage
    tag="$1" args="$2"; shift 2
    check_tag "$tag"
    for spec in "$@"; do
        read -r host workers worlds cap <<<"$(hostspec "$spec")"
        launch "$host" "$tag" "$workers" "$cap" "$(run_args "$args")"
    done
    ;;
status)
    [[ $# -ge 2 ]] || usage
    tag="$1"; shift
    check_tag "$tag"
    for host in "$@"; do
        echo "== $host"
        # shellcheck disable=SC2029
        $SSH "$host" "tmux has-session -t $SESSION 2>/dev/null && echo 'tmux $SESSION: running' || echo 'tmux $SESSION: none'; \
            cd ~/life8-search/$tag 2>/dev/null && tail -n 3 search.log; \
            python3 -c \"import json; m=json.load(open('out/search.json')); print(m['status'], m.get('stop_reason'))\" 2>/dev/null; \
            echo results.jsonl lines: \$(cat out/results.jsonl 2>/dev/null | wc -l); du -sh out 2>/dev/null; df -h ~ | tail -1; uptime" || true
    done
    ;;
stop)
    [[ $# -ge 2 ]] || usage
    tag="$1"; shift
    check_tag "$tag"
    for host in "$@"; do
        # shellcheck disable=SC2029
        $SSH "$host" "tmux kill-session -t $SESSION 2>/dev/null && echo '$host: stopped $SESSION' || echo '$host: no $SESSION session'"
    done
    ;;
collect)
    [[ $# -ge 2 ]] || usage
    tag="$1"; shift
    check_tag "$tag"
    folders=()
    for host in "$@"; do
        mkdir -p "$LOCAL/$tag/$host"
        $SSH "$host" "cd ~/life8-search/$tag && tar -czf - -C out results.jsonl search.json -C .. search.log" \
            | tar -xzf - -C "$LOCAL/$tag/$host" && folders+=("$LOCAL/$tag/$host") || echo "$host: nothing to collect" >&2
    done
    [[ ${#folders[@]} -gt 0 ]] || exit 1
    mkdir -p "$LOCAL/$tag/merged"
    "$PY" -m haishool.life8.search collect --out "$LOCAL/$tag/merged" "${folders[@]}" > "$LOCAL/$tag/merged/summary.txt"
    echo "merged ${#folders[@]} hosts -> $LOCAL/$tag/merged/report.json"
    ;;
*) usage ;;
esac
