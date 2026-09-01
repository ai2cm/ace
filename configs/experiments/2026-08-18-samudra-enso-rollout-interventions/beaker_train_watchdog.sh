#!/bin/bash
# Watchdog for long Beaker training jobs: hourly sweep, resume what died
# transiently, park what needs a human.
#
# Policy:
#   - Transient deaths (NCCL collective timeout, watchdog kill, SIGTERM /
#     preemption residue) get `beaker experiment resume`: the experiment
#     keeps its results dataset, so fme restarts from its own checkpoints.
#     Capped at MAX_RESUMES per watched name, so a crash-loop cannot spin.
#   - OOMs are parked, never resumed: an OOM repeats until a config changes.
#   - Unrecognized failures are parked with the last error lines echoed.
#   - Exits when everything watched is finished (exit 0) or parked.
#
# Configuration (environment):
#   WATCH       space-separated experiment names or name prefixes, required.
#               A prefix also matches gantry's 4-hex-suffix duplicates, so
#               "myexp" matches "myexp" and "myexp-a3f7" and follows the
#               NEWEST non-canceled attempt.
#   WORKSPACES  beaker workspaces to sweep (default "ai2/ace")
#   OWNER       experiment owner prefix (default: `beaker account whoami`)
#   INTERVAL    seconds between sweeps (default 3600)
#   MAX_RESUMES per watched name (default 5)
#   LOG         log file (default ./watchdog.log)
#   DRY=1       report what would happen without resuming anything
#
# Run it detached and single-instance, e.g.:
#   WATCH="myexp-a myexp-b" nohup ./beaker_train_watchdog.sh & disown
set -u

WATCH="${WATCH:?set WATCH to experiment names/prefixes}"
WORKSPACES="${WORKSPACES:-ai2/ace}"
OWNER="${OWNER:-$(beaker account whoami --format=json | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["name"])')}"
LOG="${LOG:-./watchdog.log}"
INTERVAL="${INTERVAL:-3600}"
DRY="${DRY:-0}"
MAX_RESUMES="${MAX_RESUMES:-5}"
declare -A parked_ resumes_

say() { echo "$(date -u +%Y-%m-%dT%H:%M) $*" | tee -a "$LOG"; }

say "watchdog started (watch: $WATCH; workspaces: $WORKSPACES; interval ${INTERVAL}s, dry=$DRY, pid $$)"
while true; do
  snapshot=$(for ws in $WORKSPACES; do
    beaker workspace experiments "$ws" --format json 2>/dev/null
  done | python3 -c "
import json,sys
out=[]; dec=json.JSONDecoder(); s=sys.stdin.read().strip(); i=0
while i < len(s):
    obj,j=dec.raw_decode(s,i); out.extend(obj); i=j
    while i<len(s) and s[i] in ' \n\r\t': i+=1
print(json.dumps(out))")
  unfinished=0
  for name in $WATCH; do
    [ "${parked_[$name]:-}" = "1" ] && continue
    read -r exp code <<<"$(echo "$snapshot" | NAME="$name" python3 -c "
import json,os,re,sys
try: es=json.load(sys.stdin)
except Exception: sys.exit()
name=os.environ['NAME']
c=[]
for e in es:
    n=e.get('name','')
    if not re.fullmatch(re.escape(name)+r'(-[0-9a-f]{4})?', n): continue
    j=(e.get('jobs') or [None])[-1]
    st=(j or {}).get('status',{})
    if st.get('canceled') and not st.get('started'): continue
    c.append((e.get('created',''), n, str(st.get('exitCode'))))
c.sort()
if c: print(c[-1][1], c[-1][2])")"
    if [ -z "${exp:-}" ]; then say "$name: no live experiment found (API hiccup?) — will retry next sweep"; unfinished=1; continue; fi
    if [ "$code" = "None" ] || [ -z "$code" ]; then unfinished=1; continue; fi
    if [ "$code" = "0" ]; then say "$name ($exp): FINISHED (exit 0)"; parked_[$name]=1; continue; fi
    tailtxt=$(beaker experiment logs "$OWNER/$exp" 2>/dev/null | tail -300)
    if echo "$tailtxt" | grep -qiE "CUDA out of memory|OutOfMemoryError"; then
      say "$name ($exp): DEAD by OOM — needs a config change, NOT auto-resuming; parked"
      parked_[$name]=1; continue
    fi
    if echo "$tailtxt" | grep -qiE "collective operation timeout|Watchdog caught|NCCL|Signal 15|SIGTERM"; then
      n=$(( ${resumes_[$name]:-0} + 1 ))
      if [ "$n" -gt "$MAX_RESUMES" ]; then
        say "$name ($exp): transient death but resume cap ($MAX_RESUMES) reached — parked"
        parked_[$name]=1; continue
      fi
      resumes_[$name]=$n
      if [ "$DRY" = "1" ]; then
        say "$name ($exp): would RESUME (transient death, attempt $n) [dry]"
      elif beaker experiment resume "$OWNER/$exp" >>"$LOG" 2>&1; then
        say "$name ($exp): transient death — RESUMED (attempt $n/$MAX_RESUMES)"
      else
        say "$name ($exp): resume FAILED — parked (see log)"; parked_[$name]=1
      fi
      unfinished=1; continue
    fi
    say "$name ($exp): DEAD, unrecognized failure — not auto-resuming; parked. Last error lines:"
    echo "$tailtxt" | grep -B1 -A3 -iE "Error|Traceback" | head -8 | tee -a "$LOG"
    parked_[$name]=1
  done
  [ "$unfinished" = "0" ] && { say "everything watched is finished or parked — watchdog exiting"; break; }
  sleep "$INTERVAL"
done
