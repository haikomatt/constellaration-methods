#!/usr/bin/env bash
# E31: restore a verified tolerance-edge design toward strict nominal (--success-tol 0.0), then, if the
# margin improved, a front chain from the restored design in its own representation with the anchor
# render check. Registered and locked 2026-10-29 in the pre-registration snapshot (docs/preregistration, E31 lock). One container
# at a time on cores 2-13; hard stop for new launches 2026-11-05 06:00. Run from the repository root:
#   scripts/p3_nominal_front_runner.sh TAG SOURCE_CANDIDATE > results/p3-TAG.log 2>&1
set -u
# Site settings, overridable from the environment. M is the campaign checkout that holds the inputs
# these runners read but this repository does not ship (restoration checkpoint, control candidate).
M=${CM_CAMPAIGN_ROOT:-$(pwd)}
IMG=${CM_IMAGE:-localhost/constellaration-methods:test}
INIT_SCALE=0.005
DEADLINE=$(date -d "${DEADLINE_AT:-2026-11-05 06:00}" +%s)
TAG=${1:-}; SRC=${2:-}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/p3_chain_lib.sh"
deadline_ok() { [ "$(date +%s)" -lt "$DEADLINE" ] || { log "deadline passed; no new launch"; return 1; }; }
PODRUN="podman run --rm --network none -v $(pwd):/workspace:Z -v $M/.git:$M/.git:ro,Z -v $M/results:$M/results:ro,Z -w /workspace -e OMP_NUM_THREADS=1 --pids-limit 8192 --ulimit nproc=16384:16384 --entrypoint taskset $IMG -c 2-13 python3"
wait_exit() { while podman ps --format '{{.Names}}' | grep -q "^$1\$"; do sleep 300; done; log "$1 exited: $(podman ps -a --format '{{.Names}} {{.Status}}' | grep "^$1 ")"; }
jget() { python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2], {}, {'d': d}))" "$1" "$2"; }
REP=""  # set to the feasible design's own representation before any front link

render_check() {
  local name=$1 anchors=$2 arch=results/p3-$1-nsga2/archive.jsonl rc
  for _ in $(seq 1 60); do
    if [ -f "$arch" ]; then
      local src=""; case "$anchors" in results/p3-*-front/REPORT.json) src="--source-archive ${anchors%-front/REPORT.json}-nsga2/archive.jsonl";; esac
      python3 "$HERE/p3_anchor_render_check.py" "$anchors" "$arch" $src > results/p3-$name-render.txt; rc=$?
      [ $rc -eq 0 ] && { log "$name: anchor render check OK ($(grep -c '^OK' results/p3-$name-render.txt) anchors)"; return 0; }
      [ $rc -eq 1 ] && { log "$name: anchor render check FAILED: $(grep MISMATCH results/p3-$name-render.txt | head -2)"; podman stop "p3-$name" >/dev/null; return 1; }
    fi
    sleep 60
  done
  log "$name: anchor render check timed out waiting for generation 0"; podman stop "p3-$name" >/dev/null; return 1
}

front_link() { # $1 name, $2 anchors REPORT
  deadline_ok || return 1
  local n; n=$(jget $2 "len([m for m in d['members'] if m.get('retained')])"); local pop; pop=$(python3 -c "import math; print(max(56, 4*math.ceil($n/4)))")
  log "$1: launching from $2 ($n anchors, pop $pop, init scale $INIT_SCALE)"
  launch_detached p3-$1 --network none -v "$(pwd)":/workspace:Z -v $M/.git:$M/.git:ro,Z -w /workspace -e OMP_NUM_THREADS=1 --pids-limit 8192 --ulimit nproc=16384:16384 --entrypoint taskset $IMG -c 2-13 python3 scripts/b3_nsga2_mhd.py --anchors $2 --representation $REP --eval-fidelity canonical --pop-size $pop --init-scale $INIT_SCALE --workers 12 --total-hours 24 --max-evals 3000 --out-dir results/p3-$1-nsga2
  sleep 120; podman ps --format '{{.Names}}' | grep -q "^p3-$1\$" || { log "$1 exited at once: $(podman logs p3-$1 2>&1 | tail -1)"; return 1; }
  render_check "$1" "$2" || return 1
  wait_exit p3-$1
  timeout 10800 $PODRUN scripts/p3_front_from_archive.py results/p3-$1-nsga2 --confirm --max-confirmations 300 --workers 12 --out-dir results/p3-$1-front > results/p3-$1-front.log 2>&1
  log "$1: confirm exit $? members=$(jget results/p3-$1-front/REPORT.json "d['n_retained']") hv=$(jget results/p3-$1-front/REPORT.json "round(d['confirmed_hypervolume'],2)")"
}

nominal_restore() { # strict-nominal restoration from the source candidate
  local R=results/p3-$TAG-nominal
  deadline_ok || return 1
  mkdir -p $R; cp "$SRC" $R/start_candidate.json
  local v; v=$(jget $R/start_candidate.json "max(d['r'])")
  log "$TAG: strict-nominal restoration from V=$v"
  launch_detached p3-$TAG --network none -v "$(pwd)":/workspace:Z -v $M/.git:$M/.git:ro,Z -v $M/results:$M/results:ro,Z -w /workspace -e OMP_NUM_THREADS=1 --pids-limit 8192 --ulimit nproc=16384:16384 --entrypoint taskset $IMG -c 2-13 python3 scripts/p3_final_restoration.py --checkpoint $M/results/m1-post-ladder-d176/seed_0176/checkpoint.json --control-candidate $M/results/p3-restoration-176-run2/candidate2_0.152095.json --start-candidate $R/start_candidate.json --out-dir $R --workers 12 --line-search --second-order --coeff-space-box --delta0 0.0025 --delta-cap 0.01 --material-bar 0.005 --success-tol 0.0 --max-opt-evals 2400 --max-verif-evals 12 --total-hours 12 --opt-cutoff-hours 10
  wait_exit p3-$TAG
  log "$TAG: label=$(jget $R/REPORT.json "d['label']") stop=$(jget $R/REPORT.json "d['stop_reason']") adj0=$(jget $R/REPORT.json "round(d['adjudication'][0]['worst_v'],6)") strict=$(jget $R/REPORT.json "d['adjudication'][0].get('strictly_nominal')")"
  $PODRUN -c "
import json,sys; sys.path.insert(0,'src')
from constellaration_methods.restoration.p3_promote_candidate import boundary_dict_of_candidate, reflect_toroidally
R='$R'; r=json.load(open(R+'/REPORT.json')); sc=json.load(open(R+'/start_candidate.json')); a=r['adjudication'][0]
c={'tag':'anchor:$TAG-nominal','note':'$TAG strict-nominal best '+a['tag'],'x':a['x'],'r':a['r'],'raw':None,'boundary_dict':sc['boundary_dict'],'mpol':sc['mpol'],'ntor':sc['ntor'],'nfp':sc['nfp'],'scaling':sc['scaling'],'source':sc['source'],'provenance':{**sc['provenance'],'nominal_tag':a['tag'],'runner_label':r['label'],'feasible':a['feasible'],'strictly_nominal':a.get('strictly_nominal')}}
json.dump(c,open(R+'/best_candidate.json','w'),indent=1); bd=boundary_dict_of_candidate(c); json.dump(bd,open(R+'/best_boundary.json','w'),indent=1); json.dump(reflect_toroidally(bd),open(R+'/best_boundary_reflected.json','w'),indent=1)
" > $R/export.log 2>&1
  $PODRUN scripts/score_mhd_stable.py --boundary $R/best_boundary.json --out $R/best_recheck.json > $R/recheck.log 2>&1
  python3 -c "
import json; R='$R'; c=json.load(open(R+'/best_candidate.json')); s=json.load(open(R+'/best_recheck.json'))[0]
c['raw']=s.get('raw'); c['recheck_feasible']=s.get('feasible'); json.dump(c,open(R+'/best_candidate.json','w'),indent=1); print('recheck feasible', s.get('feasible'), 'V', round(s.get('V',9),6))"
}

[[ "${BASH_SOURCE[0]}" == "$0" ]] || return 0  # sourced for its functions (tests): stop here
[ -n "$TAG" ] && [ -f "$SRC" ] || { log "usage: TAG SOURCE_CANDIDATE (an existing verified candidate)"; exit 2; }
log "$TAG: starting from $SRC"
nominal_restore || { log "$TAG: restoration not run; stop"; exit 0; }
R=results/p3-$TAG-nominal
start_v=$(jget $R/start_candidate.json "max(d['r'])"); new_v=$(jget $R/REPORT.json "d['adjudication'][0]['worst_v']")
python3 -c "import sys; sys.exit(0 if $new_v <= $start_v - 0.0025 else 1)" || { log "$TAG: null (gain < 0.0025); no front"; exit 0; }
[ "$(jget $R/best_candidate.json "d.get('recheck_feasible')")" = "True" ] || { log "$TAG: restored point fails its re-check; no front"; exit 0; }
REP=$R/start_serialised.json
mkdir -p results/p3-$TAG-anchors && $PODRUN scripts/p3_union_anchors.py --candidate $R/best_candidate.json --out results/p3-$TAG-anchors/REPORT.json > results/p3-$TAG-anchors/union.log 2>&1 && log "$TAG anchors: $(tail -1 results/p3-$TAG-anchors/union.log)"
prev_hv=0; anchors=results/p3-$TAG-anchors/REPORT.json
for suf in f fb fc fd; do
  front_link "$TAG$suf" $anchors || { log "$TAG$suf: not launched or stopped (see above); chain stopped"; break; }
  hv=$(jget results/p3-$TAG$suf-front/REPORT.json "d['confirmed_hypervolume']")
  if [ "$suf" != "f" ]; then python3 -c "import sys; sys.exit(0 if $hv >= $prev_hv*1.02 else 1)" || { log "$TAG front chain stop: +$(python3 -c "print(round(($hv/$prev_hv-1)*100,2))")% < 2%"; break; }; fi
  prev_hv=$hv; anchors=results/p3-$TAG$suf-front/REPORT.json
done
log "$TAG runner done"
