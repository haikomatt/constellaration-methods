#!/usr/bin/env bash
# E30 (and later banks): one breadth bank at a requested aspect, the E24r restoration chain from its
# best confirmed seed, and, if that reaches feasibility, a front in the design's own representation
# with the anchor render check. The E29 pipeline with the tag and aspect as arguments. Registered and
# locked 2026-10-27 in the pre-registration snapshot (docs/preregistration, E30 lock). One container at a time on cores 2-13;
# hard stop for new launches 2026-11-05 06:00 (DEADLINE_AT overrides).
# Not shipped here: the campaign's database generator CLI (scripts/nae_db_p3_generator.py, whose
# `build` step nae_db_seed_screen ports) and the E21 selection files, the near-axis database and the
# restoration checkpoint it reads (docs/external-data.md). This file records the orchestration; its
# launch discipline is pinned by tests/fronts/test_chain_runner_sh.py. Run from the repository root:
#   scripts/p3_bank_chain_runner.sh TAG ASPECT MAX_ASPECT > results/p3-TAG.log 2>&1
set -u
# Site settings, all overridable from the environment. M is the campaign checkout that holds the
# inputs these runners read but this repository does not ship (the restoration checkpoint and control
# candidate, the E21 selection files, the near-axis database); PYF is the interpreter of the
# database authors' pyQIC fork (docs/external-data.md).
M=${CM_CAMPAIGN_ROOT:-$(pwd)}
IMG=${CM_IMAGE:-localhost/constellaration-methods:test}
PYF=${CM_FORK_PYTHON:-$M/data/qi_nae_db/fork-venv/bin/python}
INIT_SCALE=0.005
MAX_LINKS=4
DEADLINE=$(date -d "${DEADLINE_AT:-2026-11-05 06:00}" +%s)
LOCK_LABEL=${LOCK_LABEL:-E30-lock}  # provenance of the candidate step; set per registration
LOCK=docs/preregistration/$(ls docs/preregistration | grep "$LOCK_LABEL" | tail -1)
TAG=${1:-}; ASPECT=${2:-}; MAX_ASPECT=${3:-}
BANK_TAG=${BANK_TAG:-$TAG}   # E33: reuse another run's bank (its build and screen are skipped when present)
SEED_LABEL=${SEED_LABEL:-}   # E33: restore from this confirmed seed instead of the bank's best
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

bank() { # build + canonical screen, as E24 with the aspect set to 10.0
  local D=results/p3-$BANK_TAG-bank
  deadline_ok || return 1
  if [ ! -f $D/seeds/manifest.json ]; then
    log "$TAG bank: host build at aspect $ASPECT (cores 2-13)"
    nice -n 10 taskset -c 2-13 $PYF scripts/nae_db_p3_generator.py build --selection results/p3-e21-nae-db/selection_primary.json --selection-extra results/p3-e21-nae-db/selection_extra.json --replace-failures --data-repo $M --mag-well 0.02 --aspect $ASPECT --aspect-by estimate --min-rmin-ratio 0.5 --workers 12 --out-dir $D/seeds > $D-build.log 2>&1
    log "$TAG bank: build exit $? n_success=$(jget $D/seeds/manifest.json "d['n_success']")"
  fi
  if [ ! -f $D/eval/REPORT.json ]; then
    log "$TAG bank: canonical screen"
    timeout 14400 $PODRUN scripts/p3_construction_bank_eval.py --boundary-dir $D/seeds/boundaries --manifest $D/seeds/manifest.json --top 960 --confirm-top 12 --workers 12 --material-v 0.30 --substantial-v 0.10 --require-rows iota,mirror,well --max-aspect $MAX_ASPECT --out $D/eval > $D-eval.log 2>&1
    log "$TAG bank: screen exit $? label=$(jget $D/eval/REPORT.json "d['label_with_rows']") best=$(jget $D/eval/REPORT.json "d['best_confirmed_label']") V=$(jget $D/eval/REPORT.json "round(d['best_confirmed_worst_of_three_v'],4)")"
  fi
}

restore_link() { # $1 link number; link 1 starts from the bank's best confirmed seed
  local k=$1 R=results/p3-${TAG}r-$1
  deadline_ok || return 1
  mkdir -p $R
  if [ "$k" -eq 1 ]; then
    $PODRUN scripts/nae_db_p3_generator.py candidate --eval-report results/p3-$BANK_TAG-bank/eval/REPORT.json --seed-manifest results/p3-$BANK_TAG-bank/seeds/manifest.json --boundary-dir results/p3-$BANK_TAG-bank/eval/boundaries ${SEED_LABEL:+--label $SEED_LABEL} --lock-snapshot "$LOCK" --out $R/start_candidate.json > $R/candidate.log 2>&1 || { log "${TAG}r link 1: candidate failed"; return 1; }
  else
    cp results/p3-${TAG}r-$((k - 1))/best_candidate.json $R/start_candidate.json
  fi
  local v bar; v=$(jget $R/start_candidate.json "max(d['r'])"); bar=$(python3 -c "print(round($v-0.010,4))")
  log "${TAG}r link $k: launching from V=$v (material bar $bar)"
  launch_detached p3-${TAG}r-$k --network none -v "$(pwd)":/workspace:Z -v $M/.git:$M/.git:ro,Z -v $M/results:$M/results:ro,Z -w /workspace -e OMP_NUM_THREADS=1 --pids-limit 8192 --ulimit nproc=16384:16384 --entrypoint taskset $IMG -c 2-13 python3 scripts/p3_final_restoration.py --checkpoint $M/results/m1-post-ladder-d176/seed_0176/checkpoint.json --control-candidate $M/results/p3-restoration-176-run2/candidate2_0.152095.json --start-candidate $R/start_candidate.json --out-dir $R --workers 12 --line-search --second-order --coeff-space-box --delta0 0.0025 --delta-cap 0.01 --material-bar $bar --max-opt-evals 2400 --max-verif-evals 12 --total-hours 12 --opt-cutoff-hours 10
  wait_exit p3-${TAG}r-$k
  log "${TAG}r link $k: label=$(jget $R/REPORT.json "d['label']") adj0=$(jget $R/REPORT.json "round(d['adjudication'][0]['worst_v'],4)") feasible=$(jget $R/REPORT.json "d['adjudication'][0]['feasible']")"
  $PODRUN -c "
import json,sys; sys.path.insert(0,'src')
from constellaration_methods.restoration.p3_promote_candidate import boundary_dict_of_candidate, reflect_toroidally
R='$R'; r=json.load(open(R+'/REPORT.json')); sc=json.load(open(R+'/start_candidate.json')); a=r['adjudication'][0]
c={'tag':'anchor:${TAG}r-$k','note':'${TAG}r link $k best '+a['tag'],'x':a['x'],'r':a['r'],'raw':None,'boundary_dict':sc['boundary_dict'],'mpol':sc['mpol'],'ntor':sc['ntor'],'nfp':sc['nfp'],'scaling':sc['scaling'],'source':sc['source'],'provenance':{**sc['provenance'],'link':$k,'chain_tag':a['tag'],'runner_label':r['label'],'feasible':a['feasible']}}
json.dump(c,open(R+'/best_candidate.json','w'),indent=1); bd=boundary_dict_of_candidate(c); json.dump(bd,open(R+'/best_boundary.json','w'),indent=1); json.dump(reflect_toroidally(bd),open(R+'/best_boundary_reflected.json','w'),indent=1)
" > $R/export.log 2>&1
  $PODRUN scripts/score_mhd_stable.py --boundary $R/best_boundary.json --out $R/best_recheck.json > $R/recheck.log 2>&1
  python3 -c "
import json; R='$R'; c=json.load(open(R+'/best_candidate.json')); s=json.load(open(R+'/best_recheck.json'))[0]
c['raw']=s.get('raw'); c['recheck_feasible']=s.get('feasible'); json.dump(c,open(R+'/best_candidate.json','w'),indent=1); print('recheck feasible', s.get('feasible'), 'V', round(s.get('V',9),4))"
}

[[ "${BASH_SOURCE[0]}" == "$0" ]] || return 0  # sourced for its functions (tests): stop here
[ -n "$TAG" ] && [ -n "$ASPECT" ] && [ -n "$MAX_ASPECT" ] || { log "usage: TAG ASPECT MAX_ASPECT"; exit 2; }
log "$TAG: starting (aspect $ASPECT, max canonical aspect $MAX_ASPECT)"
bank || { log "$TAG: bank not run; stop"; exit 0; }
case "$(jget results/p3-$BANK_TAG-bank/eval/REPORT.json "d['label_with_rows']")" in material|substantial|goal) ;; *) log "$TAG: bank not material; restoration not funded"; exit 0;; esac
feasible_link=""
for k in $(seq 1 $MAX_LINKS); do
  restore_link $k || break
  R=results/p3-${TAG}r-$k
  if [ "$(jget $R/best_candidate.json "d.get('recheck_feasible')")" = "True" ] && [ "$(jget $R/REPORT.json "d['adjudication'][0]['feasible']")" = "True" ]; then feasible_link=$k; log "${TAG}r: feasible at link $k"; break; fi
  python3 -c "import sys; sys.exit(0 if $(jget $R/REPORT.json "d['adjudication'][0]['worst_v']") <= $(jget $R/start_candidate.json "max(d['r'])")-0.010 else 1)" || { log "${TAG}r: link $k not material; chain stopped"; break; }
done
[ -n "$feasible_link" ] || { log "$TAG runner done (no feasible design)"; exit 0; }
R=results/p3-${TAG}r-$feasible_link; REP=$R/start_serialised.json
mkdir -p results/p3-$TAG-anchors && $PODRUN scripts/p3_union_anchors.py --candidate $R/best_candidate.json --out results/p3-$TAG-anchors/REPORT.json > results/p3-$TAG-anchors/union.log 2>&1 && log "$TAG anchors: $(tail -1 results/p3-$TAG-anchors/union.log)"
prev_hv=0; anchors=results/p3-$TAG-anchors/REPORT.json
for suf in f fb fc fd; do
  front_link "$TAG$suf" $anchors || { log "$TAG$suf: not launched or stopped (see above); chain stopped"; break; }
  hv=$(jget results/p3-$TAG$suf-front/REPORT.json "d['confirmed_hypervolume']")
  if [ "$suf" != "f" ]; then python3 -c "import sys; sys.exit(0 if $hv >= $prev_hv*1.02 else 1)" || { log "$TAG front chain stop: +$(python3 -c "print(round(($hv/$prev_hv-1)*100,2))")% < 2%"; break; }; fi
  prev_hv=$hv; anchors=results/p3-$TAG$suf-front/REPORT.json
done
log "$TAG runner done"
