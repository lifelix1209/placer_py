#!/usr/bin/env bash
# A/B two frozen snapshots on one exclusive node.
#   1. one worker, typical and pericentromeric 500 kb, A then B (x REPS);
#   2. three concurrent 16-worker runs (48 processes, like a release node),
#      8 Mb each, all A, then all B;
#   3. B again on the concurrent regions with PLACER_VERIFY_LOCAL_FETCH=1,
#      which fails the run if a buffered local fetch differs from the index.
# All outputs are compared byte for byte; calls.vcf without its `##source=`
# line, which names the code's version and so differs between a release
# install and a source tree by design.
#   sbatch --export=ALL,A=<snapshot>,B=<snapshot>,TAG=... ab_step.sh
#SBATCH --job-name=perf1-ab
#SBATCH --partition=2204
#SBATCH --exclusive
#SBATCH --mem=300G
#SBATCH --time=08:00:00
#SBATCH --exclude=node5
#SBATCH --output=/mnt/home1/miska/hl725/scratch/placer_dev/prof/%x_%j.log
set -euo pipefail
export PATH=$HOME/anaconda3/envs/placer-dev/bin:$PATH
: "${A:?}" "${B:?}" "${TAG:?}" "${REPS:=2}" "${VERIFY:=1}"
T=/mnt/home1/miska/hl725/scratch/projects/TE_bechmark
BAM=$T/results/alignments/human_hg002/full/r0.primary.md.bam
REF=/mnt/home1/miska/hl725/scratch/placer_dev/ref/GRCh38.primary.fa
LIB=$T/resources/human/dfam_human.freeze.fa
O=${OUT_ROOT:-/mnt/home1/miska/hl725/scratch/placer_dev/prof}/ab_${TAG}_${SLURM_JOB_ID}
mkdir -p "$O"
echo "host $(hostname) $(date -Iseconds); load $(cut -d' ' -f1-3 /proc/loadavg)"
echo "A=$A"; echo "B=$B"

run() {  # NAME REPO CPUSET REGION THREADS
  local name=$1 repo=$2 cpus=$3 region=$4 threads=$5
  mkdir -p "$O/$name"
  (cd "$repo" && PLACER_PERF_LOG="$O/$name/perf.tsv" taskset -c "$cpus" \
      /usr/bin/time -v python -m placer.main "$BAM" "$REF" "$LIB" \
      --region "$region" --threads "$threads" --output-dir "$O/$name/out" \
      2> "$O/$name/stderr.log") || echo "FAILED $name (see $O/$name/stderr.log)"
  echo "$name: $(grep -E 'User time|System time|Elapsed|Maximum resident' \
      "$O/$name/stderr.log" | sed 's/^\s*//' | tr '\n' ' ' || true)"
}
same() {
  local bad=0
  for f in scientific.txt evidence_ledger.tsv calls.csv; do
    cmp -s "$O/$1/out/$f" "$O/$2/out/$f" || { echo "DIFFERS   $1 $2 $f"; bad=1; }
  done
  cmp -s <(grep -v '^##source=' "$O/$1/out/calls.vcf") \
         <(grep -v '^##source=' "$O/$2/out/calls.vcf") \
      || { echo "DIFFERS   $1 $2 calls.vcf"; bad=1; }
  [ $bad = 0 ] && echo "IDENTICAL $1 $2 (4 outputs)"
  return 0
}

echo "== 1. one worker"
for rep in $(seq 1 "$REPS"); do
  for r in typ peri; do
    region=chr1:30000001-30500000; [ $r = peri ] && region=chr1:121000001-121500000
    run A_${r}_$rep "$A" 0-3 "$region" 1
    run B_${r}_$rep "$B" 0-3 "$region" 1
    same A_${r}_$rep B_${r}_$rep
  done
done

if [ -n "${SLOW_REGION:-}" ]; then
  echo "== 1b. one worker, $SLOW_REGION, A and B side by side on separate cores"
  run A_slow "$A" 64-67 "$SLOW_REGION" 1 &
  run B_slow "$B" 68-71 "$SLOW_REGION" 1 &
  wait
  same A_slow B_slow
fi

echo "== 2. three concurrent 16-worker runs"
REGIONS=(chr1:20000001-28000000 chr1:28000001-36000000 chr1:36000001-44000000)
for side in A B; do
  repo=${!side}
  for i in 0 1 2; do
    lo=$(( i * 16 )); hi=$(( lo + 15 ))
    ( export PLACER_SCAN_CHUNK_BP=500000
      run conc_${side}_$i "$repo" $lo-$hi "${REGIONS[$i]}" 16 ) &
  done
  wait
done
for i in 0 1 2; do same conc_A_$i conc_B_$i; done

if [ "$VERIFY" = 1 ]; then
  echo "== 3. B, concurrent, every buffered local fetch checked against the index"
  for i in 0 1 2; do
    lo=$(( i * 16 )); hi=$(( lo + 15 ))
    ( export PLACER_SCAN_CHUNK_BP=500000 PLACER_VERIFY_LOCAL_FETCH=1
      run verify_B_$i "$B" $lo-$hi "${REGIONS[$i]}" 16 ) &
  done
  wait
  for i in 0 1 2; do same conc_B_$i verify_B_$i; done
  grep -l "AssertionError" "$O"/verify_B_*/stderr.log || echo "no buffer mismatch"
fi

echo "== per-run accounting (tools/perf/perf_summary.py)"
for f in "$O"/*/perf.tsv; do
  echo "-- $(basename "$(dirname "$f")")"
  python3 "$B/tools/perf/perf_summary.py" "$f" --top 3
done
echo "done $(date -Iseconds)"
