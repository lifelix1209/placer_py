#!/usr/bin/env bash
# Sampling profile (py-spy, 100 Hz) of one region with one worker, from a
# FROZEN snapshot (REPO). PYSPY: the py-spy binary; OUT_ROOT: where results go.
#   cd tools/perf && sbatch --export=ALL,REPO=<snapshot>,REGION=chr1:30000001-30500000,TAG=typ spy_region.sh
# Sampling, unlike cProfile, adds no per-call cost, so tiny
# functions called millions of times are not inflated.
#SBATCH --job-name=placer-spy
#SBATCH --partition=2204
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --exclude=node5
#SBATCH --output=/mnt/home1/miska/hl725/scratch/placer_dev/prof/%x_%j.log
set -euo pipefail
PYBIN=/mnt/home1/miska/hl725/anaconda3/envs/placer-dev/bin
export PATH=$PYBIN:$PATH
: "${REPO:?}" "${REGION:?}" "${TAG:?}"
T=/mnt/home1/miska/hl725/scratch/projects/TE_bechmark
O=${OUT_ROOT:-/mnt/home1/miska/hl725/scratch/placer_dev/prof}/spy_$TAG
mkdir -p "$O"
cd "$REPO"
/usr/bin/time -v ${PYSPY:-/mnt/beegfs/scratch/miska/hl725/placer_dev/tools_bin/py-spy} record -r 100 -f raw \
    -o "$O/folded.txt" -- $PYBIN/python -m placer.main \
    $T/results/alignments/human_hg002/full/r0.primary.md.bam \
    /mnt/home1/miska/hl725/scratch/placer_dev/ref/GRCh38.primary.fa \
    $T/resources/human/dfam_human.freeze.fa \
    --region "$REGION" --threads 1 --decision mechanism --output-dir "$O/out" 2> "$O/stderr.log"
grep -E "Elapsed|User time|System time" "$O/stderr.log"
# sbatch runs a spool copy of this script, so $0 is not beside fold_summary.py:
# submit from tools/perf, or set PERF_TOOLS.
$PYBIN/python "${PERF_TOOLS:-${SLURM_SUBMIT_DIR:-.}}/fold_summary.py" "$O/folded.txt" > "$O/summary.txt"
