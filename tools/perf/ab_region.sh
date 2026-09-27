#!/usr/bin/env bash
# A/B one region with one worker: the same inputs through two frozen snapshots,
# then the five outputs compared byte for byte and the CPU time reported.
#   sbatch --export=ALL,A=<baseline snapshot>,B=<changed snapshot>,REGION=...,TAG=... ab_region.sh
#SBATCH --job-name=placer-ab
#SBATCH --partition=2204
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --exclude=node5
#SBATCH --output=/mnt/home1/miska/hl725/scratch/placer_dev/prof/%x_%j.log
set -euo pipefail
export PATH=$HOME/anaconda3/envs/placer-dev/bin:$PATH
: "${A:?}" "${B:?}" "${REGION:?}" "${TAG:?}"
T=/mnt/home1/miska/hl725/scratch/projects/TE_bechmark
O=${OUT_ROOT:-/mnt/home1/miska/hl725/scratch/placer_dev/prof}/ab_$TAG
mkdir -p "$O"
for side in A B; do
  repo=${!side}
  (cd "$repo" && /usr/bin/time -v python -m placer.main \
      $T/results/alignments/human_hg002/full/r0.primary.md.bam \
      /mnt/home1/miska/hl725/scratch/placer_dev/ref/GRCh38.primary.fa \
      $T/resources/human/dfam_human.freeze.fa \
      --region "$REGION" --threads 1 --decision mechanism --output-dir "$O/$side" 2> "$O/$side.stderr")
  echo "$side $repo: $(grep -E 'User time|System time|Elapsed' "$O/$side.stderr" | tr -s ' ' | tr '\n' ' ')"
done
for f in scientific.txt evidence_ledger.tsv structural_calls.tsv calls.vcf calls.csv; do
  if cmp -s "$O/A/$f" "$O/B/$f"; then echo "IDENTICAL $f"; else echo "DIFFERS   $f"; fi
done
