#!/usr/bin/env bash
# A/B one region with one worker: the same inputs through two frozen snapshots,
# then the outputs compared byte for byte and the CPU time reported.
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
  # A tree from before the legacy decision was deleted still needs the flag
  # to run the same decision; a later one no longer accepts it.
  decision=$(grep -q '"--decision"' "$repo/placer/main.py" && echo "--decision mechanism")
  (cd "$repo" && PLACER_PERF_LOG="$O/$side.perf.tsv" /usr/bin/time -v python -m placer.main \
      $T/results/alignments/human_hg002/full/r0.primary.md.bam \
      /mnt/home1/miska/hl725/scratch/placer_dev/ref/GRCh38.primary.fa \
      $T/resources/human/dfam_human.freeze.fa \
      --region "$REGION" --threads 1 $decision --output-dir "$O/$side" 2> "$O/$side.stderr")
  echo "$side $repo: $(grep -E 'User time|System time|Elapsed' "$O/$side.stderr" | tr -s ' ' | tr '\n' ' ')"
done
for f in scientific.txt evidence_ledger.tsv calls.csv; do
  if cmp -s "$O/A/$f" "$O/B/$f"; then echo "IDENTICAL $f"; else echo "DIFFERS   $f"; fi
done
# calls.vcf without its `##source=` line, which names the code's version: an
# installed release says `1.0.0a1`, a source tree `1.0.0a1+source`.
if cmp -s <(grep -v '^##source=' "$O/A/calls.vcf") <(grep -v '^##source=' "$O/B/calls.vcf"); then
  echo "IDENTICAL calls.vcf (without ##source)"; else echo "DIFFERS   calls.vcf"; fi
