#!/usr/bin/env bash
# Run the pipeline on the fixed development slices, one 10 Mb region per species,
# so any two commits can be compared on the same inputs.
#
#   sbatch tools/run_dev_slices.sh                 # outputs under $OUT_ROOT/<commit>/
#   OUT_ROOT=/elsewhere sbatch tools/run_dev_slices.sh
#
# The slices are inside TEBench's development_contigs (human chr1-8); the cichlid
# dataset defines none, so chr1 is used. Nothing here touches a holdout contig.
# Mouse joins once results/alignments/mouse_b6x129_f1/full/r0.primary.md.bam exists.
#
#SBATCH --job-name=placer-dev-slices
#SBATCH --partition=2204
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.log
set -euo pipefail

REPO=${REPO:-/mnt/beegfs6/home1/miska/hl725/placer_py}
PY=${PY:-$HOME/anaconda3/envs/te_bench/bin/python}
TEBENCH=${TEBENCH:-/mnt/home1/miska/hl725/scratch/projects/TE_bechmark}
OUT_ROOT=${OUT_ROOT:-/mnt/home1/miska/hl725/scratch/placer_dev/runs}
REF_DIR=${REF_DIR:-/mnt/home1/miska/hl725/scratch/placer_dev/ref}
THREADS=${THREADS:-16}

commit=$(git -C "$REPO" rev-parse --short HEAD)
dirty=$(git -C "$REPO" status --porcelain --untracked-files=no | grep -q . && echo "-dirty" || true)
out="$OUT_ROOT/$commit$dirty"
mkdir -p "$out" "$REF_DIR"

# TEBench's reference is checksum-pinned and has no .fai; index a symlink here
# rather than write a sidecar into the registry's resources.
ref_link() {
    local src=$1 name=$2
    [ -e "$REF_DIR/$name" ] || ln -s "$src" "$REF_DIR/$name"
    [ -e "$REF_DIR/$name.fai" ] || samtools faidx "$REF_DIR/$name"
    echo "$REF_DIR/$name"
}

HUMAN_REF=$(ref_link "$TEBENCH/resources/human/GRCh38.primary.fa" GRCh38.primary.fa)
CICHLID_REF=$(ref_link /mnt/beegfs/scratch/miska/hl725/projects/cichlid/urika/te_callers/runs/ref/fAstCal671.primary.fa fAstCal671.primary.fa)

#       name     bam                                                              reference     library                                                 region
SLICES=(
  "human_hg002 $TEBENCH/results/alignments/human_hg002/full/r0.primary.md.bam $HUMAN_REF $TEBENCH/resources/human/dfam_human.freeze.fa chr1:10000001-20000000"
  "cichlid_d2 /mnt/beegfs/scratch/miska/hl725/projects/cichlid/urika/data/D2.bam $CICHLID_REF $TEBENCH/resources/cichlid/MWCichlidTE-3.2.tebench.fa chr1:10000001-20000000"
)
MOUSE_BAM=$TEBENCH/results/alignments/mouse_b6x129_f1/full/r0.primary.md.bam
if [ -e "$MOUSE_BAM" ]; then
    MOUSE_REF=$(ref_link "$TEBENCH/resources/mouse/GRCm39.primary.fa" GRCm39.primary.fa)
    SLICES+=("mouse_b6x129_f1 $MOUSE_BAM $MOUSE_REF $TEBENCH/resources/mouse/dfam_mouse.freeze.fa chr1:10000001-20000000")
fi

cd "$REPO"
for slice in "${SLICES[@]}"; do
    read -r name bam ref lib region <<<"$slice"
    dest="$out/$name"
    mkdir -p "$dest"
    echo "== $name $region -> $dest"
    /usr/bin/time -v "$PY" -m placer_py.main "$bam" "$ref" "$lib" \
        --region "$region" --threads "$THREADS" --output-dir "$dest" \
        2> "$dest/stderr.log" || echo "FAILED: $name (see $dest/stderr.log)"
    grep -E "Elapsed|Maximum resident" "$dest/stderr.log" || true
done
echo "done: $out"
