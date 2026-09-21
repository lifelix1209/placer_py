#!/usr/bin/env python3
"""Build a TE consensus library from Dfam, in the header format PLACER parses.

WHY A SCRIPT AND NOT A COMMITTED FASTA. The library is an INPUT to the caller,
not part of it, and which library you use changes every call. Pinning the
Dfam release in a script that anyone can re-run is reproducible; committing a
few MB of consensus sequence is a fork of Dfam that silently goes stale.

WHY THE HEADER FORMAT MATTERS. `seqtools.parse_te_name_parts` reads the FIRST
whitespace-delimited token and then splits on `#` and `/`. Dfam's own FASTA
headers are `>DF000000053.4 AluYa5`, whose first token is the accession -- so
used as-is, every call comes back with family `NA`, and `family_state_
compatibility` in the decision policy has nothing to key on. This rewrites
them to RepeatMasker's `AluYa5#SINE/Alu`, which is the convention the parser
was built for.

    python3 tools/fetch_te_library.py --out te_library.fa

Default clade is human (9606) plus its ancestral clades, which is ~1400
families. `--clade` takes any NCBI taxon id, so the same script produces the
mouse or zebrafish library -- the point being that nothing here is specific to
the dataset the caller is evaluated on.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import sys
import urllib.request

API = "https://dfam.org/api/families"


def _get(url: str, timeout: int) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def _decode_body(body: str) -> str:
    """The download endpoint's `body`, which is not consistently encoded.

    It is normally base64 of a gzip stream, but the padding is sometimes
    stripped and the field is occasionally plain FASTA already. Guessing from
    the `encoding` field does not work -- it says "identity" in both cases --
    so this sniffs the decoded bytes instead of trusting the envelope.
    """
    if body.lstrip().startswith(">"):
        return body
    padded = body + "=" * (-len(body) % 4)
    raw = base64.b64decode(padded)
    if raw[:2] == b"\x1f\x8b":
        return gzip.decompress(raw).decode()
    return raw.decode()


def fetch_sequences(clade: int, relatives: str, timeout: int) -> dict[str, str]:
    """Consensus sequences, keyed by the Dfam family NAME (not the accession)."""
    url = (f"{API}?clade={clade}&clade_relatives={relatives}"
           f"&download=true&format=fasta&limit=2000")
    payload = _get(url, timeout)
    text = _decode_body(payload["body"])

    out: dict[str, str] = {}
    name: str | None = None
    chunks: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if name:
                out.setdefault(name, "".join(chunks))
            # ">DF000000053.4 AluYa5" -- the name is the SECOND token.
            parts = line[1:].strip().split(None, 1)
            name = parts[1].strip() if len(parts) > 1 else parts[0]
            chunks = []
        else:
            chunks.append(line.strip())
    if name:
        out.setdefault(name, "".join(chunks))
    return out


def fetch_classifications(clade: int, relatives: str, timeout: int) -> dict[str, tuple[str, str]]:
    """`name -> (repeat_type, repeat_subtype)`, paged through the summary API."""
    out: dict[str, tuple[str, str]] = {}
    start = 0
    while True:
        url = (f"{API}?clade={clade}&clade_relatives={relatives}"
               f"&limit=500&start={start}&format=summary")
        data = _get(url, timeout)
        results = data.get("results", [])
        for family in results:
            out[family["name"]] = (family.get("repeat_type_name") or "Unknown",
                                   family.get("repeat_subtype_name") or "")
        start += len(results)
        if not results or start >= data.get("total_count", 0):
            break
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="output FASTA path")
    parser.add_argument("--clade", type=int, default=9606,
                        help="NCBI taxon id (default 9606, human)")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--min-length", type=int, default=50,
                        help="drop consensus sequences shorter than this")
    args = parser.parse_args()

    sequences: dict[str, str] = {}
    classes: dict[str, tuple[str, str]] = {}
    # `descendants` first so a clade-specific definition wins over the
    # ancestral one of the same name.
    for relatives in ("descendants", "ancestors"):
        sequences.update({k: v for k, v in
                          fetch_sequences(args.clade, relatives, args.timeout).items()
                          if k not in sequences})
        classes.update({k: v for k, v in
                        fetch_classifications(args.clade, relatives, args.timeout).items()
                        if k not in classes})
        print(f"  {relatives:12s} cumulative: {len(sequences)} sequences, "
              f"{len(classes)} classifications", file=sys.stderr)

    written = skipped = unclassified = 0
    with open(args.out, "w") as handle:
        for name, seq in sorted(sequences.items()):
            if len(seq) < args.min_length:
                skipped += 1
                continue
            repeat_type, repeat_subtype = classes.get(name, ("Unknown", ""))
            if repeat_type == "Unknown":
                unclassified += 1
            label = f"{repeat_type}/{repeat_subtype}" if repeat_subtype else repeat_type
            handle.write(f">{name}#{label}\n")
            for i in range(0, len(seq), 60):
                handle.write(seq[i:i + 60] + "\n")
            written += 1

    print(f"wrote {written} families to {args.out} "
          f"({skipped} too short, {unclassified} unclassified)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
