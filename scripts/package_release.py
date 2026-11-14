"""Assemble a release archive from the artefacts a reader actually needs.

    python scripts/package_release.py --out dist/
    python scripts/package_release.py --out dist/ --tier full --i-have-the-dua
    python scripts/package_release.py --list          # what would be included

WHAT GOES IN, AND THE ONE THING THAT DOES NOT
---------------------------------------------
Model weights, ONNX graphs, the result JSONs, the generated LaTeX tables and the
figures. Those are derived statistics and parameters: publishing them is the
point of an artefact release.

Per-window PREDICTION DUMPS are excluded by default, and this is not a size
decision. `runs/paper_results/*/predictions/*.npz` carry `y_reg`, `last_bg`,
`patient` and `t_time` -- per-subject CGM values, subject identifiers and
timestamps. That is OhioT1DM itself, which is released under a data use
agreement that does not permit redistribution. Publishing the dumps alongside
the code would re-release the licensed dataset in a form nobody audits, and the
figures those dumps feed are already in the archive as PDFs. `--tier full`
includes them for a collaborator who holds the DUA, and requires
`--i-have-the-dua` so that it cannot be reached by accident.

Regenerable caches (`ohio_ws.pkl`, `ohio_windows.pt`, `naive_ws.pkl`) are always
excluded. They are hundreds of megabytes, they are a pure function of the raw
data plus the pipeline version, and shipping a stale one is how a reader ends up
reproducing the wrong windows -- the failure the version key exists to prevent.

SMOKE ARTEFACTS ARE REFUSED
---------------------------
A JSON labelled `"smoke": true` was produced at a two-epoch budget. Packaging it
would put numbers into a citable archive that the repository itself does not
consider results, so this refuses rather than warns. `--allow-smoke` overrides,
for testing this script.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (repo-relative path, tier, description, suffixes). A directory entry sweeps
# only files matching `suffixes`; `()` means every file under it.
#
# The suffix filter is load-bearing, not tidiness. Without it the entry for
# `runs/paper_results/ablations` -- meant to pick up the generated `.tex`
# tables -- also swept `ablations/predictions/*.npz` into the PUBLIC tier, which
# is the licensed-data leak this whole tiering exists to prevent. Directory
# globs are the wrong primitive for a safety boundary, so `assert_publishable`
# below re-checks the result independently of how it was assembled.
CONTENTS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    # ---- license ------------------------------------------------------------
    ("LICENSE", "public", "open source software license", ()),
    # ---- aggregate results: the numbers every table and figure derives from --
    ("runs/paper_results/benchmarks/results_summary.json", "public", "SOTA benchmark metrics", ()),
    ("runs/paper_results/ablations/ablations_summary.json", "public", "ablation studies A-F", ()),
    ("runs/paper_results/personalization/personalization_sweep.json", "public", "12-subject personalisation sweep", ()),
    ("runs/paper_results/edge/latency_sweep.json", "public", "C++ runner latency percentiles", ()),
    # ---- generated LaTeX: tables and the macro file the prose resolves through
    ("runs/paper_results/benchmarks/table_sota_comparison.tex", "public", "Tables 1-2 combined", ()),
    # Named explicitly, not by suffix sweep: main.tex \inputs the per-horizon
    # files (PH = 30 in the body, PH = 60 in the appendix), so an archive
    # cannot rebuild the paper without them.
    ("runs/paper_results/benchmarks/table_sota_h30.tex", "public", "Table 1 (PH = 30)", ()),
    ("runs/paper_results/benchmarks/table_sota_h60.tex", "public", "Table 2 (PH = 60, appendix)", ()),
    ("runs/paper_results/ablations", "public", "ablation tables", (".tex",)),
    ("runs/paper_results/personalization", "public", "personalisation tables", (".tex",)),
    ("manuscript/numbers.tex", "public", "every inline number in the manuscript", ()),
    # The MetaboNet appendix table, named EXPLICITLY -- one file, no suffix
    # sweep. It is the only artefact from that tree any tier ships, and it is
    # here because `main.tex` \inputs it: without it an archive cannot rebuild
    # the appendix whose numbers `numbers.tex` quotes.
    #
    # The isolation invariant is unchanged in substance. What it exists to stop
    # is a DIRECTORY entry sweeping a tree it was not meant to (the incident
    # that put `ablations/predictions/*.npz` into the public tier), and
    # `assert_publishable` still refuses anything carrying a restricted marker.
    # A single named `.tex` of aggregate metrics -- over a corpus that excludes
    # OhioT1DM and HUPA-UCM by default -- can be neither of those things.
    ("runs/metabonet_results/benchmarks/table_metabonet_comparison.tex", "public",
     "appendix table: MetaboNet corpus-scale comparison", ()),
    # ---- multi-seed sweep and its post-processing: aggregates only. Named
    # file by file; the per-seed directories hold `predictions/` and are never
    # swept. main.tex \inputs these tables and numbers.tex quotes these JSONs.
    ("runs/paper_results/benchmarks_seeds/multiseed_summary.json", "public",
     "multi-seed benchmark: mean/SD and subject-cluster bootstrap CIs", ()),
    ("runs/paper_results/benchmarks_seeds/table_sota_multiseed_h30.tex", "public", "multi-seed table (PH = 30)", ()),
    ("runs/paper_results/benchmarks_seeds/table_sota_multiseed_h60.tex", "public", "multi-seed table (PH = 60)", ()),
    ("runs/paper_results/benchmarks_seeds/table_multiseed_diffs_h30.tex", "public", "multi-seed paired differences (PH = 30)", ()),
    ("runs/paper_results/benchmarks_seeds/table_multiseed_diffs_h60.tex", "public", "multi-seed paired differences (PH = 60)", ()),
    ("runs/paper_results/conformal_bootstrap.json", "public",
     "calibration-bootstrap stability of conformal coverage", ()),
    ("runs/paper_results/fapd_latching/fapd_latching.json", "public", "alarm-policy sensitivity", ()),
    ("runs/paper_results/fapd_latching/table_fapd_latching.tex", "public", "alarm-policy sensitivity table", ()),
    # ---- figures ------------------------------------------------------------
    ("manuscript/figures", "public", "figures 1-6", (".pdf", ".png")),
    # ---- deployable model ---------------------------------------------------
    # The population checkpoint was trained on OhioT1DM, so it is not public:
    # it ships only in the DUA tier.
    ("runs/paper_results/personalization/base_model.pt", "full", "population checkpoint (PyTorch; trained on OhioT1DM)", ()),
    ("runs/demo/causalcgm_mt.onnx", "public", "exported ONNX graph, CGM-only", ()),
    ("runs/demo/causalcgm_mt_fusion.onnx", "public", "exported ONNX graph, fusion (the timed one)", ()),
    # ---- DUA-restricted -----------------------------------------------------
    ("runs/paper_results/benchmarks/predictions", "full", "per-window predictions (CONTAINS OhioT1DM CGM VALUES)", (".npz",)),
    ("runs/paper_results/ablations/predictions", "full", "per-window predictions (CONTAINS OhioT1DM CGM VALUES)", (".npz",)),
)

# Never packaged at any tier: regenerable, large, and dangerous when stale.
EXCLUDE_PARTS: tuple[str, ...] = (
    "ohio_ws.pkl", "naive_ws.pkl", "ohio_windows.pt",
    "_snapshots", "_superseded", "__pycache__", ".DS_Store", "logs",
)

# JSONs whose provenance is checked before anything is written.
PROVENANCE_JSON: tuple[str, ...] = (
    "runs/paper_results/benchmarks/results_summary.json",
    "runs/paper_results/ablations/ablations_summary.json",
    "runs/paper_results/personalization/personalization_sweep.json",
    # Not a paper-tree artefact -- the corpus-scale driver is forbidden to write
    # there -- but `manuscript/numbers.tex` IS packaged, in the public tier, and
    # 46 of its macros resolve out of this file. Omitting it meant a release
    # could publish corpus-scale numbers backed by a two-epoch run through the
    # one script written to prevent exactly that.
    "runs/metabonet_results/benchmarks/metabonet_summary.json",
)


def _excluded(rel: str) -> bool:
    return any(part in rel.split(os.sep) or part in rel for part in EXCLUDE_PARTS)


def _walk(rel: str, suffixes: tuple[str, ...] = ()) -> list[str]:
    """Repo-relative files under `rel`, filtered. Missing paths yield nothing."""
    full = os.path.join(REPO, rel)
    if os.path.isfile(full):
        return [] if _excluded(rel) else [rel]
    if not os.path.isdir(full):
        return []
    out = []
    for dirpath, dirnames, filenames in os.walk(full):
        dirnames[:] = [d for d in dirnames if not _excluded(d)]
        for fn in sorted(filenames):
            if suffixes and not fn.endswith(suffixes):
                continue
            r = os.path.relpath(os.path.join(dirpath, fn), REPO)
            if not _excluded(r):
                out.append(r)
    return sorted(out)


# Anything matching these is OhioT1DM-derived per-window data and may never
# appear in a `public` archive, however it got there.
RESTRICTED_MARKERS: tuple[str, ...] = (".npz", "/predictions/", "predictions" + os.sep)


def assert_publishable(files: list[tuple[str, str]], tier: str) -> None:
    """Independent re-check that a public archive carries no licensed data.

    The tier filter above decides what to include; this decides what may leave.
    Keeping them separate is the point: the first version of this script had a
    correct-looking tier table and still swept thirty prediction dumps into the
    public tier through a directory glob, because one entry meant to collect
    `.tex` files collected everything beneath it. A boundary that depends on
    every glob being right is not a boundary.
    """
    if tier != "public":
        return
    bad = sorted(f for f, _ in files
                 if any(m in f for m in RESTRICTED_MARKERS))
    if bad:
        raise SystemExit(
            "REFUSING to write a public archive containing OhioT1DM-derived "
            "per-window data:\n  " + "\n  ".join(bad) +
            "\nThese carry CGM values, subject identifiers and timestamps. Fix "
            "CONTENTS, or use --tier full --i-have-the-dua.")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_provenance(allow_smoke: bool) -> list[str]:
    """Refuse to package smoke artefacts; report what each summary was built at.

    Returns human-readable provenance lines for the manifest. Raises SystemExit
    on a smoke artefact, because the alternative is a citable archive of numbers
    the repository does not consider results.
    """
    lines, smoke = [], []
    for rel in PROVENANCE_JSON:
        full = os.path.join(REPO, rel)
        if not os.path.exists(full):
            lines.append(f"{rel}: ABSENT")
            continue
        with open(full) as f:
            d = json.load(f)
        is_smoke = bool(d.get("smoke"))
        if is_smoke:
            smoke.append(rel)
        lines.append(f"{rel}: smoke={is_smoke} generated={d.get('generated', 'n/a')}")
        for study, meta in sorted((d.get("study_meta") or {}).items()):
            ep = (meta.get("budget") or {}).get("epochs")
            lines.append(f"    study {study}: {ep} epochs, generated {meta.get('generated')}")
    if smoke and not allow_smoke:
        raise SystemExit(
            "refusing to package smoke artefacts:\n  " + "\n  ".join(smoke) +
            "\nRe-run the drivers without --smoke, or pass --allow-smoke if you "
            "are testing this script.")
    return lines


def collect(tier: str) -> list[tuple[str, str]]:
    """(repo-relative path, description) for every file the tier includes."""
    want = ("public",) if tier == "public" else ("public", "full")
    desc_by_file: dict[str, str] = {}
    order: list[str] = []
    # Directories first, then files, so an explicitly listed file keeps its own
    # description rather than inheriting the sweep it happens to fall inside.
    for pass_files in (False, True):
        for rel, t, desc, suffixes in CONTENTS:
            if t not in want:
                continue
            if os.path.isfile(os.path.join(REPO, rel)) != pass_files:
                continue
            for f in _walk(rel, suffixes):
                if f not in desc_by_file:
                    order.append(f)
                desc_by_file[f] = desc
    return [(f, desc_by_file[f]) for f in order]


def build_manifest(files: list[tuple[str, str]], tier: str,
                   provenance: list[str]) -> str:
    from causalcgm_mt.data import WINDOW_PIPELINE_VERSION

    total = sum(os.path.getsize(os.path.join(REPO, f)) for f, _ in files)
    now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        "CausalCGM-MT release manifest",
        "=" * 60,
        f"generated                {now}",
        f"tier                     {tier}",
        f"window pipeline version  {WINDOW_PIPELINE_VERSION}",
        f"files                    {len(files)}",
        f"total size               {total / 1e6:.1f} MB",
        "",
        "PROVENANCE",
        "-" * 60,
        *provenance,
        "",
        "CONTENTS (sha256  size  path  -- description)",
        "-" * 60,
    ]
    for rel, desc in files:
        full = os.path.join(REPO, rel)
        lines.append(f"{_sha256(full)}  {os.path.getsize(full):>10}  {rel}  -- {desc}")
    lines += [
        "",
        "LICENSING",
        "-" * 60,
        "Code and model parameters: licensed under the accompanying LICENSE file.",
        "OhioT1DM is licensed separately and is NOT redistributed here. Obtain it",
        "from https://webpages.charlotte.edu/rbunescu/data/ohiot1dm/OhioT1DM-dataset.html to re-run the",
        "pipeline; every number in this archive is derived from it.",
    ]
    if tier == "full":
        lines += [
            "",
            "*** THIS IS A `full` ARCHIVE ***",
            "It contains per-window prediction dumps holding OhioT1DM CGM values,",
            "subject identifiers and timestamps. Do NOT publish it. Share only with",
            "recipients who independently hold the OhioT1DM data use agreement.",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="dist", help="output directory (default: dist)")
    ap.add_argument("--tier", choices=["public", "full"], default="public")
    ap.add_argument("--i-have-the-dua", action="store_true",
                    help="required for --tier full; confirms the recipient holds "
                         "the OhioT1DM data use agreement")
    ap.add_argument("--allow-smoke", action="store_true",
                    help="package artefacts labelled smoke (for testing only)")
    ap.add_argument("--list", action="store_true",
                    help="print what would be included and exit")
    ap.add_argument("--name", default=None, help="archive basename")
    args = ap.parse_args()

    if args.tier == "full" and not args.i_have_the_dua:
        print("--tier full ships per-window OhioT1DM CGM values and is not "
              "publishable.\nPass --i-have-the-dua to confirm the recipient holds "
              "the data use agreement.", file=sys.stderr)
        return 2

    files = collect(args.tier)
    if not files:
        print("nothing to package: run scripts/reproduce_all_paper_results.sh first",
              file=sys.stderr)
        return 2

    assert_publishable(files, args.tier)

    if args.list:
        for rel, desc in files:
            size = os.path.getsize(os.path.join(REPO, rel))
            print(f"  {size:>10}  {rel}  -- {desc}")
        print(f"\n{len(files)} file(s), "
              f"{sum(os.path.getsize(os.path.join(REPO, f)) for f, _ in files) / 1e6:.1f} MB, "
              f"tier={args.tier}")
        return 0

    provenance = check_provenance(args.allow_smoke)

    stamp = datetime.datetime.now().strftime("%Y%m%d")
    name = args.name or f"causalcgm-mt-artefacts-{args.tier}-{stamp}"
    out_dir = os.path.join(REPO, args.out)
    os.makedirs(out_dir, exist_ok=True)
    manifest = build_manifest(files, args.tier, provenance)

    manifest_path = os.path.join(out_dir, f"{name}.MANIFEST.txt")
    with open(manifest_path, "w") as f:
        f.write(manifest)

    archive = os.path.join(out_dir, f"{name}.tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(manifest_path, arcname=f"{name}/MANIFEST.txt")
        for rel, _ in files:
            tar.add(os.path.join(REPO, rel), arcname=os.path.join(name, rel))

    print(manifest)
    print(f"wrote {archive}  ({os.path.getsize(archive) / 1e6:.1f} MB)")
    print(f"wrote {manifest_path}")
    if args.tier == "full":
        print("\n*** full tier: contains OhioT1DM CGM values. Do not publish. ***")
    return 0


if __name__ == "__main__":
    sys.exit(main())
