#!/usr/bin/env python3
"""
prepare_cyclegan_dataset_full_single_file.py

One-file, PyCharm-ready script to build CycleGAN-style dataset folders.
- Uses an inline Python dict CONFIG (no JSON file required).
- Optional --debug flag generates a tiny fake dataset to test end-to-end.
- Prints a concise summary (counts per split) after each run.
- Includes self-tests: --selftest

Outputs layout:
  TargetFolder/
    trainA/  valA/  testA/
    trainB/  valB/  testB/
    _reports/
      summary.json
      summary.csv
      run.log

Author: (you)
"""

from __future__ import annotations
import argparse
import csv
import json
import logging
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time
from typing import Dict, List, Tuple

# ------------------------ USER CONFIGURATION ------------------------ #
# Replace paths and percentages below with your real data when not using --debug.
CONFIG: Dict[str, object] = {
    "target_folder": r"E:\Mahendra\cyclegan_Magnetite_target",   # e.g., r"E:/cyclegan_target"
    "domain_a": {
        "exp_sets": [
            {"path": r"E:\Mahendra\Experimental\Exp_images", "take_percent": 60, "name": "Mag_exp_"},
#            {"path": r"E:\Sumit\Patterns_for_training\Carburized\Experimental\Carburized_exp_images", "take_percent": 40, "name": "Carb_"},
#            {"path": r"E:\Sumit\Patterns_for_training\Normalized\Experimental\Normalized_exp_images", "take_percent": 40, "name": "Norm_"},

        ],
        "split": {"train": 90, "val": 5, "test": 5},  # sums to 100
    },
    "domain_b": {
        "simulated_roots": [r"E:\Mahendra\Simuated\Sim_images",
                            #r"E:\Sumit\Patterns_for_training\Carburized\Simulated\Carburized_sim_images",
                            #r"E:\Sumit\Patterns_for_training\Normalized\Simulated\Normalized_sim_images",
#                            r"E:\Sumit\Patterns_for_training\Normalized\Simulated_synthetic",
                            ],
        "take_percent": 40,
        "split": {"train": 90, "val": 5, "test": 5},  # sums to 100
    },
    "random_seed": 42,
    "operation": "copy",  # "copy" | "move" | "symlink"
    "allowed_extensions": [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"],
    "recursive": True,
}
# ------------------------------------------------------------------- #

# ------------------------------- Logging ------------------------------------ #

def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("prep")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger

# ------------------------------- Utilities ---------------------------------- #

DEFAULT_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]

def is_image_file(p: Path, allowed_exts: List[str]) -> bool:
    return p.is_file() and p.suffix.lower() in allowed_exts

def scan_images(root: Path, allowed_exts: List[str], recursive: bool) -> List[Path]:
    if not root.exists():
        return []
    if recursive:
        return [p for p in root.rglob("*") if is_image_file(p, allowed_exts)]
    return [p for p in root.glob("*") if is_image_file(p, allowed_exts)]

def choose_n(items: List[Path], n: int, rng: random.Random) -> List[Path]:
    if n <= 0:
        return []
    if n >= len(items):
        return list(items)
    return rng.sample(items, n)

def ensure_sums_to_100(name: str, ratios: Dict[str, float]):
    total = sum(ratios.values())
    if not math.isclose(total, 100.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError(f"{name} must sum to 100; got {total}")

def safe_copy_or_move(src: Path, dst_dir: Path, op: str) -> Path:
    """Copy/move/symlink `src` into `dst_dir`. If a name collision exists, append _{k}."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    base = src.stem
    ext = src.suffix
    candidate = dst_dir / (base + ext)
    k = 1
    while candidate.exists():
        candidate = dst_dir / (f"{base}_{k}{ext}")
        k += 1

    if op == "copy":
        shutil.copy2(src, candidate)
    elif op == "move":
        shutil.move(str(src), str(candidate))
    elif op == "symlink":
        # Relative symlink; if OS denies, fallback to copy
        rel = os.path.relpath(src, start=dst_dir)
        try:
            candidate.symlink_to(rel)
        except OSError:
            shutil.copy2(src, candidate)
    else:
        raise ValueError(f"Unsupported op: {op}")
    return candidate

def take_percent_round_down(total: int, percent: float) -> int:
    return int(math.floor((percent / 100.0) * total))

def split_by_ratios(items: List[Path], ratios: Dict[str, float], rng: random.Random) -> Dict[str, List[Path]]:
    """Split list into buckets according to percentage ratios (summing to 100)."""
    ensure_sums_to_100("Split ratios", ratios)

    n = len(items)
    idxs = list(range(n))
    rng.shuffle(idxs)

    desired = {k: (ratios[k] / 100.0) * n for k in ratios}
    floors = {k: int(math.floor(desired[k])) for k in ratios}
    allocated = sum(floors.values())
    remainder = n - allocated
    fracs = sorted(((k, desired[k] - floors[k]) for k in ratios), key=lambda x: x[1], reverse=True)
    sizes = floors.copy()
    for i in range(remainder):
        sizes[fracs[i][0]] += 1

    out: Dict[str, List[Path]] = {k: [] for k in ratios}
    pos = 0
    for split_name in ratios:
        take = sizes[split_name]
        take_idxs = idxs[pos:pos+take]
        out[split_name] = [items[j] for j in take_idxs]
        pos += take
    return out

# -------------------------- Core Build Functions ---------------------------- #

def build_domain_a(
    logger: logging.Logger,
    exp_sets: List[Dict[str, object]],
    allowed_exts: List[str],
    recursive: bool,
    rng: random.Random,
) -> Tuple[List[Path], Dict[str, object]]:
    domain_a: List[Path] = []
    per_set_report = []

    for i, d in enumerate(exp_sets):
        root = Path(str(d["path"]))
        take_percent = float(d["take_percent"])
        name = str(d.get("name", f"exp_set_{i+1}"))

        all_imgs = scan_images(root, allowed_exts, recursive)
        n_all = len(all_imgs)
        n_take = take_percent_round_down(n_all, take_percent)
        chosen = choose_n(all_imgs, n_take, rng)
        domain_a.extend(chosen)

        logger.info(f"[Domain A] {name}: found={n_all}, take_percent={take_percent} -> chosen={len(chosen)}")

        per_set_report.append({
            "name": name, "path": str(root),
            "found": n_all, "take_percent": take_percent, "chosen": len(chosen)
        })

    return domain_a, {"per_set": per_set_report}

def build_domain_b(
    logger: logging.Logger,
    simulated_roots: List[str],
    take_percent: float,
    allowed_exts: List[str],
    recursive: bool,
    rng: random.Random,
) -> Tuple[List[Path], Dict[str, object]]:
    pool: List[Path] = []
    per_root = []
    for root in simulated_roots:
        rootp = Path(root)
        imgs = scan_images(rootp, allowed_exts, recursive)
        pool.extend(imgs)
        per_root.append({"path": str(rootp), "found": len(imgs)})

    n_all = len(pool)
    n_take = take_percent_round_down(n_all, take_percent)
    chosen = choose_n(pool, n_take, rng)
    logger.info(f"[Domain B] simulated_pool: found={n_all}, take_percent={take_percent} -> chosen={len(chosen)}")

    return chosen, {"pool_found": n_all, "take_percent": take_percent, "per_root": per_root}

def materialize_split(
    logger: logging.Logger,
    split_map: Dict[str, List[Path]],
    target_folder: Path,
    suffix: str,
    op: str
) -> Dict[str, int]:
    counts = {}
    for split_name, files in split_map.items():
        outdir = target_folder / f"{split_name}{suffix}"
        c = 0
        for src in files:
            safe_copy_or_move(src, outdir, op)
            c += 1
        counts[f"{split_name}{suffix}"] = c
        logger.info(f"Wrote {c} files to {outdir}")
    return counts

def write_reports(report_dir: Path, summary: Dict[str, object]):
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    counts = summary.get("counts", {})
    with open(report_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bucket", "count"])
        for k, v in counts.items():
            w.writerow([k, v])

# ------------------------------- Orchestrator -------------------------------- #

def run_with_config(config: Dict[str, object]) -> None:
    target_folder = Path(str(config["target_folder"]))
    target_folder.mkdir(parents=True, exist_ok=True)

    log_file = target_folder / "_reports" / "run.log"
    logger = setup_logger(log_file)
    t0 = time.time()

    rng_seed = int(config.get("random_seed", 42))
    rng = random.Random(rng_seed)
    op = str(config.get("operation", "copy")).lower()
    allowed_exts = [e.lower() for e in config.get("allowed_extensions", DEFAULT_EXTS)]
    recursive = bool(config.get("recursive", True))

    dom_a = config["domain_a"]
    dom_b = config["domain_b"]

    exp_sets = dom_a["exp_sets"]
    for d in exp_sets:
        p = float(d["take_percent"])
        if p < 0 or p > 100:
            raise ValueError(f"Domain A set percentage must be within [0,100], got {p}")

    domain_a_files, a_report = build_domain_a(
        logger=logger,
        exp_sets=exp_sets,
        allowed_exts=allowed_exts,
        recursive=recursive,
        rng=rng,
    )

    take_b = float(dom_b["take_percent"])
    if take_b < 0 or take_b > 100:
        raise ValueError(f"Domain B percentage must be within [0,100], got {take_b}")

    domain_b_files, b_report = build_domain_b(
        logger=logger,
        simulated_roots=[str(p) for p in dom_b["simulated_roots"]],
        take_percent=take_b,
        allowed_exts=allowed_exts,
        recursive=recursive,
        rng=rng,
    )

    split_a = dom_a["split"]
    split_b = dom_b["split"]

    required_keys = {"train", "val", "test"}
    if set(split_a.keys()) != required_keys or set(split_b.keys()) != required_keys:
        raise ValueError("Split dicts must have exactly keys: train, val, test")
    ensure_sums_to_100("Domain A split", {k: float(split_a[k]) for k in split_a})
    ensure_sums_to_100("Domain B split", {k: float(split_b[k]) for k in split_b})

    random.Random(rng_seed).shuffle(domain_a_files)
    random.Random(rng_seed + 1).shuffle(domain_b_files)
    split_map_a = split_by_ratios(domain_a_files, {k: float(split_a[k]) for k in split_a}, rng)
    split_map_b = split_by_ratios(domain_b_files, {k: float(split_b[k]) for k in split_b}, rng)

    counts_a = materialize_split(logger, split_map_a, target_folder, suffix="A", op=op)
    counts_b = materialize_split(logger, split_map_b, target_folder, suffix="B", op=op)

    counts: Dict[str, int] = {}
    counts.update(counts_a)
    counts.update(counts_b)
    counts["DomainA_total"] = len(domain_a_files)
    counts["DomainB_total"] = len(domain_b_files)

    summary = {
        "target_folder": str(target_folder),
        "random_seed": rng_seed,
        "operation": op,
        "allowed_extensions": allowed_exts,
        "recursive": recursive,
        "domain_a": a_report,
        "domain_b": b_report,
        "splits": {"A": {k: float(split_a[k]) for k in split_a}, "B": {k: float(split_b[k]) for k in split_b}},
        "counts": counts,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    write_reports(target_folder / "_reports", summary)
    logger.info("All done.")
    logger.info(json.dumps(summary, indent=2))

# ------------------------------- Debug Mode ---------------------------------- #

_ONE_BY_ONE_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108020000009077"
    "3dae0000000a49444154789c6360000002000154a24f0b0000000049454e44"
    "ae426082"
)

def _write_fake_png(dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        f.write(_ONE_BY_ONE_PNG)

def make_debug_dataset(root: Path) -> Dict[str, object]:
    exp1 = root / "exp_set1"
    exp2 = root / "exp_set2"
    exp3 = root / "exp_set3"
    simr = root / "simulated"

    for i in range(5):
        _write_fake_png(exp1 / f"img_{i:03d}.png")
    for i in range(7):
        _write_fake_png(exp2 / f"img_{i:03d}.png")
    for i in range(4):
        _write_fake_png(exp3 / f"img_{i:03d}.png")
    for i in range(10):
        _write_fake_png(simr / f"sim_{i:03d}.png")

    cfg = {
        "target_folder": str(root / "TargetFolder"),
        "domain_a": {
            "exp_sets": [
                {"path": str(exp1), "take_percent": 60, "name": "set1"},
                {"path": str(exp2), "take_percent": 50, "name": "set2"},
                {"path": str(exp3), "take_percent": 100, "name": "set3"},
            ],
            "split": {"train": 70, "val": 15, "test": 15},
        },
        "domain_b": {
            "simulated_roots": [str(simr)],
            "take_percent": 50,
            "split": {"train": 60, "val": 20, "test": 20},
        },
        "random_seed": 123,
        "operation": "copy",
        "allowed_extensions": [".png"],
        "recursive": True,
    }
    return cfg

# ------------------------------- Self Tests --------------------------------- #

def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)

def run_selftests() -> None:
    # Test 1: Debug end-to-end
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = make_debug_dataset(root)
        run_with_config(cfg)
        s = json.loads((Path(cfg["target_folder"]) / "_reports" / "summary.json").read_text(encoding="utf-8"))
        _assert(s["counts"]["DomainA_total"] == 10, "Expected DomainA_total=10")
        _assert(s["counts"]["DomainB_total"] == 5, "Expected DomainB_total=5")
        for b in ["trainA", "valA", "testA", "trainB", "valB", "testB"]:
            _assert((Path(cfg["target_folder"]) / b).exists(), f"Missing bucket: {b}")

    # Test 2: Custom tiny config
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        A1 = root / "A1"; A2 = root / "A2"; B1 = root / "B1"
        for i in range(3): _write_fake_png(A1 / f"a1_{i}.png")
        for i in range(2): _write_fake_png(A2 / f"a2_{i}.png")
        for i in range(3): _write_fake_png(B1 / f"b1_{i}.png")
        cfg = {
            "target_folder": str(root / "out"),
            "domain_a": {
                "exp_sets": [
                    {"path": str(A1), "take_percent": 100, "name": "A1"},
                    {"path": str(A2), "take_percent": 100, "name": "A2"},
                ],
                "split": {"train": 50, "val": 25, "test": 25},
            },
            "domain_b": {
                "simulated_roots": [str(B1)],
                "take_percent": 67,  # floor(3*0.67)=2
                "split": {"train": 50, "val": 25, "test": 25},
            },
            "random_seed": 1,
            "operation": "copy",
            "allowed_extensions": [".png"],
            "recursive": True,
        }
        run_with_config(cfg)
        s = json.loads((Path(cfg["target_folder"]) / "_reports" / "summary.json").read_text(encoding="utf-8"))
        _assert(s["counts"]["DomainA_total"] == 5, "Domain A total should be 5 (3+2)")
        _assert(s["counts"]["DomainB_total"] == 2, "Domain B total should be 2 (floor(3*0.67))")
        sumA = s["counts"]["trainA"] + s["counts"]["valA"] + s["counts"]["testA"]
        sumB = s["counts"]["trainB"] + s["counts"]["valB"] + s["counts"]["testB"]
        _assert(sumA == s["counts"]["DomainA_total"], "A split must sum to total")
        _assert(sumB == s["counts"]["DomainB_total"], "B split must sum to total")

    print("All selftests passed.")

# ------------------------------- Summary Print ------------------------------ #

def print_quick_summary(target_folder: Path) -> None:
    summary_path = target_folder / "_reports" / "summary.json"
    if not summary_path.exists():
        print(f"[summary] Not found: {summary_path}")
        return
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    counts = s.get("counts", {})
    def g(k: str) -> int: return int(counts.get(k, 0))
    lines = [
        "\n=== CycleGAN Prep Summary ===",
        f"Target: {target_folder}",
        f"DomainA total: {g('DomainA_total')}\t(trainA={g('trainA')}, valA={g('valA')}, testA={g('testA')})",
        f"DomainB total: {g('DomainB_total')}\t(trainB={g('trainB')}, valB={g('valB')}, testB={g('testB')})",
        f"Elapsed (s): {s.get('elapsed_sec')}",
        "================================\n",
    ]
    print("\n".join(lines))

# --------------------------------- CLI -------------------------------------- #

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Prepare CycleGAN dataset folders from inline CONFIG. "
            "Default (no flags) uses CONFIG; add --debug to run a tiny fake dataset; "
            "add --selftest to run built-in tests."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--debug", action="store_true", help="Run with a small auto-generated fake dataset for testing.")
    p.add_argument("--selftest", action="store_true", help="Run built-in tests and exit.")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()

    if args.selftest:
        run_selftests()
        return

    if args.debug:
        debug_root = Path("./_debug_run_single").resolve()
        if debug_root.exists():
            shutil.rmtree(debug_root)
        debug_root.mkdir(parents=True, exist_ok=True)
        cfg = make_debug_dataset(debug_root)
        run_with_config(cfg)
        print_quick_summary(Path(cfg["target_folder"]))
        print(f"Debug run complete. See: {cfg['target_folder']}")
        return

    # Default: run with inline CONFIG
    run_with_config(CONFIG)
    print_quick_summary(Path(str(CONFIG["target_folder"])) )


if __name__ == "__main__":
    main()
