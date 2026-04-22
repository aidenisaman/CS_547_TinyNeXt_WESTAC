import argparse
import csv
import os
import random
import stat
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class SplitStats:
    rows: int = 0
    linked_or_copied: int = 0
    skipped_existing: int = 0
    missing_sources: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert mini-ImageNet CSV manifests (filename,label) to ImageFolder layout."
    )
    parser.add_argument("--image-root", type=Path, required=True, help="Root directory containing image files")
    parser.add_argument("--train-csv", type=Path, required=True, help="Path to train.csv")
    parser.add_argument("--val-csv", type=Path, required=True, help="Path to val.csv")
    parser.add_argument("--test-csv", type=Path, default=None, help="Optional path to test.csv")
    parser.add_argument(
        "--output-root", type=Path, required=True, help="Destination root for ImageFolder data"
    )
    parser.add_argument("--train-split", type=str, default="train", help="Train split directory name")
    parser.add_argument("--val-split", type=str, default="val", help="Validation split directory name")
    parser.add_argument(
        "--merge-csv-splits",
        action="store_true",
        help="Merge CSV manifests and create per-class train/val split for closed-set classification",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Per-class ratio used for train split when --merge-csv-splits is enabled",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --merge-csv-splits is enabled",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hardlink",
        choices=["hardlink", "copy", "symlink"],
        help="Materialization mode for split files",
    )
    parser.add_argument(
        "--expected-classes",
        type=int,
        default=-1,
        help="Expected total class count across selected splits; set -1 to disable",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing output train/val split directories before preparing",
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Fail if any source images are missing",
    )
    parser.add_argument(
        "--no-fallback-copy",
        action="store_true",
        help="When using hardlink/symlink mode, do not fall back to copy on failure",
    )
    return parser.parse_args()


def read_manifest(csv_path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "filename" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(f"CSV must contain filename,label headers: {csv_path}")
        for row in reader:
            filename = (row.get("filename") or "").strip()
            label = (row.get("label") or "").strip()
            if not filename or not label:
                continue
            rows.append((filename, label))
    return rows


def materialize_file(src: Path, dst: Path, mode: str, fallback_copy: bool) -> str:
    if dst.exists():
        return "exists"

    if mode == "copy":
        shutil.copy2(src, dst)
        return "copied"

    if mode == "symlink":
        try:
            os.symlink(src, dst)
            return "symlinked"
        except OSError:
            if fallback_copy:
                shutil.copy2(src, dst)
                return "copied"
            raise

    try:
        os.link(src, dst)
        return "hardlinked"
    except FileExistsError:
        return "exists"
    except OSError:
        if fallback_copy:
            try:
                shutil.copy2(src, dst)
                return "copied"
            except shutil.SameFileError:
                return "exists"
        raise


def _remove_readonly(func, path, exc_info) -> None:
    del exc_info
    os.chmod(path, stat.S_IWRITE)
    func(path)


def safe_rmtree(path: Path, retries: int = 6, delay_sec: float = 0.25) -> None:
    if not path.exists():
        return
    last_error: Optional[OSError] = None
    for _ in range(retries):
        try:
            shutil.rmtree(path, onerror=_remove_readonly)
            return
        except OSError as exc:
            last_error = exc
            time.sleep(delay_sec)
    if last_error is not None:
        raise last_error


def build_merged_split_rows(
    rows: List[Tuple[str, str]], train_ratio: float, seed: int
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train-ratio must be in (0, 1), got {train_ratio}")

    by_label: Dict[str, List[str]] = {}
    for filename, label in rows:
        by_label.setdefault(label, []).append(filename)

    rnd = random.Random(seed)
    train_rows: List[Tuple[str, str]] = []
    val_rows: List[Tuple[str, str]] = []

    for label, filenames in by_label.items():
        unique_filenames = list(dict.fromkeys(filenames))
        rnd.shuffle(unique_filenames)
        count = len(unique_filenames)

        if count == 1:
            train_count = 1
        else:
            train_count = int(count * train_ratio)
            train_count = max(1, min(count - 1, train_count))

        train_files = unique_filenames[:train_count]
        val_files = unique_filenames[train_count:]

        train_rows.extend((name, label) for name in train_files)
        val_rows.extend((name, label) for name in val_files)

    return train_rows, val_rows


def process_split(
    manifest_rows: List[Tuple[str, str]],
    image_root: Path,
    split_root: Path,
    mode: str,
    fallback_copy: bool,
) -> Tuple[SplitStats, Dict[str, int]]:
    stats = SplitStats()
    class_counts: Dict[str, int] = {}

    for filename, label in manifest_rows:
        stats.rows += 1
        src = image_root / filename
        if not src.exists():
            stats.missing_sources += 1
            continue

        class_dir = split_root / label
        class_dir.mkdir(parents=True, exist_ok=True)

        dst = class_dir / filename
        result = materialize_file(src, dst, mode=mode, fallback_copy=fallback_copy)
        if result == "exists":
            stats.skipped_existing += 1
        else:
            stats.linked_or_copied += 1
        class_counts[label] = class_counts.get(label, 0) + 1

    return stats, class_counts


def main() -> int:
    args = parse_args()

    image_root = args.image_root.resolve()
    train_csv = args.train_csv.resolve()
    val_csv = args.val_csv.resolve()
    test_csv: Optional[Path] = args.test_csv.resolve() if args.test_csv else None
    output_root = args.output_root.resolve()

    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found: {image_root}")
    if not train_csv.exists():
        raise FileNotFoundError(f"train.csv not found: {train_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"val.csv not found: {val_csv}")
    if args.merge_csv_splits and test_csv is not None and not test_csv.exists():
        raise FileNotFoundError(f"test.csv not found: {test_csv}")

    train_root = output_root / args.train_split
    val_root = output_root / args.val_split

    if args.clean:
        if train_root.exists():
            safe_rmtree(train_root)
        if val_root.exists():
            safe_rmtree(val_root)

    train_rows = read_manifest(train_csv)
    val_rows = read_manifest(val_csv)
    test_rows = read_manifest(test_csv) if test_csv is not None else []

    if args.merge_csv_splits:
        merged_rows = train_rows + val_rows + test_rows
        train_rows, val_rows = build_merged_split_rows(
            rows=merged_rows,
            train_ratio=args.train_ratio,
            seed=args.seed,
        )

    fallback_copy = not args.no_fallback_copy

    train_stats, train_counts = process_split(
        manifest_rows=train_rows,
        image_root=image_root,
        split_root=train_root,
        mode=args.mode,
        fallback_copy=fallback_copy,
    )
    val_stats, val_counts = process_split(
        manifest_rows=val_rows,
        image_root=image_root,
        split_root=val_root,
        mode=args.mode,
        fallback_copy=fallback_copy,
    )

    all_classes = sorted(set(train_counts.keys()) | set(val_counts.keys()))
    classes_path = output_root / "classes.txt"
    classes_path.parent.mkdir(parents=True, exist_ok=True)
    classes_path.write_text("\n".join(all_classes) + "\n", encoding="utf-8")

    print("mini-ImageNet folder preparation complete")
    print(f"  output root: {output_root}")
    print(f"  class count: {len(all_classes)}")
    print(f"  classes file: {classes_path}")
    print("  train split:")
    print(f"    rows: {train_stats.rows}")
    print(f"    linked/copied: {train_stats.linked_or_copied}")
    print(f"    skipped existing: {train_stats.skipped_existing}")
    print(f"    missing sources: {train_stats.missing_sources}")
    print("  val split:")
    print(f"    rows: {val_stats.rows}")
    print(f"    linked/copied: {val_stats.linked_or_copied}")
    print(f"    skipped existing: {val_stats.skipped_existing}")
    print(f"    missing sources: {val_stats.missing_sources}")
    if args.merge_csv_splits:
        print(f"  merged split mode: enabled (train-ratio={args.train_ratio}, seed={args.seed})")

    missing_total = train_stats.missing_sources + val_stats.missing_sources
    if args.strict_missing and missing_total > 0:
        raise RuntimeError(f"Missing source files detected: {missing_total}")

    if args.expected_classes > 0 and len(all_classes) != args.expected_classes:
        raise RuntimeError(
            f"Class count mismatch: expected {args.expected_classes}, got {len(all_classes)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
