"""Prepare the source offer mapping and selected exterior/plan metric caches."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from utils.paper_offers import (
    DATASETS_DIR, LISTING_FILES, OFFER_FILE, PHOTO_KEY, PHOTO_OUTPUTS,
    build_source_offers, load_selected_inventory,
)


METRICS = ["width", "height", "resolution", "format", "mode", "brightness", "sharpness", "clarity", "brisque"]
METRIC_COLUMNS = PHOTO_KEY + METRICS + ["metric_source"]


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as file:
        temporary = Path(file.name)
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def valid_metrics(frame: pd.DataFrame) -> pd.Series:
    numeric = frame[[column for column in METRICS if column not in ("format", "mode")]]
    return (
        pd.Series(np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1), index=frame.index)
        & frame[["format", "mode"]].notna().all(axis=1)
        & frame["width"].gt(0)
        & frame["height"].gt(0)
    )


def archived_metrics(directory: Path) -> pd.DataFrame:
    frames = []
    for market, listing_file in LISTING_FILES.items():
        tag = listing_file.split("_", 1)[0]
        frame = pd.read_parquet(directory / f"all_photo_stats_{tag}.parquet", columns=["url", *METRICS])
        parts = frame.pop("url").str.extract(
            r"/cian_project_photos/([^/]+)/([^/]+)/(\d+)/([^/?#]+)(?:[?#].*)?$"
        )
        parts.columns = PHOTO_KEY
        frame = parts.join(frame).loc[parts["deal_type"].eq(market)]
        frames.append(frame.loc[valid_metrics(frame)].assign(metric_source="archive"))
    archived = pd.concat(frames, ignore_index=True)
    if archived.duplicated(PHOTO_KEY).any():
        raise ValueError("Duplicate photo keys in the archived image statistics")
    return archived[METRIC_COLUMNS]


def read_photo_batch(paths: list[str], args) -> dict[str, bytes]:
    if args.photo_root:
        return {path: (args.photo_root / path).read_bytes() for path in paths}

    tar_command = shlex.join([
        "tar", "--directory", args.remote_root, "--create", "--file", "-",
        "--no-recursion", "--null", "--files-from", "-",
    ])
    command = [
        "ssh", "-F", str(args.ssh_config), "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=yes", "-o", "UpdateHostKeys=no", args.ssh_target, tar_command,
    ]
    request = b"".join(path.encode() + b"\0" for path in paths)
    result = subprocess.run(command, input=request, capture_output=True, timeout=180)
    if result.returncode:
        raise RuntimeError(f"Photo archive request failed: {result.stderr.decode(errors='replace').strip()}")
    with tarfile.open(fileobj=BytesIO(result.stdout), mode="r:") as archive:
        images = {member.name: archive.extractfile(member).read() for member in archive if member.isfile()}
    if set(images) != set(paths):
        missing = sorted(set(paths) - images.keys())
        raise ValueError(f"Photo archive returned an incomplete or unexpected batch. Missing: {missing[:3]}")
    return images


def measure_missing(missing: pd.DataFrame, args) -> pd.DataFrame:
    if not (args.photo_root or args.ssh_target):
        raise FileNotFoundError(f"{len(missing):,} photos need metrics. Supply --photo-root or --ssh-target.")
    from utils.image_quality import image_statistics

    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for start in range(0, len(missing), 128):
            keys = list(missing.iloc[start:start + 128].itertuples(index=False, name=None))
            if any(part in ("", ".", "..") or "/" in part or "\0" in part for key in keys for part in key):
                raise ValueError("Photo keys must be individual path components")
            paths = ["/".join(key) for key in keys]
            images = read_photo_batch(paths, args)
            stats = pool.map(image_statistics, (images[path] for path in paths), chunksize=4)
            for key, values in zip(keys, stats, strict=True):
                records.append({**dict(zip(PHOTO_KEY, key)), **values, "metric_source": "computed"})
            print(f"Measured {len(records):,}/{len(missing):,} photos", flush=True)
    frame = pd.DataFrame(records, columns=METRIC_COLUMNS)
    succeeded = valid_metrics(frame)
    if not succeeded.all():
        failed = frame.loc[~succeeded, PHOTO_KEY]
        raise ValueError(f"Image metrics failed for {len(failed):,} photos. Example: {failed.iloc[0].to_dict()}")
    return frame


def prepare_photo_stats(root: Path, args) -> None:
    inventory = load_selected_inventory(root)
    targets = inventory.loc[inventory["photo_type"].isin(["exterior", "plan"]), PHOTO_KEY + ["photo_type"]]
    photo_dir = root / PHOTO_OUTPUTS
    available = []
    for kind in ("exterior", "plan"):
        path = photo_dir / f"{kind}_photo_stats.parquet"
        if path.is_file():
            frame = pd.read_parquet(path, columns=METRIC_COLUMNS)
            available.append(frame.loc[valid_metrics(frame)])
    if args.archived_stats_dir:
        available.append(archived_metrics(args.archived_stats_dir))

    if available:
        lookup = pd.concat(available, ignore_index=True).drop_duplicates(PHOTO_KEY)
        matched = targets.merge(lookup, on=PHOTO_KEY, how="left", validate="one_to_one")
        missing = matched.loc[matched["metric_source"].isna(), PHOTO_KEY]
        cached = matched.loc[matched["metric_source"].notna(), METRIC_COLUMNS]
    else:
        missing = targets[PHOTO_KEY]
        cached = pd.DataFrame(columns=METRIC_COLUMNS)

    print(f"Selected {len(targets):,} exterior and plan photos. Reusing {len(cached):,} metric rows.", flush=True)
    if not missing.empty:
        measured = measure_missing(missing, args)
        cached = pd.concat([cached, measured], ignore_index=True) if not cached.empty else measured

    complete = targets.merge(cached, on=PHOTO_KEY, how="left", validate="one_to_one")
    if complete["metric_source"].isna().any():
        raise ValueError("Incomplete exterior or floor-plan metric coverage")
    for kind, frame in complete.groupby("photo_type", sort=True):
        path = photo_dir / f"{kind}_photo_stats.parquet"
        write_parquet(frame[METRIC_COLUMNS], path)
        print(f"{path.name}: {len(frame):,} rows, {frame['metric_source'].value_counts().to_dict()}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT.parent / "source" / "eda" / "inputs")
    parser.add_argument("--offers-only", action="store_true")
    parser.add_argument("--archived-stats-dir", type=Path)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--photo-root", type=Path)
    sources.add_argument("--ssh-target")
    parser.add_argument("--ssh-config", type=Path, default=Path.home() / ".ssh" / "config")
    parser.add_argument("--remote-root", default="/srv/images/cian_project_photos")
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    offers_path = ROOT / DATASETS_DIR / OFFER_FILE
    if args.source_dir.is_dir() or not offers_path.is_file():
        write_parquet(build_source_offers(args.source_dir), offers_path)
    if not args.offers_only:
        prepare_photo_stats(ROOT, args)


if __name__ == "__main__":
    main()
