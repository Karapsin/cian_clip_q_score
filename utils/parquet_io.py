import hashlib
import shutil
from pathlib import Path

import pandas as pd

from .image_quality import METRIC_COLUMNS
from .photo_io import KEY

ALL_STAGES = ["iep", "interior_scores", "photo_quality", "exterior_plan_scores"]
IEP_CLASSES = ["interior", "view", "street", "floor_plan", "common"]
IEP_SIMILARITY_COLUMNS = [f"{name}_similarity" for name in IEP_CLASSES]
IEP_PROBABILITY_COLUMNS = [f"{name}_probability" for name in IEP_CLASSES]
IEP_OUTPUT_COLUMNS = KEY + IEP_SIMILARITY_COLUMNS + IEP_PROBABILITY_COLUMNS + ["label", "prediction"]
INTERIOR_SCORE_COLUMNS = [
    "old_pos_similarity", "old_mid_similarity", "old_neg_similarity",
    "recommended_pos_similarity", "recommended_mid_similarity", "recommended_neg_similarity",
    "recommended_unfinished_similarity",
]
EXTERIOR_PLAN_SCORE_COLUMNS = ["low_similarity", "mid_similarity", "high_similarity"]
BATCH_COLUMNS = KEY + IEP_OUTPUT_COLUMNS[len(KEY):] + INTERIOR_SCORE_COLUMNS + METRIC_COLUMNS + EXTERIOR_PLAN_SCORE_COLUMNS


def parts_directory(project_root: Path, stages: list[str], photo_df: pd.DataFrame) -> Path:
    digest = hashlib.sha256()
    for path in photo_df["path"]:
        digest.update(path.encode("utf-8"))
        digest.update(b"\n")
    run_name = "-".join(stages) + "-" + digest.hexdigest()[:12]
    return project_root / ".parts" / run_name


def batch_path(parts_dir: Path, batch_index: int) -> Path:
    return parts_dir / f"batch_{batch_index:06d}.parquet"


def completed_batches(parts_dir: Path) -> set[int]:
    if not parts_dir.exists():
        return set()
    return {int(path.stem.split("_")[-1]) for path in parts_dir.glob("batch_*.parquet")}


def save_batch(frame: pd.DataFrame, parts_dir: Path, batch_index: int) -> None:
    parts_dir.mkdir(parents=True, exist_ok=True)
    output_path = batch_path(parts_dir, batch_index)
    temporary_path = output_path.with_suffix(".tmp")
    frame.reindex(columns=BATCH_COLUMNS).to_parquet(temporary_path, index=False)
    temporary_path.replace(output_path)


def read_parts(parts_dir: Path, columns: list[str], filters=None) -> pd.DataFrame:
    read_columns = list(dict.fromkeys(columns + ["label"]))
    if not any(parts_dir.glob("batch_*.parquet")):
        return pd.DataFrame(columns=columns)
    frame = pd.read_parquet(parts_dir, columns=read_columns, filters=filters)
    return frame[columns]


def write_output(parts_dir: Path, output_path: Path, columns: list[str], filters=None) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = read_parts(parts_dir, columns, filters)
    if output_path.name == "interior_photo_stats.parquet":
        for column in ["width", "height", "resolution"]:
            frame[column] = frame[column].astype("Int64")
    frame.to_parquet(output_path, index=False)
    return output_path


def write_stage_outputs(parts_dir: Path, outputs_dir: Path, stage: str) -> list[Path]:
    if stage == "iep":
        return [write_output(parts_dir, outputs_dir / "photo_classifications.parquet", IEP_OUTPUT_COLUMNS)]
    if stage == "interior_scores":
        return [write_output(
            parts_dir,
            outputs_dir / "interior_photo_scores.parquet",
            KEY + INTERIOR_SCORE_COLUMNS,
            [("label", "=", "interior")],
        )]
    if stage == "photo_quality":
        return [write_output(
            parts_dir,
            outputs_dir / "interior_photo_stats.parquet",
            KEY + METRIC_COLUMNS,
            [("label", "=", "interior")],
        )]
    if stage == "exterior_plan_scores":
        return [
            write_output(
                parts_dir,
                outputs_dir / "exterior_photo_scores.parquet",
                KEY + EXTERIOR_PLAN_SCORE_COLUMNS,
                [("label", "in", ["street", "view", "common"])],
            ),
            write_output(
                parts_dir,
                outputs_dir / "plan_photo_scores.parquet",
                KEY + EXTERIOR_PLAN_SCORE_COLUMNS,
                [("label", "=", "floor_plan")],
            ),
        ]
    raise ValueError(f"Unknown stage: {stage}")


def remove_parts(parts_dir: Path) -> None:
    if parts_dir.exists():
        shutil.rmtree(parts_dir)


def write_final_outputs(parts_dir: Path, outputs_dir: Path, stages: list[str]) -> dict[str, list[Path]]:
    return {stage: write_stage_outputs(parts_dir, outputs_dir, stage) for stage in stages}
