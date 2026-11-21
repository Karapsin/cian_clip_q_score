import asyncio
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from utils.parquet_io import write_stage_outputs
from utils.pipeline import prepare_photo_frame, run_pipeline, selected_stages

STAGES_TO_RUN = [
    "iep",
    "interior_scores",
    "photo_quality",
    "exterior_plan_scores",
]
FRESH_RUN = False

PROJECT_ROOT = Path(__file__).resolve().parent
INPUTS_PATH = PROJECT_ROOT / "inputs"
OUTPUTS_PATH = PROJECT_ROOT / "outputs"
PHOTO_BASE_URL = "http://192.168.31.149:8765/"
REMOTE_ROOT = "/srv/images/cian_project_photos"
SSH_TARGET = "kardinal@192.168.31.149"
SSH_AUTH_SOCK = "/run/user/1000/rpi-agent.sock"
MODEL_ID = "openai/clip-vit-base-patch32"

PROPERTIES_PER_BATCH = 20
PREFETCH = 2
MAX_CONNECTIONS = 64
CLIP_BATCH_SIZE = 64
METRIC_WORKERS = 16
SEED = 108


async def main():
    if "SSH_AUTH_SOCK" not in os.environ and Path(SSH_AUTH_SOCK).exists():
        os.environ["SSH_AUTH_SOCK"] = SSH_AUTH_SOCK

    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stages = selected_stages(STAGES_TO_RUN)
    OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")
    print(f"Stages: {stages}")
    print(f"Fresh run: {FRESH_RUN}")

    with (INPUTS_PATH / "prompts.json").open() as prompt_file:
        prompts = json.load(prompt_file)

    photo_df = prepare_photo_frame(PROJECT_ROOT, stages, REMOTE_ROOT, SSH_TARGET)
    print(f"Photos to download: {len(photo_df):,}")
    print(photo_df["deal_type"].value_counts().rename("photos").to_string())

    parts_dir = await run_pipeline(
        photo_df,
        prompts,
        stages,
        PROJECT_ROOT,
        PHOTO_BASE_URL,
        MODEL_ID,
        device,
        properties_per_batch=PROPERTIES_PER_BATCH,
        prefetch=PREFETCH,
        max_connections=MAX_CONNECTIONS,
        clip_batch_size=CLIP_BATCH_SIZE,
        metric_workers=METRIC_WORKERS,
        fresh_run=FRESH_RUN,
    )

    if "iep" in stages:
        iep_files = write_stage_outputs(parts_dir, OUTPUTS_PATH, "iep")
        iep_result = pd.read_parquet(iep_files[0])
        print(iep_files[0])
        print(iep_result.head().to_string(index=False))

    if "interior_scores" in stages:
        interior_files = write_stage_outputs(parts_dir, OUTPUTS_PATH, "interior_scores")
        interior_result = pd.read_parquet(interior_files[0])
        print(interior_files[0])
        print(interior_result.head().to_string(index=False))

    if "photo_quality" in stages:
        quality_files = write_stage_outputs(parts_dir, OUTPUTS_PATH, "photo_quality")
        quality_result = pd.read_parquet(quality_files[0])
        print(quality_files[0])
        print(quality_result.head().to_string(index=False))

    if "exterior_plan_scores" in stages:
        exterior_plan_files = write_stage_outputs(parts_dir, OUTPUTS_PATH, "exterior_plan_scores")
        for output_file in exterior_plan_files:
            print(output_file)
            print(pd.read_parquet(output_file).head().to_string(index=False))

    print(f"Batch parts retained: {parts_dir}")


if __name__ == "__main__":
    asyncio.run(main())
