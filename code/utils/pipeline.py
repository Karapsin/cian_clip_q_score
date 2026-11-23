import asyncio
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from io import BytesIO
from multiprocessing import get_context
from pathlib import Path

import aiohttp
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

from .clip_utils import build_prototypes, embed_images, embed_prompts, load_model, similarities
from .image_quality import METRIC_COLUMNS, calculate_metrics
from .parquet_io import ALL_STAGES, PHOTO_OUTPUTS, completed_batches, parts_directory, remove_parts, save_batch
from .photo_io import download_batch, list_remote_photos, load_iep_photos

CLIP_STAGES = {"iep", "interior_scores", "exterior_plan_scores"}
EXTERIOR_LABELS = ["street", "view", "common"]
LABEL_TO_PREDICTION = {
    "interior": "I",
    "floor_plan": "P",
    "street": "E",
    "view": "E",
    "common": "E",
}


def selected_stages(stages: list[str]) -> list[str]:
    unknown = set(stages) - set(ALL_STAGES)
    if unknown:
        raise ValueError(f"Unknown stages: {sorted(unknown)}")
    selected = [stage for stage in ALL_STAGES if stage in stages]
    if not selected:
        raise ValueError("Select at least one stage")
    return selected


def prepare_photo_frame(project_root: Path, stages: list[str], remote_root: str, ssh_target: str = "rpi") -> pd.DataFrame:
    stages = selected_stages(stages)
    if "iep" in stages:
        return list_remote_photos(remote_root, ssh_target)
    iep_path = project_root / PHOTO_OUTPUTS / "photo_classifications.parquet"
    if not iep_path.exists():
        raise FileNotFoundError(f"{iep_path} is required. Include 'iep' in STAGES_TO_RUN for the first run.")
    return load_iep_photos(iep_path, stages)


def selected_prompt_groups(prompts: dict, stages: list[str]) -> dict:
    groups = {}
    for stage, prompt_names in {
        "iep": ["iep"],
        "interior_scores": ["interior_old", "interior_recommended"],
        "exterior_plan_scores": ["exterior", "plan"],
    }.items():
        if stage in stages:
            for prompt_name in prompt_names:
                groups[prompt_name] = prompts[prompt_name]
    return groups


def decode_photos(photos: dict[str, bytes]):
    paths = []
    images = []
    for path, content in photos.items():
        try:
            with Image.open(BytesIO(content)) as source:
                images.append(source.convert("RGB"))
                paths.append(path)
        except Exception:
            continue
    return paths, images


def store_similarities(result, image_features, prototype, row_mask, columns):
    row_mask = np.asarray(row_mask, dtype=bool)
    values = None
    if row_mask.any():
        feature_mask = torch.from_numpy(row_mask).to(image_features.device)
        values = similarities(image_features[feature_mask], prototype).detach().cpu().numpy()
    for index, column in enumerate(columns):
        if column not in result:
            result[column] = np.nan
        if values is not None:
            result.loc[row_mask, column] = values[:, index]


def score_clip_images(
    paths: list[str],
    images: list,
    known_labels,
    prototypes: dict,
    stages: list[str],
    model,
    processor,
    device: str,
    clip_batch_size: int,
):
    result = pd.DataFrame({"path": paths})
    if not images:
        if "iep" in stages:
            result["label"] = pd.Series(dtype=str)
            result["prediction"] = pd.Series(dtype=str)
        return result

    image_features = embed_images(model, processor, images, device, clip_batch_size)
    if "iep" in stages:
        iep_prototype = prototypes["iep"]
        raw_scores = similarities(image_features, iep_prototype)
        probabilities = torch.softmax(model.logit_scale.exp() * raw_scores, dim=-1)
        raw_scores = raw_scores.detach().cpu().numpy()
        probabilities = probabilities.detach().cpu().numpy()
        class_names = iep_prototype["class_names"]
        for index, class_name in enumerate(class_names):
            result[f"{class_name}_similarity"] = raw_scores[:, index]
            result[f"{class_name}_probability"] = probabilities[:, index]
        labels = np.array(class_names)[probabilities.argmax(axis=1)]
        result["label"] = labels
        result["prediction"] = pd.Series(labels).map(LABEL_TO_PREDICTION)
    else:
        labels = np.asarray(known_labels)

    if "interior_scores" in stages:
        interior_mask = labels == "interior"
        for group, prefix in [("interior_old", "old"), ("interior_recommended", "recommended")]:
            prototype = prototypes[group]
            columns = [f"{prefix}_{class_name}_similarity" for class_name in prototype["class_names"]]
            store_similarities(result, image_features, prototype, interior_mask, columns)

    if "exterior_plan_scores" in stages:
        store_similarities(
            result,
            image_features,
            prototypes["exterior"],
            np.isin(labels, EXTERIOR_LABELS),
            ["low_similarity", "mid_similarity", "high_similarity"],
        )
        store_similarities(
            result,
            image_features,
            prototypes["plan"],
            labels == "floor_plan",
            ["low_similarity", "mid_similarity", "high_similarity"],
        )
    return result


async def calculate_batch_metrics(photos: list[tuple[str, bytes]], process_pool):
    if not photos:
        return pd.DataFrame(columns=["path"] + METRIC_COLUMNS)

    by_property = OrderedDict()
    for path, content in photos:
        property_dir = path.rsplit("/", 2)[0]
        by_property.setdefault(property_dir, []).append((path, content))

    loop = asyncio.get_running_loop()
    property_results = await asyncio.gather(*[
        loop.run_in_executor(process_pool, calculate_metrics, property_photos)
        for property_photos in by_property.values()
    ])
    rows = [row for property_rows in property_results for row in property_rows]
    return pd.DataFrame(rows, columns=["path"] + METRIC_COLUMNS)


async def run_pipeline(
    photo_df: pd.DataFrame,
    prompts: dict,
    stages: list[str],
    project_root: Path,
    base_url: str,
    model_id: str,
    device: str,
    properties_per_batch: int = 20,
    prefetch: int = 3,
    max_connections: int = 64,
    clip_batch_size: int = 64,
    metric_workers: int = 32,
    fresh_run: bool = False,
) -> Path:
    stages = selected_stages(stages)
    parts_dir = parts_directory(project_root, stages, photo_df)
    if fresh_run:
        remove_parts(parts_dir)
    completed = completed_batches(parts_dir)
    property_dirs = photo_df["property_dir"].unique().tolist()
    batches = [
        (index, property_dirs[start:start + properties_per_batch])
        for index, start in enumerate(range(0, len(property_dirs), properties_per_batch))
    ]
    remaining_batches = [batch for batch in batches if batch[0] not in completed]
    if not remaining_batches:
        return parts_dir

    paths_by_property = photo_df.groupby("property_dir")["path"].apply(list)
    needs_clip = bool(set(stages) & CLIP_STAGES)
    prompt_groups = selected_prompt_groups(prompts, stages)

    with ExitStack() as stack:
        process_pool = None
        if "photo_quality" in stages:
            process_pool = stack.enter_context(ProcessPoolExecutor(
                max_workers=metric_workers,
                mp_context=get_context("fork"),
            ))

        model = processor = prototypes = None
        if needs_clip:
            model, processor = load_model(model_id, device)
            prompt_embeddings = embed_prompts(model, processor, prompt_groups, device)
            prototypes = build_prototypes(prompt_groups, prompt_embeddings)

        clip_lock = asyncio.Lock()
        connector = aiohttp.TCPConnector(limit=max_connections)
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=60)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            async def process_batch(batch):
                batch_index, batch_properties = batch
                paths = [path for property_dir in batch_properties for path in paths_by_property[property_dir]]
                photos = await download_batch(session, paths, base_url)
                batch_keys = photo_df[photo_df["property_dir"].isin(batch_properties)].copy()
                batch_by_path = batch_keys.set_index("path")
                clip_result = None

                if needs_clip:
                    valid_paths, images = await asyncio.to_thread(decode_photos, photos)
                    known_labels = None
                    if "iep" not in stages:
                        known_labels = batch_by_path.loc[valid_paths, "label"].to_numpy()
                    async with clip_lock:
                        clip_result = await asyncio.to_thread(
                            score_clip_images,
                            valid_paths,
                            images,
                            known_labels,
                            prototypes,
                            stages,
                            model,
                            processor,
                            device,
                            clip_batch_size,
                        )

                if "photo_quality" in stages:
                    if "iep" in stages:
                        interior_paths = clip_result.loc[clip_result["label"].eq("interior"), "path"].tolist()
                    else:
                        interior_paths = batch_keys.loc[batch_keys["label"].eq("interior"), "path"].tolist()
                    interior_photos = [(path, photos[path]) for path in interior_paths]
                    metric_result = await calculate_batch_metrics(interior_photos, process_pool)

                if clip_result is not None:
                    if "iep" in stages:
                        batch_keys = batch_keys.drop(columns=["label", "prediction"], errors="ignore")
                    batch_keys = batch_keys.merge(clip_result, on="path", how="left")
                if "photo_quality" in stages:
                    batch_keys = batch_keys.merge(metric_result, on="path", how="left")

                await asyncio.to_thread(save_batch, batch_keys, parts_dir, batch_index)
                return batch_index, len(batch_keys)

            pending = []
            next_batch = 0
            progress = tqdm(total=len(remaining_batches), mininterval=30)
            try:
                while next_batch < min(prefetch, len(remaining_batches)):
                    pending.append(asyncio.create_task(process_batch(remaining_batches[next_batch])))
                    next_batch += 1
                while pending:
                    await pending.pop(0)
                    progress.update(1)
                    if next_batch < len(remaining_batches):
                        pending.append(asyncio.create_task(process_batch(remaining_batches[next_batch])))
                        next_batch += 1
            finally:
                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
                progress.close()

    return parts_dir
