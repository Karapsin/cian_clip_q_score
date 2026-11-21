import asyncio
import subprocess
from pathlib import Path
from urllib.parse import quote

import aiohttp
import pandas as pd

KEY = ["deal_type", "property_id", "offer_id", "photo_name"]


def list_remote_photos(remote_root: str, ssh_target: str = "rpi") -> pd.DataFrame:
    command = ["ssh", ssh_target, f"cd {remote_root} && find . -type f -name '*.jpg'"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    paths = sorted(line[2:] for line in result.stdout.splitlines())
    photo_df = pd.DataFrame([path.split("/") for path in paths], columns=KEY)
    return add_photo_paths(photo_df)


def add_photo_paths(photo_df: pd.DataFrame) -> pd.DataFrame:
    photo_df = photo_df.copy()
    photo_df["path"] = photo_df[KEY].agg("/".join, axis=1)
    photo_df["property_dir"] = photo_df["deal_type"] + "/" + photo_df["property_id"]
    return photo_df


def load_iep_photos(output_path: Path, stages: list[str]) -> pd.DataFrame:
    columns = KEY + ["label", "prediction"]
    photo_df = pd.read_parquet(output_path, columns=columns)
    labels = []
    if "interior_scores" in stages or "photo_quality" in stages:
        labels.append("interior")
    if "exterior_plan_scores" in stages:
        labels.extend(["street", "view", "common", "floor_plan"])
    photo_df = photo_df[photo_df["label"].isin(labels)].reset_index(drop=True)
    return add_photo_paths(photo_df)


def photo_url(base_url: str, path: str) -> str:
    encoded_path = "/".join(quote(part) for part in path.split("/"))
    return base_url.rstrip("/") + "/" + encoded_path


async def download_photo(session: aiohttp.ClientSession, base_url: str, path: str):
    try:
        async with session.get(photo_url(base_url, path)) as response:
            response.raise_for_status()
            return path, await response.read(), True
    except asyncio.CancelledError:
        raise
    except Exception:
        return path, None, False


async def download_batch(
    session: aiohttp.ClientSession,
    paths: list[str],
    base_url: str,
    attempts: int = 5,
) -> dict[str, bytes]:
    photos = {}
    remaining = list(paths)
    for attempt in range(attempts):
        results = await asyncio.gather(*[
            download_photo(session, base_url, path)
            for path in remaining
        ])
        for path, content, succeeded in results:
            if succeeded:
                photos[path] = content
        remaining = [path for path in remaining if path not in photos]
        if not remaining:
            return photos
        if attempt + 1 < attempts:
            await asyncio.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Could not download {len(remaining)} photos after {attempts} attempts")
