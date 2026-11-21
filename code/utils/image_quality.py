from io import BytesIO

import cv2
import numpy as np
from brisque import BRISQUE
from PIL import Image
from skimage.color import rgb2gray

METRIC_COLUMNS = [
    "sharpness", "clarity", "brightness", "brisque",
    "width", "height", "resolution", "format", "mode",
]

cv2.setNumThreads(1)


class CompatibleBRISQUE(BRISQUE):
    def scale_features(self, features):
        features_flat = np.array([np.asarray(value).item() for value in features], dtype=np.float64)
        min_ = np.asarray(self.scale_params["min_"], dtype=np.float64)
        max_ = np.asarray(self.scale_params["max_"], dtype=np.float64)
        return -1 + (2.0 / (max_ - min_) * (features_flat - min_))

    def score(self, image):
        image = self.remove_alpha_channel(self.load_image(image))
        gray_image = rgb2gray(image)
        features = self.calculate_brisque_features(gray_image, kernel_size=7, sigma=7 / 6)
        downscaled = cv2.resize(gray_image, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_CUBIC)
        downscaled_features = self.calculate_brisque_features(downscaled, kernel_size=7, sigma=7 / 6)
        all_features = np.concatenate((features, downscaled_features))
        return self.calculate_image_quality_score(all_features)


worker_model = None


def brisque_score(rgb_image):
    global worker_model
    if worker_model is None:
        worker_model = CompatibleBRISQUE(url=False)
    return worker_model.score(rgb_image)


def image_statistics(content: bytes) -> dict:
    try:
        with Image.open(BytesIO(content)) as image:
            width, height = image.size
            result = {
                "width": width,
                "height": height,
                "resolution": width * height,
                "format": image.format,
                "mode": image.mode,
            }
            rgb_image = np.asarray(image.convert("RGB"))

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray_image, cv2.CV_64F).var()
        contrast = gray_image.std()
        lightness = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HLS)[:, :, 1]
        result.update({
            "sharpness": sharpness.item(),
            "clarity": (sharpness * contrast).item(),
            "brightness": lightness.mean().item(),
            "brisque": brisque_score(rgb_image),
        })
        return result
    except Exception:
        return {column: np.nan for column in METRIC_COLUMNS}


def calculate_metrics(photos: list[tuple[str, bytes]]) -> list[dict]:
    return [{"path": path, **image_statistics(content)} for path, content in photos]
