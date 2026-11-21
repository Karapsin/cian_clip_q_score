import torch
from transformers import CLIPModel, CLIPProcessor


def normalize(features: torch.Tensor) -> torch.Tensor:
    return features / features.norm(dim=-1, keepdim=True)


def load_model(model_id: str, device: str):
    processor = CLIPProcessor.from_pretrained(model_id, use_fast=True)
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    return model, processor


def output_features(output):
    return output.pooler_output if hasattr(output, "pooler_output") else output


def embed_prompts(model, processor, prompt_groups: dict, device: str, batch_size: int = 64):
    unique_prompts = sorted({
        prompt
        for classes in prompt_groups.values()
        for prompts in classes.values()
        for prompt in prompts
    })
    embeddings = {}
    with torch.inference_mode():
        for start in range(0, len(unique_prompts), batch_size):
            prompts = unique_prompts[start:start + batch_size]
            inputs = processor(text=prompts, padding=True, truncation=True, return_tensors="pt").to(device)
            features = normalize(output_features(model.get_text_features(**inputs)))
            for prompt, embedding in zip(prompts, features):
                embeddings[prompt] = embedding
    return embeddings


def build_prototypes(prompt_groups: dict, prompt_embeddings: dict):
    prototypes = {}
    for group, classes in prompt_groups.items():
        class_names = list(classes)
        class_embeddings = []
        for class_name in class_names:
            embeddings = torch.stack([prompt_embeddings[prompt] for prompt in classes[class_name]])
            class_embeddings.append(normalize(embeddings.mean(dim=0)))
        prototypes[group] = {
            "class_names": class_names,
            "embeddings": torch.stack(class_embeddings),
        }
    return prototypes


def embed_images(model, processor, images: list, device: str, batch_size: int = 64):
    batches = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            inputs = processor(images=images[start:start + batch_size], return_tensors="pt").to(device)
            features = normalize(output_features(model.get_image_features(**inputs)))
            batches.append(features)
    if not batches:
        return torch.empty((0, model.config.projection_dim), device=device)
    return torch.cat(batches)


def similarities(image_embeddings: torch.Tensor, prototype: dict) -> torch.Tensor:
    return image_embeddings @ prototype["embeddings"].T
