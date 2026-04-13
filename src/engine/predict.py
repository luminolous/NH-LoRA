from __future__ import annotations

import argparse

import torch

from src.utils.deploy import (
    build_inference_transform,
    clamp_topk,
    load_deploy_artifact,
    load_prediction_image,
    resolve_prediction_device,
    restore_model_for_prediction,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run NH-LoRA prediction from a deploy artifact.")
    parser.add_argument("--artifact-dir", required=True, help="Path to the exported deploy artifact directory.")
    parser.add_argument("--image", required=True, help="Path to an input image.")
    parser.add_argument("--topk", type=int, default=5, help="Number of predictions to print.")
    parser.add_argument("--device", default="cpu", help="Prediction device, for example 'cpu' or 'cuda'.")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        device, device_warning = resolve_prediction_device(args.device)
        if device_warning:
            print(f"[WARN] {device_warning}")

        artifact = load_deploy_artifact(args.artifact_dir, map_location=device)
        model, inference_profile, used_profile_fallback = restore_model_for_prediction(
            artifact["inference_config"],
            artifact["artifact_payload"],
            device=device,
        )
        if used_profile_fallback:
            print("[WARN] Artifact is missing inference_profile; rebuilt it from the restored model structure.")

        preprocess_transform = build_inference_transform(artifact["preprocess_config"])
        image = load_prediction_image(args.image)
        image_tensor = preprocess_transform(image).unsqueeze(0).to(device)

        effective_topk, topk_warning = clamp_topk(args.topk, int(artifact["manifest"]["num_classes"]))
        if topk_warning:
            print(f"[WARN] {topk_warning}")

        with torch.no_grad():
            outputs = model.forward_with_state(image_tensor, task_state=None, planner_out=inference_profile)
            logits = outputs["logits"]
            probabilities = torch.softmax(logits, dim=-1)
            top_probabilities, top_indices = torch.topk(probabilities, k=effective_topk, dim=-1)
            top_scores = torch.gather(logits, dim=-1, index=top_indices)

        idx_to_class = artifact["idx_to_class"]
        print(f"Artifact: {artifact['artifact_dir']}")
        print(f"Image: {args.image}")
        print(f"Device: {device}")
        print("")
        print("Top-k predictions:")
        for rank, (probability, score, class_index) in enumerate(
            zip(top_probabilities[0].tolist(), top_scores[0].tolist(), top_indices[0].tolist()),
            start=1,
        ):
            class_name = idx_to_class.get(int(class_index), f"class_{class_index}")
            print(
                f"{rank:>2}. class_name={class_name} class_index={int(class_index)} "
                f"score={float(score):.6f} probability={float(probability):.6f}"
            )
    except Exception as exc:  # pragma: no cover - CLI wrapper
        raise SystemExit(f"[ERROR] {exc}") from exc


if __name__ == "__main__":
    main()
