#!/usr/bin/env python3
"""Convert an EmotiEffLib / HSEmotion ONNX model to a Core ML package.

    python convert_to_coreml.py enet_b0_8_best_vgaf.onnx EmotiEffNetB0.mlpackage

The weights are NOT part of this script. Download the ONNX file from
https://github.com/sb-ai-lab/EmotiEffLib (models/affectnet_emotions/onnx/) and check the
license and dataset terms (VGGFace2, AffectNet) before redistributing anything converted
from it.

The Core ML model takes an RGB image and applies the ImageNet normalization itself, so a
caller only crops the face and passes it in. It returns the raw 8-class logits in this order:
Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise.

Needs: coremltools, onnx, onnx2torch, torch, numpy (and onnxruntime to run --verify).
"""

import argparse
import sys

import coremltools as ct
import numpy as np
import onnx
import torch
from onnx2torch import convert

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SIZE = 224
CLASSES = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]


class Normalized(torch.nn.Module):
    """Takes pixels in 0...255 (RGB, NCHW) and normalizes them the way the model was trained."""

    def __init__(self, network: torch.nn.Module):
        super().__init__()
        self.network = network
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1) * 255)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.network((pixels - self.mean) / self.std)


def build(onnx_path: str, output_path: str) -> torch.nn.Module:
    network = convert(onnx.load(onnx_path)).eval()
    model = Normalized(network).eval()
    example = torch.rand(1, 3, SIZE, SIZE) * 255
    traced = torch.jit.trace(model, example)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.ImageType(
                name="face",
                shape=(1, 3, SIZE, SIZE),
                color_layout=ct.colorlayout.RGB,
            )
        ],
        outputs=[ct.TensorType(name="logits")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_units=ct.ComputeUnit.ALL,
    )
    mlmodel.short_description = "EmotiEffLib / HSEmotion EfficientNet-B0 facial emotion logits"
    mlmodel.user_defined_metadata["classes"] = ",".join(CLASSES)
    mlmodel.user_defined_metadata["source"] = "https://github.com/sb-ai-lab/EmotiEffLib"
    mlmodel.save(output_path)
    return model


def verify(onnx_path: str, mlpackage: str, torch_model: torch.nn.Module) -> bool:
    """Compare Core ML against onnxruntime on random images. Returns True when they agree."""
    import onnxruntime as ort
    from PIL import Image

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    mlmodel = ct.models.MLModel(mlpackage)
    rng = np.random.default_rng(0)
    mean = np.array(MEAN, dtype=np.float32)
    std = np.array(STD, dtype=np.float32)

    worst = 0.0
    same_top = 0
    trials = 20
    for _ in range(trials):
        pixels = rng.integers(0, 256, size=(SIZE, SIZE, 3), dtype=np.uint8)
        reference_input = ((pixels.astype(np.float32) / 255 - mean) / std).transpose(2, 0, 1)[None]
        reference = session.run(None, {"input": reference_input.astype(np.float32)})[0][0]
        got = np.asarray(mlmodel.predict({"face": Image.fromarray(pixels)})["logits"]).reshape(-1)
        worst = max(worst, float(np.abs(reference - got).max()))
        same_top += int(reference.argmax() == got.argmax())
    print(f"max |onnxruntime - coreml| logit difference: {worst:.4f}")
    print(f"same top class: {same_top}/{trials}")
    return worst < 0.5 and same_top == trials


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("onnx_path")
    parser.add_argument("output_path")
    parser.add_argument("--verify", action="store_true", help="compare with onnxruntime afterwards")
    args = parser.parse_args()

    torch_model = build(args.onnx_path, args.output_path)
    print(f"saved {args.output_path}")
    if args.verify and not verify(args.onnx_path, args.output_path, torch_model):
        print("verification failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
