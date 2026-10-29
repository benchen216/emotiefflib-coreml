# emotiefflib-coreml

A script that converts the ONNX facial-emotion models from
[EmotiEffLib / HSEmotion](https://github.com/sb-ai-lab/EmotiEffLib) to Core ML.

**No model weights are included in this repository.** Download the ONNX file yourself and read
its terms first. The upstream code is Apache-2.0, but the licence of the pretrained weights is
not stated; they were trained on VGGFace2 and AffectNet, which have their own terms. I asked
upstream here: https://github.com/sb-ai-lab/EmotiEffLib/issues/65

## Use

```sh
pip install -r requirements.txt
curl -L -o enet_b0_8_best_vgaf.onnx \
  https://github.com/sb-ai-lab/EmotiEffLib/raw/main/models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx
python convert_to_coreml.py enet_b0_8_best_vgaf.onnx EmotiEffNetB0.mlpackage --verify
```

The Core ML model takes a 224×224 RGB face image (ImageNet normalization is built in) and
returns 8 logits in this order: Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness,
Surprise. Crop the face first.

## Checked

`--verify` compares Core ML with onnxruntime on 20 random images. For `enet_b0_8_best_vgaf`
the largest logit difference was 0.04 and the top class matched every time. That checks the
conversion only, not recognition quality on real faces.

## Credit

If you use the models, cite the papers listed in the
[EmotiEffLib README](https://github.com/sb-ai-lab/EmotiEffLib#readme).
