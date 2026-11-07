# Data Card: ML Graph Compiler

The compiler is not trained; data is used only to check numerical parity and to time models.
Nothing below is committed to the repository: data and weights live under
`../.bench-data/ml-graph-compiler/` and are fetched by the benchmark scripts.

## Sources and provenance

| Item | Source | Size used | License |
|---|---|---|---|
| Imagenette, 320 px, validation split | fast.ai, `https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz` (a 10 class subset of ImageNet) | 3,925 JPEG images, 10 classes, about 100 MB | Apache 2.0 (images from ImageNet, subject to its terms) |
| ResNet-50 weights | torchvision `IMAGENET1K_V1` | 25.6 M parameters, ONNX file 102.5 MB | BSD-3-Clause (torchvision) |
| MobileNetV2 weights | torchvision `IMAGENET1K_V1` | 3.5 M parameters, ONNX file 14.2 MB | BSD-3-Clause (torchvision) |
| BERT-base uncased | Hugging Face `bert-base-uncased` | 110 M parameters, ONNX file 438.2 MB | Apache 2.0 |
| GPT-2 (124M) | Hugging Face `gpt2` | 124 M parameters, ONNX file 498.3 MB | MIT |
| Token inputs | `numpy.random.default_rng(seed)`, seeds 0 to 49, length 128 | 50 sequences per text model | generated |

The ONNX files are exported by `benchmarks/export_models.py` at opset 14 with static shapes
(batch 1, 224x224 images, 128 tokens) and with constant folding off, so the compiler and
onnxruntime start from the same unoptimized graph. The PyTorch baselines load the same
checkpoints through torchvision and transformers.

## Preprocessing
Images: short side resized to 256 (bilinear), center crop 224, scaled to [0, 1] and normalized
with the ImageNet mean and standard deviation. The ten Imagenette WordNet ids are mapped to their
ImageNet-1k class indices so that top-1 accuracy can be computed over all 1,000 logits.

## Known gaps
- Imagenette contains ten easy-to-separate classes; top-1 accuracy on it is far higher than on
  full ImageNet and is reported only to show that both engines classify identically.
- Random token ids are not natural text; they exercise every embedding row range and the full
  attention path but not realistic activation statistics.

## Ethical considerations
The images are public photographs from ImageNet; no personal data is collected or produced.
