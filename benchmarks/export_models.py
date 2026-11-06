"""Export ResNet-50, MobileNetV2, BERT-base and GPT-2 to ONNX (opset 14, static shapes).

The exports keep the raw graphs (``do_constant_folding=False``): BatchNormalization nodes, Shape
arithmetic, decomposed LayerNorm and GELU stay in the file so that the compiler's passes, and
onnxruntime's own optimizer, start from the same unoptimized graph.

    uv run --group bench python benchmarks/export_models.py [resnet50 mobilenetv2 bert-base gpt2]
"""

from __future__ import annotations

import gc
import json
import os
import sys

from common import DATA, MODELS, model_path

os.environ.setdefault("HF_HOME", str(DATA / "hf"))
os.environ.setdefault("TORCH_HOME", str(DATA / "torch"))

import torch

SEQ = 128
OPSET = 14


def export_vision(name: str) -> None:
    import torchvision

    if name == "resnet50":
        model = torchvision.models.resnet50(weights="IMAGENET1K_V1")
    else:
        model = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V1")
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        (x,),
        str(model_path(name)),
        input_names=["image"],
        output_names=["logits"],
        opset_version=OPSET,
        do_constant_folding=False,
        dynamo=False,
    )


class _LMHead(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, use_cache=False).logits


class _Encoder(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(input_ids=input_ids)
        return out.last_hidden_state, out.pooler_output


def export_text(name: str) -> None:
    from transformers import AutoModel, GPT2LMHeadModel

    ids = torch.randint(0, 1000, (1, SEQ))
    if name == "gpt2":
        import transformers.models.gpt2.modeling_gpt2 as gpt2_module

        # The vmap-based mask builder cannot be traced; GPT-2's eager attention applies its own
        # causal mask from the registered `bias` buffer, and no padding mask is used.
        gpt2_module.create_causal_mask = lambda *args, **kwargs: None
        model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
        wrapped: torch.nn.Module = _LMHead(model)
        outputs = ["logits"]
    else:
        model = AutoModel.from_pretrained("bert-base-uncased", attn_implementation="eager").eval()
        wrapped = _Encoder(model)
        outputs = ["last_hidden_state", "pooler_output"]
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            (ids,),
            str(model_path(name)),
            input_names=["input_ids"],
            output_names=outputs,
            opset_version=OPSET,
            do_constant_folding=False,
            dynamo=False,
        )


def main(names: list[str]) -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    info = {}
    for name in names:
        if model_path(name).exists():
            print(f"{name}: exists")
        else:
            (export_vision if name in ("resnet50", "mobilenetv2") else export_text)(name)
            gc.collect()
        info[name] = {"file": model_path(name).name, "bytes": model_path(name).stat().st_size}
        print(name, info[name])
    meta = MODELS / "models.json"
    old = json.loads(meta.read_text()) if meta.exists() else {}
    meta.write_text(json.dumps({**old, **info}, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:] or ["resnet50", "mobilenetv2", "bert-base", "gpt2"])
