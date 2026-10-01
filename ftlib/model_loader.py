"""Load Gemma 4 E-series so it fits an 8 GB GPU.

Why this exists: Gemma 4 "E" models carry a huge per-layer embedding table
(`embed_tokens_per_layer`, ~4.7 GB for E2B). It is a pure lookup table -- no
matrix math -- so Google designed it to live in CPU RAM. That is what
"Effective 2B" means: only ~2B params need to sit on the accelerator.

Unsloth's built-in `offload_embedding` does not cover this table (and is off on
Windows), so we do it here: put the table on CPU, look up token ids there, and
ship only the small result (a few MB per batch) to the GPU.

Usage (from any task):
    from ftlib.model_loader import load_model
    model, tokenizer = load_model("unsloth/gemma-4-E4B-it", max_seq_length=512)
"""
from ftlib import paths  # noqa: F401  (sets cache dirs -- must run before HF imports)
from unsloth import FastModel
import torch
from transformers import BitsAndBytesConfig

PER_LAYER_TABLE = "model.language_model.embed_tokens_per_layer"


def _find_module(model, suffix):
    for name, module in model.named_modules():
        if name.endswith(suffix):
            return module
    raise RuntimeError(f"{suffix} not found - is this a Gemma 4 E-series model?")


def _run_lookup_on_cpu(table, gpu):
    # accelerate's "cpu" offload leaves a meta placeholder and would stream the
    # whole 4+ GB table to the GPU on every forward pass. Replace that with a
    # real CPU tensor and do the lookup on the CPU instead.
    from accelerate.hooks import remove_hook_from_module

    if table.weight.device.type == "meta":
        hook = table._hf_hook
        hooks = getattr(hook, "hooks", [hook])  # may be a SequentialHook
        weight = next(h.weights_map["weight"] for h in hooks if getattr(h, "weights_map", None) is not None)
        remove_hook_from_module(table)
        table.weight = torch.nn.Parameter(weight.to("cpu", torch.bfloat16), requires_grad=False)
    table.to("cpu")  # buffers too (embed_scale)

    def pre_hook(module, args):
        return (args[0].to(module.weight.device),) + tuple(args[1:])

    def post_hook(module, args, output):
        return output.to(gpu, non_blocking=True)

    table.register_forward_pre_hook(pre_hook, prepend=True)
    table.register_forward_hook(post_hook, prepend=True)


def load_model(model_name, max_seq_length=1024, offload_per_layer_table=True):
    kwargs = {}
    if offload_per_layer_table:
        # Load the original bf16 weights and quantize while loading. Unsloth's
        # pre-quantized "-bnb-4bit" repos carry their own quantization config,
        # which would override the CPU-offload setting below.
        kwargs["use_exact_model_name"] = True
        kwargs["device_map"] = {"": 0, PER_LAYER_TABLE: "cpu"}
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_enable_fp32_cpu_offload=True,  # lets bnb accept a CPU entry in device_map
        )

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        full_finetuning=False,
        dtype=None,
        **kwargs,
    )

    if offload_per_layer_table:
        table = _find_module(model, "embed_tokens_per_layer")
        _run_lookup_on_cpu(table, torch.device("cuda", 0))
        # We manage that table ourselves now. If PEFT/Trainer still see "cpu" in the
        # device map they re-dispatch the whole model and undo the placement.
        model.hf_device_map = {"": 0}
        print(f"Per-layer embedding table on {table.weight.device} "
              f"({table.weight.numel() * table.weight.element_size() / 1024**3:.1f} GB kept off the GPU)")
    print(f"GPU memory after load: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    return model, tokenizer
