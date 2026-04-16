import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# RTX 5060 (Blackwell, 8GB VRAM): NF4 double-quant keeps model at ~2.2GB,
# leaving ~5GB headroom for activation memory during vision processing.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,  # saves ~0.4GB extra
    bnb_4bit_quant_type="nf4",
)

# Default pixel cap — safe for 8GB VRAM. Pass max_pixels=None for full resolution.
DEFAULT_MAX_PIXELS = 1280 * 28 * 28


def load_model(model_id: str = MODEL_ID, max_pixels: int = DEFAULT_MAX_PIXELS):
    """Load Qwen2.5-VL with 4-bit quantization. Returns (model, processor).

    Args:
        model_id: HuggingFace model ID.
        max_pixels: Vision encoder pixel cap. Set to None for native resolution
                    (may OOM on images larger than ~2000×2000 on 8GB VRAM).
    """
    print(f"Loading model: {model_id}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        # flash_attention_2 omitted: no stable sm_120 wheel for Blackwell yet
    )
    processor_kwargs = {}
    if max_pixels is not None:
        processor_kwargs["max_pixels"] = max_pixels
    processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)
    print("Model loaded. VRAM used: "
          f"{torch.cuda.memory_allocated() / 1e9:.2f} GB / "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB total")
    return model, processor
