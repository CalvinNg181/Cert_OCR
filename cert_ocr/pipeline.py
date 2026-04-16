import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

from .utils import parse_json_output, load_image

PROMPT = (
    "Extract the following fields from this running/marathon certificate. "
    "Output ONLY valid JSON with exactly these keys:\n"
    "  - runner_name: the participant's full name\n"
    "  - race_category: the race group or distance category "
    "(e.g. 'Full Marathon', 'Half Marathon', '10KM', '5KM')\n"
    "  - finish_time: use Chip Time if present on the certificate, "
    "otherwise use the Official Time or Gun Time\n"
    "  - time_type: the label of the time you extracted "
    "(e.g. 'Chip Time', 'Official Time', 'Gun Time')\n"
    "If a field cannot be found, set its value to null. "
    "Do not include any explanation or extra text outside the JSON."
)

# Safety cap for high-res images to avoid OOM on 8GB VRAM.
# Set to None in load_model() call to allow full native resolution.
MAX_PIXELS = 1280 * 28 * 28


def extract_certificate_data(
    image_path: str,
    model,
    processor,
    max_new_tokens: int = 512,
) -> dict:
    """
    Run the certificate OCR pipeline on a single image or PDF.

    Args:
        image_path: Local file path to a .jpg/.png/.pdf certificate.
        model: Loaded Qwen2.5-VL model.
        processor: Corresponding AutoProcessor (configure max_pixels at load time).
        max_new_tokens: Generation budget. Reduce to 256 if hitting OOM.

    Returns:
        dict with keys: runner_name, race_category, finish_time, time_type.
        On parse failure, returns {"raw_output": ..., "parse_error": True}.
    """
    pil_image, _ = load_image(image_path)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": PROMPT},
        ],
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    return parse_json_output(raw)
