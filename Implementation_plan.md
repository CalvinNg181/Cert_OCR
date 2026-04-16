# Phase 1: Environment Preparation

To run Qwen2.5-VL, you need the latest `transformers` and `accelerate` libraries, as the model uses a specialized Vision Transformer (ViT) architecture.

### 1.1 Requirements

- **Drivers:** NVIDIA Driver 530+ (for CUDA 12 support)
    
- **Python:** 3.10+
    
- **Key Libraries:** ```bash
    
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121 "null")
    
    pip install transformers accelerate bitsandbytes qwen-vl-utils python-multipart
    
    ```
    
    ```
    

# Phase 2: Deployment Strategy (8GB VRAM Optimization)

On an RTX 2080, we will use **BitsAndBytes NF4 Quantization**. This reduces the model size to ~2.2GB, leaving ~5GB for "Activation Memory" (where the actual image processing happens).

### 2.1 The "Certificate OCR" Python Wrapper

This script handles the image-to-text pipeline using the optimized 4-bit config.

```
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 1. Setup 4-bit configuration
model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

# 2. Load model with quantization
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True, # Critical for 8GB VRAM
    attn_implementation="flash_attention_2" # Use if you have high-end CUDA installed
)

# 3. Load processor
processor = AutoProcessor.from_pretrained(model_id)

def extract_certificate_data(image_path):
    # Prompt optimized for structured marathon data
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "Extract all fields from this marathon certificate. Output as valid JSON with keys: event_name, runner_name, race_number, gun_time, and rank."}
            ],
        }
    ]

    # Preprocessing
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    # Generation
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    
    return output_text[0]
```

# Phase 3: Testing & Validation Plan

### 3.1 Resolution Testing

Qwen2.5-VL uses **Native Dynamic Resolution**.

- **Test Case A:** Low res (640x480) - Check for speed.
    
- **Test Case B:** High res (2000x2000) - Check if small "Chip Time" text is captured.
    
- _Note:_ If you get `OutOfMemory`, use `processor` settings to cap `max_pixels=1280*28*28`.
    

### 3.2 Output Consistency

- Verify the JSON formatting. Qwen2.5 is much better at following schemas than Qwen2.
    
- Test with "Artistic Fonts." Marathon certificates often use script/calligraphy fonts. This is where Qwen2.5-VL beats traditional OCR.
    

# Phase 4: Scaling for Production

Once the logic is validated on your RTX 2080:

1. **Switch to vLLM:** For production, use `vllm` to host the model. It provides a 3x throughput increase.
    
2. **API Integration:** Wrap the code in a **FastAPI** endpoint to allow your "Coding Assistant" to send image URLs and receive JSON objects.