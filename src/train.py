# ==============================================================================
#  KAGGLE T4 x2 SETUP: Unsloth + Qwen2.5-Coder-7B-Instruct-bnb-4bit (Lara.coder)
# ==============================================================================

print("=" * 70)
print("STEP 1/4: Installing Unsloth (Kaggle build) + dependencies...")
print("=" * 70)

!pip install --upgrade "unsloth[kaggle-new]"
!pip install --upgrade --no-cache-dir xformers trl peft accelerate bitsandbytes

print("\n" + "=" * 70)
print("STEP 1/4 COMPLETE: All packages installed.")
print("=" * 70)

# ------------------------------------------------------------------------------
# STEP 2/4: GPU sanity check (confirms both T4s are visible)
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2/4: Checking GPU environment...")
print("=" * 70)

import torch

gpu_count = torch.cuda.device_count()
if gpu_count == 0:
    raise RuntimeError(
        "No GPU detected! In Kaggle: Notebook Settings -> Accelerator -> GPU T4 x2."
    )

print(f"Detected {gpu_count} GPU(s):")
for i in range(gpu_count):
    props = torch.cuda.get_device_properties(i)
    print(f"  -> GPU {i}: {props.name} | {props.total_memory / 1024**3:.1f} GB VRAM")

print("STEP 2/4 COMPLETE.")

# ------------------------------------------------------------------------------
# STEP 3/4: Load Qwen2.5-Coder-7B-Instruct in 4-bit
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3/4: Loading unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit ...")
print("=" * 70)

from unsloth import FastLanguageModel

MODEL_NAME      = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH  = 2048   
DTYPE           = None   
LOAD_IN_4BIT    = True   

try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype          = DTYPE,
        load_in_4bit   = LOAD_IN_4BIT,
    )
except torch.cuda.OutOfMemoryError as e:
    print("!!! OUT OF MEMORY while loading the model !!!")
    raise e

print("\n" + "=" * 70)
print("STEP 3/4 COMPLETE: Model and tokenizer loaded successfully.")
print("=" * 70)

# ------------------------------------------------------------------------------
# STEP 4/4: Post-load verification & Lara.coder configuration setup
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4/4: Verifying loaded model & setting up Lara.coder...")
print("=" * 70)

num_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters : {num_params:,}")

# Configuring LoRA for Lara.coder
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

LORA_DIR = "/kaggle/working/Lara.coder"
print(f"  Target save dir  : {LORA_DIR}")

print("\n✅ ALL STEPS COMPLETE — Lara.coder is initialized and ready for training!")

