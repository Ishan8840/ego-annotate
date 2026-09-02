set -x
pip install -q -U torch --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -3
pip install -q -U transformers accelerate qwen-vl-utils pillow 2>&1 | tail -3
python3 - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("transformers", transformers.__version__)
PY
hf download Qwen/Qwen3-VL-8B-Instruct --quiet 2>&1 | tail -2
echo QWEN_SETUP_DONE
