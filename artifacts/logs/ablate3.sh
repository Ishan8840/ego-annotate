export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for cfg in "4 10 base_f4s10" "8 5 abl_f8s5" "4 1 abl_f4s1"; do
  set -- $cfg
  echo "=== frames=$1 spans/call=$2 -> caps_$3.jsonl ==="
  FRAMES_PER_SPAN=$1 SPANS_PER_CALL=$2 python3 -u caption.py run --backend qwen-local \
    --out caps_$3.jsonl 2>&1 | grep -vE "Loading weights" | grep -E "DEBUG|captions from|throughput|mean tokens"
done
echo ABLATE3_DONE
