export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== A2: 8 frames/span, 5 spans/call (40 images/call - same as baseline) ==="
FRAMES_PER_SPAN=8 SPANS_PER_CALL=5 python3 -u caption.py run --backend qwen-local --out caps_f8b5.jsonl 2>&1 | grep -v "Loading weights"
echo "=== B2: 4 frames/span, 1 span/call (4 images/call) ==="
FRAMES_PER_SPAN=4 SPANS_PER_CALL=1 python3 -u caption.py run --backend qwen-local --out caps_s1b.jsonl 2>&1 | grep -v "Loading weights"
echo ABLATE2_DONE
