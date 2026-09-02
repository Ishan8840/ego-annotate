set -e
echo "=== A: 8 frames/span, 10 spans/call ==="
FRAMES_PER_SPAN=8 SPANS_PER_CALL=10 python3 -u caption.py run --backend qwen-local --out caps_f8.jsonl 2>&1 | grep -v "Loading weights" | tail -4
echo "=== B: 4 frames/span, 1 span/call ==="
FRAMES_PER_SPAN=4 SPANS_PER_CALL=1 python3 -u caption.py run --backend qwen-local --out caps_s1.jsonl 2>&1 | grep -v "Loading weights" | tail -4
echo ABLATE_DONE
