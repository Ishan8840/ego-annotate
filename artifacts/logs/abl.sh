export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while IFS='|' read -r F M NAME; do
  [ -z "$NAME" ] && continue
  echo "=== frames=$F context=$M -> caps_$NAME.jsonl ==="
  FRAMES_PER_SPAN=$F CONTEXT_MODE=$M SPANS_PER_CALL=5 FREE_TEXT=0 \
    python3 -u caption.py run --spans spans_v4.jsonl --backend qwen-local \
    --out caps_$NAME.jsonl 2>&1 | grep -E "captions from|throughput"
done <<'EOF'
4|sequential|f4seq
8|sequential|f8seq
4|shuffled|f4shuf
EOF
echo ABL_DONE
