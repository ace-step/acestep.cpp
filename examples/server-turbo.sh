#!/bin/bash

set -eu

../build/ace-server \
    --host 0.0.0.0 \
    --port 8085 \
    --lm ../models/acestep-5Hz-lm-4B-Q8_0.gguf \
    --embedding ../models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit ../models/acestep-v15-turbo-Q8_0.gguf \
    --vae ../models/vae-BF16.gguf \
    --max-batch 1
