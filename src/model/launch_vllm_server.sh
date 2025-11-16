#!/bin/bash

MODEL="$1"
PORT="$2"
BG_FLAG="$3"

# Create logs folder if it doesn't exist
mkdir -p logs

# Generate timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

CMD="python -m vllm.entrypoints.openai.api_server --model \"$MODEL\" --tokenizer \"$MODEL\" --dtype auto --port $PORT"

if [ "$BG_FLAG" = "bg" ] || [ "$BG_FLAG" = "--background" ]; then
    LOG_FILE="logs/${MODEL//\//_}_port${PORT}_$TIMESTAMP.log"
    eval "$CMD > \"$LOG_FILE\" 2>&1 &"
    echo "Launched in background (logs: $LOG_FILE, PID: $!)"
else
    eval "$CMD"
fi

# python -m vllm.entrypoints.openai.api_server --model "${1}" --tokenizer "${1}" --dtype auto --port ${2}

# python -m vllm.entrypoints.openai.api_server --model "${1}" --tokenizer "${1}" --dtype auto --port ${2}