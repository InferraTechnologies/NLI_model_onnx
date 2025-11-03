#!/bin/bash

# Package the ONNX model into a TorchServe MAR file

echo "Creating TorchServe MAR file..."

# Check if required files exist
if [ ! -f "onnx_model/model.onnx" ]; then
    echo "ERROR: onnx_model/model.onnx not found!"
    exit 1
fi

# List all files in onnx_model directory for debugging
echo "Files in onnx_model directory:"
ls -la onnx_model/

# Find all required files
EXTRA_FILES="onnx_model/config.json"

if [ -f "onnx_model/tokenizer.json" ]; then
    EXTRA_FILES="$EXTRA_FILES,onnx_model/tokenizer.json"
fi

if [ -f "onnx_model/tokenizer_config.json" ]; then
    EXTRA_FILES="$EXTRA_FILES,onnx_model/tokenizer_config.json"
fi

if [ -f "onnx_model/special_tokens_map.json" ]; then
    EXTRA_FILES="$EXTRA_FILES,onnx_model/special_tokens_map.json"
fi

if [ -f "onnx_model/vocab.txt" ]; then
    EXTRA_FILES="$EXTRA_FILES,onnx_model/vocab.txt"
fi

if [ -f "onnx_model/spm.model" ]; then
    EXTRA_FILES="$EXTRA_FILES,onnx_model/spm.model"
fi

echo "Extra files to include: $EXTRA_FILES"

torch-model-archiver \
    --model-name nli_deberta \
    --version 1.0 \
    --serialized-file onnx_model/model.onnx \
    --handler nli_handler.py \
    --extra-files "$EXTRA_FILES" \
    --export-path model-store \
    --force

if [ $? -eq 0 ]; then
    echo "MAR file created successfully at model-store/nli_deberta.mar"
    ls -lh model-store/
else
    echo "ERROR: Failed to create MAR file"
    exit 1
fi