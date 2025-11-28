"""
Convert the DeBERTa NLI model to ONNX format for TorchServe
"""
import os
from pathlib import Path
import onnxruntime as ort
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

MODEL_NAME = 'cross-encoder/nli-deberta-v3-large'
ONNX_MODEL_DIR = './onnx_model'

def get_execution_provider():
    """
    Detect and return the best available execution provider.
    Prioritizes GPU (CUDA) if available, falls back to CPU.
    """
    available_providers = ort.get_available_providers()
    print(f"Available ONNX Runtime providers: {available_providers}")
    
    if 'CUDAExecutionProvider' in available_providers:
        print("Using GPU (CUDA) for model conversion")
        return 'CUDAExecutionProvider'
    else:
        print("Using CPU for model conversion")
        return 'CPUExecutionProvider'

def convert_model():
    """Convert the model to ONNX format"""
    
    print(f"Converting {MODEL_NAME} to ONNX format...")
    print("This may take 5-10 minutes on first run...")
    
    # Detect best execution provider (GPU/CPU)
    provider = get_execution_provider()
    
    # Load and convert the model
    model = ORTModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        export=True,
        provider=provider
    )
    
    # Save the ONNX model
    model.save_pretrained(ONNX_MODEL_DIR)
    print(f"Model saved to {ONNX_MODEL_DIR}")
    
    # Save the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(ONNX_MODEL_DIR)
    print(f"Tokenizer saved to {ONNX_MODEL_DIR}")
    
    print("Conversion complete!")

if __name__ == "__main__":
    convert_model()