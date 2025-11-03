"""
Convert the DeBERTa NLI model to ONNX format for TorchServe
"""
import os
from pathlib import Path
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

MODEL_NAME = 'cross-encoder/nli-deberta-v3-large'
ONNX_MODEL_DIR = './onnx_model'

def convert_model():
    """Convert the model to ONNX format"""
    
    print(f"Converting {MODEL_NAME} to ONNX format...")
    print("This may take 5-10 minutes on first run...")
    
    # Load and convert the model
    model = ORTModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        export=True,
        provider="CPUExecutionProvider"
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