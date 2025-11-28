"""
TorchServe Custom Handler for NLI DeBERTa with ONNX Runtime
"""
import json
import logging
import os
import numpy as np
import onnxruntime as ort
from ts.torch_handler.base_handler import BaseHandler
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class NLIDebertaHandler(BaseHandler):
    """
    Custom handler for NLI inference using ONNX Runtime
    """
    
    def __init__(self):
        super(NLIDebertaHandler, self).__init__()
        self.model = None
        self.tokenizer = None
        self.label_mapping = ['contradiction', 'entailment', 'neutral']
        self.initialized = False
        self.device = None
    
    def _get_execution_provider(self):
        """
        Detect and return the best available execution provider.
        Prioritizes GPU (CUDA) if available, falls back to CPU.
        """
        available_providers = ort.get_available_providers()
        logger.info(f"Available ONNX Runtime providers: {available_providers}")
        
        if 'CUDAExecutionProvider' in available_providers:
            logger.info("Using GPU (CUDA) for inference")
            return 'CUDAExecutionProvider'
        else:
            logger.info("Using CPU for inference")
            return 'CPUExecutionProvider'
    
    def initialize(self, context):
        """
        Initialize model and tokenizer
        """
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        
        logger.info(f"Loading model from {model_dir}")
        
        # Detect best execution provider (GPU/CPU)
        provider = self._get_execution_provider()
        self.device = 'cuda' if provider == 'CUDAExecutionProvider' else 'cpu'
        
        # Load ONNX model with detected provider
        self.model = ORTModelForSequenceClassification.from_pretrained(
            model_dir,
            provider=provider
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        logger.info(f"Model and tokenizer loaded successfully on {self.device.upper()}")
        self.initialized = True
    
    def preprocess(self, requests):
        """
        Preprocess input data
        
        Accepts two formats:
        1. Single pair: ["text1", "text2"]
        2. Batch: {"pairs": [["text1a", "text1b"], ["text2a", "text2b"]]}
        """
        premises = []
        hypotheses = []
        request_formats = []
        
        for idx, request in enumerate(requests):
            data = request.get("data") or request.get("body")
            
            if isinstance(data, (bytes, bytearray)):
                data = data.decode('utf-8')
            
            if isinstance(data, str):
                data = json.loads(data)
            
            # Determine format and extract text pairs
            if isinstance(data, list) and len(data) == 2:
                # Single pair format: ["text1", "text2"]
                premises.append(data[0])
                hypotheses.append(data[1])
                request_formats.append('single')
            elif isinstance(data, dict) and 'pairs' in data:
                # Batch format: {"pairs": [["text1a", "text1b"], ...]}
                pairs = data['pairs']
                for pair in pairs:
                    if isinstance(pair, list) and len(pair) == 2:
                        premises.append(pair[0])
                        hypotheses.append(pair[1])
                    else:
                        raise ValueError(f"Each pair must be a list of exactly 2 texts")
                request_formats.append(('batch', len(pairs)))
            else:
                raise ValueError("Invalid format. Use either ['text1', 'text2'] or {'pairs': [['text1a', 'text1b'], ...]}")
        
        # Tokenize all pairs
        features = self.tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        return features, request_formats
    
    def inference(self, data):
        """
        Run inference on preprocessed data
        """
        features, request_formats = data
        
        # Run ONNX inference
        outputs = self.model(**features)
        logits = outputs.logits.detach().numpy()
        
        return logits, request_formats
    
    def postprocess(self, inference_output):
        """
        Format the inference output
        """
        logits, request_formats = inference_output
        
        # Get predictions
        predictions = np.argmax(logits, axis=1)
        labels = [self.label_mapping[pred] for pred in predictions]
        
        # Format results based on original request format
        results = []
        idx = 0
        
        for req_format in request_formats:
            if req_format == 'single':
                # Single pair - return single result
                result = {
                    "label": labels[idx],
                    "scores": {
                        "contradiction": float(logits[idx][0]),
                        "entailment": float(logits[idx][1]),
                        "neutral": float(logits[idx][2])
                    }
                }
                results.append(result)
                idx += 1
            else:
                # Batch format - return array of results
                batch_type, count = req_format
                batch_results = []
                for i in range(count):
                    batch_results.append({
                        "label": labels[idx],
                        "scores": {
                            "contradiction": float(logits[idx][0]),
                            "entailment": float(logits[idx][1]),
                            "neutral": float(logits[idx][2])
                        }
                    })
                    idx += 1
                results.append({"predictions": batch_results})
        
        return results