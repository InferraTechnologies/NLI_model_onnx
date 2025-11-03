FROM pytorch/torchserve:latest-cpu

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install torch-model-archiver
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch-model-archiver torch-workflow-archiver

# Install dependencies from PyPI
RUN pip install --no-cache-dir \
    transformers \
    optimum[onnxruntime] \
    onnxruntime

# Install PyTorch CPU version
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Create directories
RUN mkdir -p /home/model-server/model-store && \
    mkdir -p /home/model-server/onnx_model && \
    chown -R model-server:model-server /home/model-server

USER model-server

WORKDIR /home/model-server

# Copy handler and conversion script
COPY --chown=model-server:model-server nli_handler.py .
COPY --chown=model-server:model-server convert_to_onnx.py .
COPY --chown=model-server:model-server package_model.sh .
COPY --chown=model-server:model-server config.properties .

# Make script executable
USER root
RUN chmod +x package_model.sh
USER model-server

# Convert model to ONNX
RUN python convert_to_onnx.py

# Verify ONNX model was created
RUN echo "Checking ONNX files:" && ls -la onnx_model/

# Create MAR file  
RUN bash package_model.sh

# Verify MAR file was created
RUN echo "Checking MAR file:" && ls -la model-store/ && \
    test -f model-store/nli_deberta.mar && echo "MAR file verified!"

# Expose TorchServe ports
# 8080: Inference API
# 8081: Management API
# 8082: Metrics API
EXPOSE 8080 8081 8082

# Start TorchServe with the model
CMD ["torchserve", \
     "--start", \
     "--model-store", "/home/model-server/model-store", \
     "--models", "nli_deberta=/home/model-server/model-store/nli_deberta.mar", \
     "--ts-config", "/home/model-server/config.properties", \
     "--foreground"]