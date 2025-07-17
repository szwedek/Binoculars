FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

WORKDIR /app

ADD binoculars binoculars
COPY pyproject.toml requirements.txt LICENSE.md README.md binoculars/

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e binoculars

# Copy your handler file
COPY runpod-worker/handler.py .

# Start the container
CMD ["python3", "-u", "handler.py"]