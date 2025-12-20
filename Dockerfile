FROM python:3.11-slim

WORKDIR /app

# Install ffmpeg and clean up in single layer
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the required files for the serverless handler
COPY rp_handler.py .
COPY inference.py .
COPY inference_utils.py .
COPY convex_helpers.py .
COPY rvm_mobilenetv3.pt .

CMD ["python3", "-u", "rp_handler.py"]