# Use a clean Python base image (compatible with CPU only)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files from local app directory to container
COPY ./app /app

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Disable internet access (optional)
ENV no_proxy="*"

# Default command to run your script
CMD ["python", "main.py"]
