FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for image/video processing
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend and frontend code, preserving structure
COPY backend/ ./backend/
COPY frontend/static/ ./frontend/static/

# Set working directory to backend for app startup
WORKDIR /app/backend

# Ensure models directory exists (if not already in repo)
RUN mkdir -p models

EXPOSE 8080

ENV FLASK_ENV=production
ENV PORT=8080

# Use Gunicorn to serve the Flask app
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
