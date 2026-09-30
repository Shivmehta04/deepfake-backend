FROM python:3.11-slim

WORKDIR /app

# System dependencies for OpenCV, audio, and downloading
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download the three models from Hugging Face at build time
RUN mkdir -p models && \
    wget -q --show-progress -O models/image_deepfake_model.h5 "https://huggingface.co/Shivmehta04/deepfake-models/resolve/main/image_deepfake_model.h5" && \
    wget -q --show-progress -O models/video_deepfake_model.h5 "https://huggingface.co/Shivmehta04/deepfake-models/resolve/main/video_deepfake_model.h5" && \
    wget -q --show-progress -O models/audio_deepfake_model.h5 "https://huggingface.co/Shivmehta04/deepfake-models/resolve/main/audio_deepfake_model.h5" && \
    ls -lh models/

COPY . .

EXPOSE 10000

CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--timeout", "300", "app:app"]