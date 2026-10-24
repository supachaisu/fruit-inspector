FROM --platform=linux/amd64 python:3.11-slim
# CPU-only PyTorch (much smaller than the GPU build)
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# The app listens on port 7860 (Cloudflare: defaultPort in src/index.ts)
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
