---
title: Produce Quality Inspector
emoji: 🍎
colorFrom: gray
colorTo: gray
sdk: docker
app_port: 7860
---

# Automated Produce Quality Inspector — Web App

Upload a photo of an apple, banana or orange → **ผ่าน / ทิ้ง / ส่งให้พนักงานตรวจ** + confidence + Grad-CAM heatmap.

## Files
| File | What it does |
|---|---|
| `adam_lr1e-3.pt` | Trained weights — copy from the Kaggle notebook's Output tab (`outputs/`) |
| `model.py` | Same Custom CNN as the notebook + preprocessing + prediction + Grad-CAM |
| `app.py` | FastAPI server: `GET /` page, `POST /predict` API, `GET /health` |
| `static/index.html` | Upload page |
| `Dockerfile` | For Hugging Face Spaces (or any Docker host) |

## Run on your computer
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app:app --reload
# open http://127.0.0.1:8000
```

## Settings (environment variables)
- `MODEL_PATH` — weights file (default `adam_lr1e-3.pt`)
- `THRESHOLD` — confidence below this → "ส่งให้พนักงานตรวจ" (default `0.80`)

## API
```bash
curl -F "file=@apple.jpg" http://127.0.0.1:8000/predict
```
Returns `label`, `confidence`, `decision` (PASS / REJECT / REVIEW), `probabilities`, `heatmap_png` (base64).
