# Automated Produce Quality Inspector — Web App

Upload a photo of an apple, banana or orange → **ผ่าน / ทิ้ง / ส่งให้พนักงานตรวจ** + confidence + Grad-CAM heatmap.

## Files
| File | What it does |
|---|---|
| `adam_lr1e-3.pt` | Trained weights — copy from the [Kaggle notebook](https://www.kaggle.com/code/supachaisuwanthip/cs461-checkin-baseline)'s Output tab (`outputs/`) |
| `model.py` | Same Custom CNN as the notebook + preprocessing + prediction + Grad-CAM |
| `app.py` | FastAPI server: `GET /` page, `POST /predict` API, `GET /health` |
| `static/index.html` | Upload page |
| `Dockerfile` | Container image for the FastAPI app (port 7860) |
| `src/index.ts` | Cloudflare Worker that forwards requests to the container |
| `wrangler.jsonc` | Cloudflare Workers + Containers config |

## Run on your computer
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app:app --reload
# open http://127.0.0.1:8000
```

## Deploy to Cloudflare
```bash
npm install
npm run deploy   # wrangler deploy — builds the Dockerfile and deploys the Worker + container
npm run logs     # wrangler tail
```

## Settings (environment variables)
- `MODEL_PATH` — weights file (default `adam_lr1e-3.pt`)
- `THRESHOLD` — confidence below this → "ส่งให้พนักงานตรวจ" (default `0.80`)

On Cloudflare these are set in `envVars` in `src/index.ts`.

## API
```bash
curl -F "file=@apple.jpg" http://127.0.0.1:8000/predict
```
Returns `label`, `confidence`, `decision` (PASS / REJECT / REVIEW), `probabilities`, `heatmap_png` (base64).
