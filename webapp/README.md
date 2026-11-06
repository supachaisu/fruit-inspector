# Automated Produce Quality Inspector — Web App

Upload a photo of an apple, banana or orange → **ผ่าน / ทิ้ง / ส่งให้พนักงานตรวจ** + confidence + Grad-CAM heatmap.

## Files
| File | What it does |
|---|---|
| `adam_lr1e-3.pt` | Trained weights — copy from the [Kaggle notebook](https://www.kaggle.com/code/supachaisuwanthip/cs461-checkin-baseline)'s Output tab (`outputs/`) |
| `model.py` | Same Custom CNN as the notebook + preprocessing + prediction + Grad-CAM |
| `app.py` | FastAPI server: `GET /` page, `POST /predict` and `POST /predict/multi` APIs, `GET /health` |
| `fusion.py` | Combines the per-photo results of 1–3 cameras into one verdict (`worst` / `mean`) |
| `tests/` | pytest tests for fusion and the endpoints |
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

### Several cameras, one fruit
```bash
curl -F "files=@cam1.jpg" -F "files=@cam2.jpg" -F "files=@cam3.jpg" \
     "http://127.0.0.1:8000/predict/multi?fusion=worst"
```
Send 1–3 photos of the **same** fruit. The model is unchanged: it runs once per photo, then the results are combined.
- `fusion=worst` (default): any confident REJECT → REJECT; otherwise the views disagree on the fruit, or any view is below the threshold → REVIEW; otherwise PASS.
- `fusion=mean`: average the class probabilities over the photos, then apply the same threshold.

Returns `decision`, `reason`, `label`, `confidence`, `max_rotten_score`, the averaged `probabilities`, and `views` (each photo's full `/predict` result plus `view` and `rotten_score`).

## Tests
```bash
pip install pytest httpx
python -m pytest tests      # run from webapp/, needs the weights file
```
