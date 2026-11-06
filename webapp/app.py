import io, os
from typing import Literal
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from fusion import fuse
from model import load_model, predict

MODEL_PATH = os.getenv('MODEL_PATH', 'adam_lr1e-3.pt')    # weights file from the Kaggle Output tab
THRESHOLD = float(os.getenv('THRESHOLD', '0.80'))         # below this confidence → send to a person
MAX_BYTES = 10 * 1024 * 1024
MAX_VIEWS = 3                                             # cameras photographing the same fruit

app = FastAPI(title='Automated Produce Quality Inspector')
model = load_model(MODEL_PATH)


@app.get('/')
def index():
    return FileResponse('static/index.html')


@app.get('/favicon.svg')
def favicon():
    return FileResponse('static/favicon.svg')


@app.get('/apple-touch-icon.png')
def apple_touch_icon():
    return FileResponse('static/apple-touch-icon.png')


@app.get('/health')
def health():
    return {'status': 'ok', 'model': MODEL_PATH, 'threshold': THRESHOLD, 'max_views': MAX_VIEWS}


async def read_image(file: UploadFile, prefix: str = '') -> Image.Image:
    """Read an upload and check it is an image under 10 MB. `prefix` names the image in error messages."""
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f'{prefix}ไฟล์ใหญ่เกิน 10 MB')
    try:
        image = Image.open(io.BytesIO(data))
        image.load()                     # decode now, so a broken file fails here and not inside predict()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, f'{prefix}ไม่ใช่ไฟล์รูปภาพ')
    return image


@app.post('/predict')
async def predict_endpoint(file: UploadFile = File(...)):
    return predict(model, await read_image(file), THRESHOLD)


@app.post('/predict/multi')
async def predict_multi(files: list[UploadFile] = File(...),
                        fusion: Literal['worst', 'mean'] = Query('worst')):
    """1–3 photos of the SAME fruit (one per camera) → one verdict."""
    if len(files) > MAX_VIEWS:
        raise HTTPException(400, f'ส่งรูปได้ไม่เกิน {MAX_VIEWS} รูปต่อผลไม้ 1 ลูก')
    images = [await read_image(f, f'รูปที่ {n}: ') for n, f in enumerate(files, start=1)]
    return fuse([predict(model, im, THRESHOLD) for im in images], THRESHOLD, fusion)
