import io, os
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from model import load_model, predict

MODEL_PATH = os.getenv('MODEL_PATH', 'adam_lr1e-3.pt')    # weights file from the Kaggle Output tab
THRESHOLD = float(os.getenv('THRESHOLD', '0.80'))         # below this confidence → send to a person

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
    return {'status': 'ok', 'model': MODEL_PATH, 'threshold': THRESHOLD}


@app.post('/predict')
async def predict_endpoint(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(413, 'ไฟล์ใหญ่เกิน 10 MB')
    try:
        image = Image.open(io.BytesIO(data))
    except UnidentifiedImageError:
        raise HTTPException(400, 'ไม่ใช่ไฟล์รูปภาพ')
    return predict(model, image, THRESHOLD)
