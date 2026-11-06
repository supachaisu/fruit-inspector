"""Endpoints with the real weights (run from webapp/: python -m pytest tests)."""
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from app import app

client = TestClient(app)


def png(color=(200, 40, 30)):
    buf = io.BytesIO(); Image.new('RGB', (64, 48), color).save(buf, format='PNG')
    return buf.getvalue()


def files(n):
    return [('files', (f'cam{i}.png', png((40 * i, 120, 60)), 'image/png')) for i in range(1, n + 1)]


def test_health():
    assert client.get('/health').json()['max_views'] == 3


def test_old_predict_still_works():
    r = client.post('/predict', files={'file': ('a.png', png(), 'image/png')})
    assert r.status_code == 200
    assert {'label', 'confidence', 'decision', 'probabilities', 'heatmap_png'} <= set(r.json())
    bad = client.post('/predict', files={'file': ('a.txt', b'hello', 'text/plain')})
    assert (bad.status_code, bad.json()['detail']) == (400, 'ไม่ใช่ไฟล์รูปภาพ')


@pytest.mark.parametrize('n', [1, 2, 3])
@pytest.mark.parametrize('mode', ['worst', 'mean'])
def test_multi_1_to_3(n, mode):
    r = client.post(f'/predict/multi?fusion={mode}', files=files(n))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['mode'] == mode and body['n_views'] == n and len(body['views']) == n
    assert body['decision'] in {'PASS', 'REJECT', 'REVIEW'}
    assert all(v['heatmap_png'] for v in body['views'])


def test_multi_default_is_worst():
    assert client.post('/predict/multi', files=files(1)).json()['mode'] == 'worst'


def test_multi_single_view_matches_predict():
    img = png((180, 30, 20))
    one = client.post('/predict', files={'file': ('a.png', img, 'image/png')}).json()
    multi = client.post('/predict/multi', files=[('files', ('a.png', img, 'image/png'))]).json()
    assert (multi['decision'], multi['label'], multi['confidence']) == (one['decision'], one['label'], one['confidence'])


def test_multi_four_images_400():
    r = client.post('/predict/multi', files=files(4))
    assert r.status_code == 400 and '3' in r.json()['detail']


def test_multi_non_image_400_names_the_image():
    f = files(2) + [('files', ('notes.txt', b'not an image', 'text/plain'))]
    r = client.post('/predict/multi', files=f)
    assert (r.status_code, r.json()['detail']) == (400, 'รูปที่ 3: ไม่ใช่ไฟล์รูปภาพ')


def test_multi_truncated_image_400():
    r = client.post('/predict/multi', files=[('files', ('broken.png', png()[:60], 'image/png'))])
    assert (r.status_code, r.json()['detail']) == (400, 'รูปที่ 1: ไม่ใช่ไฟล์รูปภาพ')


def test_multi_too_big_413():
    r = client.post('/predict/multi', files=files(1) + [('files', ('big.png', b'0' * (10 * 1024 * 1024 + 1), 'image/png'))])
    assert (r.status_code, r.json()['detail']) == (413, 'รูปที่ 2: ไฟล์ใหญ่เกิน 10 MB')


def test_multi_bad_fusion_422():
    assert client.post('/predict/multi?fusion=median', files=files(1)).status_code == 422


def test_multi_no_files_422():
    assert client.post('/predict/multi').status_code == 422
