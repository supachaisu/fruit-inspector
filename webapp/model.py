"""Model, preprocessing, prediction and Grad-CAM — must match the training notebook exactly."""
import base64, io
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torchvision import transforms

CLASSES = ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges']
FRUIT_TH = {'apples': 'แอปเปิ้ล', 'banana': 'กล้วย', 'oranges': 'ส้ม'}
ROTTEN = {3, 4, 5}
IMG_SIZE = 224

PREPROCESS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class FruitCNN(nn.Module):  # identical to the notebook
    def __init__(self, n_classes=6, chans=(32, 64, 128, 256), p=0.4):
        super().__init__()
        layers, c_in = [], 3
        for c in chans:
            layers += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(p),
                                  nn.Linear(c_in, 128), nn.ReLU(inplace=True), nn.Dropout(p),
                                  nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def load_model(path):
    model = FruitCNN()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    return model.eval()


def _to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO(); img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def _jet(x):
    """Small colormap (blue → yellow → red) without matplotlib."""
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], -1)


def predict(model, image: Image.Image, threshold: float):
    image = image.convert('RGB')
    x = PREPROCESS(image).unsqueeze(0).requires_grad_(False)

    # Grad-CAM on the last Conv2d (features[-4]): capture its output and gradient
    store = {}
    layer = model.features[-4]
    h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__('act', o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__('grad', go[0]))
    try:
        logits = model(x)
        probs = logits.softmax(1)[0]
        idx = int(probs.argmax())
        model.zero_grad(); logits[0, idx].backward()
    finally:
        h1.remove(); h2.remove()

    weights = store['grad'].mean(dim=(2, 3), keepdim=True)          # how much each feature map matters
    cam = F.relu((weights * store['act']).sum(1, keepdim=True))       # weighted sum, keep positive
    cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    base = np.asarray(image.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255
    overlay = Image.fromarray((np.clip(0.55 * base + 0.45 * _jet(cam.detach().numpy()), 0, 1) * 255).astype(np.uint8))

    conf = float(probs[idx])
    rotten = idx in ROTTEN
    fruit = CLASSES[idx].replace('fresh', '').replace('rotten', '')
    if conf < threshold:
        decision = 'REVIEW'   # not confident → send to a person
    else:
        decision = 'REJECT' if rotten else 'PASS'
    return {
        'label': CLASSES[idx],
        'fruit_th': FRUIT_TH[fruit],
        'state_th': 'เน่า' if rotten else 'สด',
        'confidence': round(conf, 4),
        'decision': decision,
        'threshold': threshold,
        'probabilities': {c: round(float(p), 4) for c, p in zip(CLASSES, probs)},
        'heatmap_png': _to_b64_png(overlay),
    }
