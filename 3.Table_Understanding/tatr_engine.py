import sys, os, torch, torchvision.transforms as T
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.generic as generic; generic.is_tf_available = lambda: False
from transformers import AutoModelForObjectDetection
from config import TATR_MODEL, TATR_RESIZE, TATR_THRESHOLD

_CLS = {0:"table",1:"table column",2:"table row",3:"table column header",
        4:"table projected row header",5:"table spanning cell"}


def _resize_longest_edge(image):
    width, height = image.size
    scale = TATR_RESIZE / max(width, height)
    if scale == 1:
        return image
    return image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale)))
    )


_transform = T.Compose([
    T.Lambda(_resize_longest_edge), T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


class TATREngine:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def _load(self):
        self.model = AutoModelForObjectDetection.from_pretrained(TATR_MODEL)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self._loaded = True

    def detect(self, image):
        if not self._loaded:
            self._load()
        ow, oh = image.size
        t = _transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            o = self.model(pixel_values=t)
        p = torch.nn.functional.softmax(o.logits, dim=-1)
        s, l = p.max(dim=-1)
        no = o.logits.shape[-1] - 1
        keep = (s[0] > TATR_THRESHOLD) & (l[0] != no)
        cells = []
        for b, sc, lb in zip(o.pred_boxes[0][keep], s[0][keep], l[0][keep]):
            cx, cy, w, h = b.tolist()
            cells.append({
                "bbox": [round((cx-w/2)*ow,1), round((cy-h/2)*oh,1),
                         round((cx+w/2)*ow,1), round((cy+h/2)*oh,1)],
                "score": round(sc.item(), 3),
                "class_name": _CLS.get(lb.item(), "unknown"),
            })
        return cells
