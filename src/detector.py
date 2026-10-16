"""检测器：球 + 篮筐。

优先用自训练模型（weights/ball_hoop.pt，train.py 产出）；
没有时兜底用 COCO 预训练权重检测球（对小球/遮挡漏检较多，仅演示）。
"""
import os
from ultralytics import YOLO

import config


class BallHoopDetector:
    def __init__(self,
                 custom_model: str = config.CUSTOM_MODEL,
                 conf: float = config.CONF_THRESHOLD):
        self.conf = conf
        self.mode = "custom"
        if os.path.exists(custom_model):
            self.model = YOLO(custom_model)
            names = [n.lower() for n in self.model.names.values()]
            # 先判 ball（含 basketball），剩下的再判 hoop，避免 "basketball"
            # 同时命中 "ball" 和 "basket" 两个关键字
            self.ball_ids = {i for i, n in enumerate(names) if "ball" in n}
            self.hoop_ids = {i for i, n in enumerate(names)
                             if i not in self.ball_ids and any(
                                 k in n for k in ("hoop", "rim", "net", "backboard", "basket"))}
            print(f"[detector] custom model: {custom_model} "
                  f"ball_ids={self.ball_ids} hoop_ids={self.hoop_ids}")
        else:
            self.mode = "coco_fallback"
            self.model = YOLO(config.FALLBACK_BALL_MODEL)
            self.ball_ids = {config.BALL_CLASS_ID}
            self.hoop_ids = set()
            print("[detector] no custom model, COCO fallback for ball only "
                  "(train one with train.py for hoop + better ball)")

    def _boxes_of(self, result, ids):
        out = []
        if result.boxes is None:
            return out
        for box in result.boxes:
            if int(box.cls[0]) not in ids:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            out.append({
                "box": (x1, y1, x2, y2),
                "conf": float(box.conf[0]),
                "cx": (x1 + x2) // 2,
                "cy": (y1 + y2) // 2,
            })
        out.sort(key=lambda b: -b["conf"])
        return out

    def detect(self, frame):
        """返回 {"balls": [...], "hoops": [...]}，按置信度降序。"""
        r = self.model.predict(frame, conf=self.conf,
                               imgsz=config.IMG_SIZE, verbose=False)[0]
        return {
            "balls": self._boxes_of(r, self.ball_ids),
            "hoops": self._boxes_of(r, self.hoop_ids),
        }
