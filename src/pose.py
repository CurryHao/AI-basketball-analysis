"""姿态估计：YOLOv11-pose（COCO 17 关键点），纯 PyTorch，CPU 可跑。

关键点索引（COCO）：
    0 鼻  5/6 左右肩  7/8 左右肘  9/10 左右腕
    11/12 左右髋  13/14 左右膝  15/16 左右踝
"""
import math
import numpy as np
from ultralytics import YOLO

import config

_NOSE = 0
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_ELBOW, _R_ELBOW = 7, 8
_L_WRIST, _R_WRIST = 9, 10
_L_HIP, _R_HIP = 11, 12
_L_KNEE, _R_KNEE = 13, 14
_L_ANKLE, _R_ANKLE = 15, 16

POSE_MODEL = "yolo11n-pose.pt"
KP_CONF_MIN = 0.3


def _angle(a, b, c):
    """b 为顶点，a-b-c 的夹角（度）。"""
    ba = np.array(a, dtype=float) - np.array(b, dtype=float)
    bc = np.array(c, dtype=float) - np.array(b, dtype=float)
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return None
    cos_v = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return round(math.degrees(math.acos(cos_v)), 1)


class PoseEstimator:
    def __init__(self, model: str = POSE_MODEL):
        self.model = YOLO(model)  # 首次运行自动下载 ~6MB

    def _side(self, xy, conf, l_idx, r_idx):
        if conf[l_idx] >= conf[r_idx]:
            return tuple(xy[l_idx]), conf[l_idx]
        return tuple(xy[r_idx]), conf[r_idx]

    def estimate(self, frame):
        """返回 dict：elbow_angle / knee_angle / hand / head / ok。"""
        out = {"ok": False}
        r = self.model.predict(frame, conf=config.CONF_THRESHOLD,
                               imgsz=config.IMG_SIZE, verbose=False)[0]
        if r.keypoints is None or len(r.keypoints) == 0:
            return out
        # 取置信度最高的人
        i = int(np.argmax(r.boxes.conf.cpu().numpy()))
        xy = r.keypoints.xy[i].cpu().numpy()      # (17, 2) 像素坐标
        conf = r.keypoints.conf[i].cpu().numpy()  # (17,)

        shoulder, _ = self._side(xy, conf, _L_SHOULDER, _R_SHOULDER)
        elbow, _ = self._side(xy, conf, _L_ELBOW, _R_ELBOW)
        wrist, _ = self._side(xy, conf, _L_WRIST, _R_WRIST)
        hip, _ = self._side(xy, conf, _L_HIP, _R_HIP)
        knee, _ = self._side(xy, conf, _L_KNEE, _R_KNEE)
        ankle, _ = self._side(xy, conf, _L_ANKLE, _R_ANKLE)

        def _ok(*idx):
            return all(conf[j] >= KP_CONF_MIN for j in idx)

        elbow_angle = _angle(shoulder, elbow, wrist) \
            if _ok(_L_SHOULDER, _L_ELBOW, _L_WRIST) or \
               _ok(_R_SHOULDER, _R_ELBOW, _R_WRIST) else None
        knee_angle = _angle(hip, knee, ankle) \
            if _ok(_L_HIP, _L_KNEE, _L_ANKLE) or \
               _ok(_R_HIP, _R_KNEE, _R_ANKLE) else None

        out.update({
            "ok": True,
            "elbow_angle": elbow_angle,
            "knee_angle": knee_angle,
            "elbow_xy": (int(elbow[0]), int(elbow[1])),
            "knee_xy": (int(knee[0]), int(knee[1])),
            "hand_xy": (int(wrist[0]), int(wrist[1])),
            "head_xy": (int(xy[_NOSE][0]), int(xy[_NOSE][1])),
        })
        return out

    def close(self):
        pass
