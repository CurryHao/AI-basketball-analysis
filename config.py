"""YOLO 篮球投篮分析 - 配置."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- 模型 ----------
# 主模型：球 + 篮筐一起训练（train.py 产出），因为 COCO 预训练对小球/遮挡漏检严重
CUSTOM_MODEL = os.path.join(BASE_DIR, "weights", "ball_hoop.pt")
# 兜底：无自训练模型时，用 COCO 预训练权重检测球（效果较弱，仅演示）
FALLBACK_BALL_MODEL = "yolo11n.pt"

# ---------- 检测 ----------
CONF_THRESHOLD = 0.40          # 检测置信度阈值
BALL_CLASS_ID = 32             # COCO: sports ball
IMG_SIZE = 640

# ---------- 姿态 ----------
POSE_MIN_DETECTION_CONF = 0.5
POSE_MIN_TRACKING_CONF = 0.5

# ---------- 投篮判定 ----------
FRAME_SKIP = 2                 # 每 N 帧处理 1 帧（CPU 提速）
HEAD_BALL_MIN_DIST = 30        # 球与头的最小像素距离（像素，防把头误判成球）
HAND_BALL_DIST_RATIO = 0.12    # 球与手的距离 < 宽度*比例 视为持球
BALL_CONTINUITY_DIST = 100     # 球帧间最大位移（像素），防跳变
HOOP_MARGIN = 30               # 落点判定时篮筐高度的容差（像素）

# ---------- 输出 ----------
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "weights"), exist_ok=True)
