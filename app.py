"""Flask 网页：上传投篮视频 → 后台分析 → 看标注视频和统计。"""
import json
import os
import sys
import uuid
import cv2
from flask import Flask, request, render_template, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.detector import BallHoopDetector
from src.pose import PoseEstimator
from src.shot_analyzer import ShotAnalyzer

app = Flask(__name__)
ALLOWED = {"mp4", "mov", "avi", "mkv"}


def analyze_video(in_path, out_path):
    cap = cv2.VideoCapture(in_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    detector = BallHoopDetector()
    pose_est = PoseEstimator()
    analyzer = ShotAnalyzer(w, h, fps)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w * 2, h))
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx % config.FRAME_SKIP != 0:
            continue
        detections = detector.detect(frame)
        pose = pose_est.estimate(frame)
        writer.write(analyzer.process(frame, detections, pose))
    cap.release()
    writer.release()
    pose_est.close()
    return analyzer.stats()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        f = request.files.get("video")
        if not f or "." not in f.filename or \
                f.filename.rsplit(".", 1)[1].lower() not in ALLOWED:
            return "请上传 mp4/mov/avi/mkv 视频", 400
        uid = uuid.uuid4().hex[:8]
        name = secure_filename(f.filename)
        in_path = os.path.join(config.UPLOAD_DIR, f"{uid}_{name}")
        f.save(in_path)
        out_path = os.path.join(config.OUTPUT_DIR, f"{uid}_analyzed.mp4")
        stats = analyze_video(in_path, out_path)
        return render_template("result.html",
                               video=os.path.basename(out_path),
                               stats=stats)
    return render_template("index.html")


@app.route("/output/<path:fname>")
def output_file(fname):
    return send_from_directory(config.OUTPUT_DIR, fname)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
