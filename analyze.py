"""命令行分析：输入投篮视频 → 输出标注视频 + 统计 JSON。

    python analyze.py input.mp4 -o output.mp4
"""
import argparse
import json
import os
import sys
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.detector import BallHoopDetector
from src.pose import PoseEstimator
from src.shot_analyzer import ShotAnalyzer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="输入视频路径")
    parser.add_argument("-o", "--output", default=None, help="输出标注视频路径")
    parser.add_argument("--no-pose", action="store_true", help="关闭姿态估计（更快）")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"打不开视频: {args.input}")
        sys.exit(1)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[analyze] {w}x{h} @ {fps:.0f}fps, 共 {total} 帧")

    detector = BallHoopDetector()
    pose_est = None if args.no_pose else PoseEstimator()
    analyzer = ShotAnalyzer(w, h, fps)

    out_path = args.output or os.path.join(
        config.OUTPUT_DIR, os.path.splitext(os.path.basename(args.input))[0] + "_analyzed.mp4")
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
        pose = pose_est.estimate(frame) if pose_est else {"ok": False}
        combined = analyzer.process(frame, detections, pose)
        writer.write(combined)
        if idx % 60 == 0:
            print(f"[analyze] {idx}/{total} 帧 "
                  f"(投篮 {analyzer.made}/{analyzer.attempts})")

    cap.release()
    writer.release()
    if pose_est:
        pose_est.close()

    stats = analyzer.stats()
    stats_path = os.path.splitext(out_path)[0] + ".json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[analyze] 标注视频: {out_path}")
    print(f"[analyze] 统计结果: {stats_path}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
