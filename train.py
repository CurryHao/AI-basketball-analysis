"""训练球+篮筐检测模型（YOLOv11）。

实测结论：COCO 预训练权重对篮球视频里的小球/遮挡球漏检严重，
所以球和篮筐一起用 Roboflow 数据集训练，不偷懒。

本机 CPU 只能小规模验证：
    python train.py --data data/basketball/data.yaml --epochs 10 --model yolo11n.pt
认真训练建议用 Colab 免费 T4：
    python train.py --data data/basketball/data.yaml --epochs 100 --model yolo11n.pt --device 0
"""
import argparse
import os
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="数据集 data.yaml 路径")
    parser.add_argument("--model", default="yolo11n.pt", help="预训练起点")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="cpu 或 0（GPU）")
    parser.add_argument("--name", default="ball_hoop", help="实验名")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, name=args.name,
                project="runs", exist_ok=True)

    best = os.path.join("runs", args.name, "weights", "best.pt")
    print(f"[train] 训练完成，最优权重: {best}")
    print(f"[train] 把它复制到 weights/ball_hoop.pt 即可启用")


if __name__ == "__main__":
    main()
