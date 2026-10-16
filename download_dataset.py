"""下载 Roboflow Universe 上的篮球数据集（YOLO 格式）。

需要免费的 Roboflow API key：
    1. 注册 https://roboflow.com
    2. Settings → API Keys 复制
    3. export ROBOFLOW_API_KEY=你的key 后运行本脚本

数据集（球+篮筐已标注好，无需人工标注）：
    - sc-xqmxu/basketball-and-net-detection v7：5482 张，最大
    - basketball-detection-b977c/basketball-detection-sskux v7：~1300 张，较新
"""
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="sc-xqmxu")
    parser.add_argument("--project", default="basketball-and-net-detection")
    parser.add_argument("--version", type=int, default=7)
    parser.add_argument("--out", default=os.path.join("data", "basketball"))
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("请先设置环境变量 ROBOFLOW_API_KEY（见本文件顶部说明）")
        sys.exit(1)

    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)
    dataset = project.version(args.version).download("yolov11", location=args.out)
    print(f"[download] 数据集已下载到: {dataset.location}")
    # 类别从 data.yaml 读取（roboflow 新版 Dataset 对象无 class_names 属性）
    data_yaml = os.path.join(args.out, "data.yaml")
    if os.path.exists(data_yaml):
        import yaml
        with open(data_yaml) as f:
            names = yaml.safe_load(f).get("names")
        print(f"[download] 类别: {names}")


if __name__ == "__main__":
    main()
