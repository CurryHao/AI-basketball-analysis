# YOLO 篮球投篮分析 🏀

用 **YOLO 目标检测 + MediaPipe 姿态估计** 分析篮球投篮：检测每次出手、判定命中与否、
分析投篮姿势（肘/膝关节角度、出手角度、持球时间），并输出标注视频。

这是对 [AI-basketball-analysis](https://github.com/chonyy/AI-basketball-analysis)
的现代化重写：YOLOv11 替代老 TF 检测模型，YOLOv11-pose 替代 OpenPose
（纯 PyTorch、CPU 可跑，无需 CUDA）。

## 快速开始

```bash
pip install -r requirements.txt

# 命令行分析
python analyze.py input.mp4 -o output.mp4

# 网页版
python app.py  # 打开 http://localhost:5000
```

## 实现原理

1. **检测**：YOLOv11 自训练模型，同时检测篮球和篮筐（`weights/ball_hoop.pt`，
   `train.py` 产出）。实测 COCO 预训练权重对小球/遮挡漏检严重，所以球也一起训练；
   无自训练模型时兜底用 COCO 权重只检测球。
2. **姿态**：YOLOv11-pose 提取 COCO 17 关键点，余弦定理算肘（肩-肘-腕）、
   膝（髋-膝-踝）关节角度。纯 PyTorch，CPU 可跑。
3. **判定**（纯规则状态机）：
   - 球心高于篮筐顶部 → 进入投篮状态，记录轨迹
   - 出手角度 = 轨迹前两点连线与水平线夹角
   - 球落回篮筐高度 → 判结束：球 x 落在筐内为进，否则为丢
   - 持球（球与手距离 < 阈值）期间记录肘/膝角度最小值
4. **输出**：左=标注视频（球/筐/角度/SCORE-MISS），右=球轨迹图；
   JSON 统计：命中数、命中率、平均角度等。

## 训练篮筐检测模型

```bash
export ROBOFLOW_API_KEY=你的key   # roboflow.com 免费注册
python download_dataset.py        # 下载已标注好的球+篮筐数据集（5482 张）
python train.py --data data/basketball/data.yaml --epochs 100 --device 0
# 把 runs/ball_hoop/weights/best.pt 复制到 weights/ball_hoop.pt
```

CPU 机器只建议小规模验证（`--epochs 10`）；认真训练请用 Colab 免费 GPU——
打开 `colab_train.ipynb`，按顺序点每个格子的 ▶ 运行即可（需要 Roboflow API key，T4 上 100 epoch 约 40–60 分钟）。

## 数据集来源

- [Basketball and Net Detection](https://universe.roboflow.com/sc-xqmxu/basketball-and-net-detection)
  （Roboflow Universe，CC BY 4.0）——训练用 `basketball` + `net` 两类，
  5482 张已标注图像。本仓库不含数据集本身，用 `download_dataset.py` 下载。

```
@misc{basketball-and-net-detection_dataset,
  title = { Basketball and Net Detection Dataset },
  type = { Open Source Dataset },
  howpublished = { \url{ https://universe.roboflow.com/sc-xqmxu/basketball-and-net-detection } },
  url = { https://universe.roboflow.com/sc-xqmxu/basketball-and-net-detection },
  license = { CC BY 4.0 }
}
```

## 局限

- 需要固定机位、篮筐清晰入镜；多人同框时姿态取置信度最高者
- 判定阈值是启发式的（见 `config.py`），换场地可能需要微调
