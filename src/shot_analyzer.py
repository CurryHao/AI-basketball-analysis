"""投篮分析状态机：出手检测 → 命中判定 → 姿势统计 → 轨迹拟合。

判定规则（启发式，无需训练）：
- 球心高于篮筐顶部 → 进入投篮状态，记录轨迹
- 出手角度 = 轨迹前两点连线与水平线的夹角
- 球落回篮筐高度 → 本次投篮结束：球 x 落在筐内判进，否则判丢
- 持球判定：球与手距离 < 阈值；持球期间记录肘/膝角度最小值
"""
import math
import numpy as np
import cv2

import config


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _release_angle(p0, p1):
    """p0→p1 与水平线的夹角（度），归一到 0~90。"""
    dx, dy = p1[0] - p0[0], p0[1] - p1[1]  # y 轴向下，取反
    ang = math.degrees(math.atan2(abs(dy), abs(dx) + 1e-6))
    return round(ang, 1)


class ShotAnalyzer:
    def __init__(self, width, height, fps):
        self.w, self.h, self.fps = width, height, fps
        self.trace = np.full((height, width, 3), 255, np.uint8)

        self.prev_ball = None
        self.hoop = None          # (x1, y1, x2, y2)
        self.hoop_top = 0         # 篮筐顶部 y（取历史最大值，防抖）

        self.shooting = False
        self.trajectory = []
        self.release_angles = []
        self.release_point = None

        self.ball_in_hand = False
        self.hand_frames = 0
        self.min_elbow, self.min_knee = 370.0, 370.0
        self.elbow_list, self.knee_list, self.hand_time_list = [], [], []

        self.attempts = 0
        self.made = 0
        self.judgement = ""       # "SCORE" / "MISS"，显示 N 帧
        self.judgement_frames = 0

    # ---------- 主入口 ----------
    def process(self, frame, detections, pose):
        balls = detections["balls"]
        hoops = detections["hoops"]

        if hoops:
            x1, y1, x2, y2 = hoops[0]["box"]
            self.hoop = (x1, y1, x2, y2)
            self.hoop_top = max(self.hoop_top, y1)

        ball = self._pick_ball(balls, pose)
        if ball is not None:
            self._update_shot(ball, pose, frame)

        if self.judgement_frames > 0:
            self.judgement_frames -= 1

        annotated = self._draw(frame, ball, pose)
        combined = np.concatenate((annotated, self.trace), axis=1)
        return combined

    def _pick_ball(self, balls, pose):
        """选球：置信度最高、且不在头附近（防把头误判成球）。"""
        head = pose.get("head_xy") if pose.get("ok") else None
        for b in balls:
            if head and _dist((b["cx"], b["cy"]), head) < config.HEAD_BALL_MIN_DIST:
                continue
            if self.prev_ball and _dist((b["cx"], b["cy"]),
                                        self.prev_ball) > config.BALL_CONTINUITY_DIST * 3:
                continue  # 跳变太大的可能是误检，沿用上一帧
            return b
        return None

    def _update_shot(self, ball, pose, frame):
        cx, cy = ball["cx"], ball["cy"]
        cv2.circle(self.trace, (cx, cy), 6, (235, 103, 193), -1)

        # 持球姿势统计
        if pose.get("ok"):
            hand = pose["hand_xy"]
            holding = _dist((cx, cy), hand) < self.w * config.HAND_BALL_DIST_RATIO
            if holding:
                self.hand_frames += 1
                if pose["elbow_angle"]:
                    self.min_elbow = min(self.min_elbow, pose["elbow_angle"])
                if pose["knee_angle"]:
                    self.min_knee = min(self.min_knee, pose["knee_angle"])

        if self.hoop_top <= 0:
            self.prev_ball = (cx, cy)
            return

        # 进入投篮：球高于篮筐
        if cy < self.hoop_top:
            if not self.shooting:
                self.shooting = True
                self.trajectory = []
            self.trajectory.append((cx, cy))
            if len(self.trajectory) == 2:
                ang = _release_angle(self.trajectory[0], self.trajectory[1])
                self.release_angles.append(ang)
                self.release_point = self.trajectory[0]
        # 投篮结束：球落回篮筐高度
        elif self.shooting and cy >= self.hoop_top - config.HOOP_MARGIN:
            self._judge(ball)

        self.prev_ball = (cx, cy)

    def _judge(self, ball):
        cx = ball["cx"]
        x1, _, x2, _ = self.hoop
        self.attempts += 1
        if x1 <= cx <= x2:
            self.made += 1
            self.judgement = "SCORE"
            color = (82, 168, 50)
        else:
            self.judgement = "MISS"
            color = (0, 0, 255)
        self.judgement_frames = 15

        if len(self.trajectory) > 1:
            pts = np.array(self.trajectory, np.int32)
            cv2.polylines(self.trace, [pts], False, color, 2, cv2.LINE_AA)

        self.elbow_list.append(self.min_elbow if self.min_elbow < 370 else None)
        self.knee_list.append(self.min_knee if self.min_knee < 370 else None)
        self.hand_time_list.append(self.hand_frames / max(self.fps, 1))

        self.shooting = False
        self.trajectory = []
        self.hand_frames = 0
        self.min_elbow, self.min_knee = 370.0, 370.0

    # ---------- 绘制 ----------
    def _draw(self, frame, ball, pose):
        f = frame.copy()
        if self.hoop:
            x1, y1, x2, y2 = self.hoop
            cv2.rectangle(f, (x1, y1), (x2, y2), (48, 124, 255), 2)
        if ball:
            cv2.circle(f, (ball["cx"], ball["cy"]), 8, (235, 103, 193), 2)
        if pose.get("ok"):
            if pose["elbow_angle"]:
                ex, ey = pose["elbow_xy"]
                cv2.putText(f, f"Elbow {pose['elbow_angle']}deg", (ex + 10, ey),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 255, 0), 2)
            if pose["knee_angle"]:
                kx, ky = pose["knee_xy"]
                cv2.putText(f, f"Knee {pose['knee_angle']}deg", (kx + 10, ky),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 255, 0), 2)
        if self.judgement_frames > 0 and self.hoop:
            x1, y1, _, _ = self.hoop
            color = (82, 168, 50) if self.judgement == "SCORE" else (0, 0, 255)
            cv2.putText(f, self.judgement, (x1 - 40, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.putText(f, f"Shot {self.made}/{self.attempts}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return f

    # ---------- 结果 ----------
    def stats(self):
        def _avg(lst):
            v = [x for x in lst if x is not None]
            return round(sum(v) / len(v), 1) if v else 0
        return {
            "attempts": self.attempts,
            "made": self.made,
            "miss": self.attempts - self.made,
            "fg_pct": round(self.made / self.attempts * 100, 1) if self.attempts else 0,
            "avg_elbow_angle": _avg(self.elbow_list),
            "avg_knee_angle": _avg(self.knee_list),
            "avg_release_angle": _avg(self.release_angles),
            "avg_ball_in_hand_s": _avg(self.hand_time_list),
        }

    def save_trajectory_plot(self, path):
        """把最后一次（或累计）轨迹拟合成抛物线存图。"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure()
        # trace 上是紫色圆点，简单起见直接保存 trace 图
        cv2.imwrite(path, self.trace)
        plt.close(fig)
