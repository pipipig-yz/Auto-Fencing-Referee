<div align="right">
  <a href="README_EN.md">🇺🇸 English</a>
</div>

<h1 align="center">🤺 击剑智能裁判系统</h1>
<p align="center">基于 YOLOv8-Pose 的佩剑项目视频优先权自动判定</p>

---

## 项目介绍

本项目对奥运级佩剑比赛视频进行离线分析，自动判断每次交锋中哪位运动员拥有**优先权**（得分权）。

系统完整复现了 FIE 佩剑优先权规则：

- 通过姿态估计逐帧检测双方运动员的关节角度
- 识别**出手**（肘关节伸展）、**冲刺**（重心前移）、**弓步**（前膝伸展）等动作事件
- 按三步级联算法判断优先权归属
- 输出事件时间轴与关节角度曲线两张可视化图

**判断算法（三步级联）**

```
STEP 1: 手臂差 > 3帧？  → 先出手者得优先权
STEP 2: 弓步差 > 3帧？  → 先弓步者得优先权
STEP 3: 冲刺差 > 10帧？ → 先冲刺者得优先权
否则 → 同时动作
```

---

## 文件结构

```
Fencing Auto Referee/
├── code/
│   ├── main.py                      # 批量处理入口
│   ├── config.py                    # 全部可调参数
│   ├── requirements.txt
│   ├── PROJECT.md                   # 完整设计文档
│   │
│   ├── modules/
│   │   ├── pose_module.py           # YOLOv8-Pose 推理封装
│   │   ├── athlete_tracker.py       # 左右运动员分配 + 基线标定
│   │   ├── piste_module.py          # 剑道检测 + 运动员过滤
│   │   ├── light_module.py          # 裁判灯 HSV 检测
│   │   ├── audio_module.py          # 音频事件检测
│   │   └── event_aggregator.py      # 事件检测 + 优先权判断（核心）
│   │
│   ├── utils/
│   │   ├── angle_utils.py           # 关节角度计算（COCO-17 骨架）
│   │   ├── video_utils.py           # 视频读取 / 帧采样
│   │   ├── json_utils.py            # JSON 序列化
│   │   └── debug_utils.py           # 调试绘图工具
│   │
│   ├── visualization/
│   │   ├── timeline_plot.py         # 图1：水平泳道事件时间轴
│   │   └── angle_plot.py            # 图2：关节角度曲线（4子图）
│   │
│   └── scripts/
│       ├── debug_aggregator.py      # 单视频调试 + 事件帧图片
│       ├── batch_priority_report.py # 批量优先权步骤日志
│       ├── debug_attack_scan.py     # 出手检测逐帧调试
│       └── debug_lunge_scan.py      # 弓步检测逐帧调试
│
├── video/
│   └── 奥运-标记视频/
│       ├── 互中/                    # 互中动作视频（5个，含左侧/右侧标注）
│       └── 同时/                    # 同时动作视频（5个）
│
├── models/                          # YOLOv8 权重（本地保留，不入库）
├── reference/                       # FIE 规则手册 + 参考论文
└── present file/                    # 项目演示文稿
```

---

## 安装

**环境要求：** Python 3.10+，建议使用虚拟环境

```bash
cd code
pip install -r requirements.txt
```

**下载模型权重**（首次运行时 ultralytics 会自动下载）：

```python
from ultralytics import YOLO
YOLO("yolov8x-pose.pt")   # 自动下载到当前目录
```

将下载好的 `yolov8x-pose.pt` 移动到 `models/` 目录，然后确认 `config.py` 中路径正确：

```python
MODEL_PATH  = r"<项目根目录>\models\yolov8x-pose.pt"
VIDEO_ROOT  = r"<项目根目录>\video\奥运-标记视频"
OUTPUT_ROOT = r"<项目根目录>\code\output"
```

---

## 运行

所有命令均在 `code/` 目录下执行。

### `main.py` — 批量处理

对全部标注视频跑完整流程，保存可视化图，打印准确率汇总表。

```bash
# 处理 互中/ 和 同时/ 下所有视频
python main.py

# 处理单个视频
python main.py --video "..\video\奥运-标记视频\互中\互中1左侧.mp4"

# 只处理指定文件夹
python main.py --folders 互中

# 跳过剑道检测（剑道不可见时使用）
python main.py --no-piste
```

**控制台输出示例：**

```
============================================================
  Fencing Auto Referee — Batch Run
  Videos : 10
  Output : output/
============================================================

[1/10]  互中1左侧.mp4 … LEFT_SCORES  ✓
[2/10]  互中2右侧.mp4 … RIGHT_SCORES ✓
...

============================================================
  ACCURACY SUMMARY
============================================================
  Video                                GT                    Predicted             Match
  互中1左侧                            LEFT_SCORES           LEFT_SCORES           ✓
  互中2右侧                            RIGHT_SCORES          RIGHT_SCORES          ✓
  ...
  Labelled videos : 10
  Correct         : X
  Accuracy        : X.X%
============================================================
```

**文件输出（每个视频一个子目录）：**

```
output/
└── 互中1左侧/
    ├── timeline.png       # 事件时间轴
    ├── joint_angles.png   # 关节角度曲线
    └── summary.txt        # 判决 + 关键时间戳
```

---

### `scripts/debug_aggregator.py` — 单视频事件调试

运行完整流程，**为每个检测到的事件保存带骨架标注的帧图片**，并打印优先权判决。

```bash
# 默认视频（互中1左侧.mp4）
python scripts/debug_aggregator.py

# 指定视频
python scripts/debug_aggregator.py --video "..\video\奥运-标记视频\互中\互中1左侧.mp4"

# 打印逐帧弓步数据表，只保存弓步事件帧
python scripts/debug_aggregator.py --lunge
```

**输出：** `output/aggregator/<视频名>/00123_L_ATTACK_START.jpg` 等事件帧图片

---

### `scripts/batch_priority_report.py` — 批量优先权步骤日志

遍历所有视频，打印三步级联判断的**每一步决策细节**，用于调参分析。

```bash
python scripts/batch_priority_report.py
```

**输出示例：**

```
────────────────────────────────────────────────────────────
FILE    : 互中1左侧.mp4
VERDICT : LEFT_SCORES

Priority step 1: arm_gap=8 > threshold=3 → LEFT arm first
VERDICT: LEFT_SCORES (no penalty)
```

---

### `scripts/debug_attack_scan.py` / `debug_lunge_scan.py` — 逐帧调试

打印分析窗口内每一帧的出手 / 弓步检测数值，用于排查阈值问题。

```bash
python scripts/debug_attack_scan.py
python scripts/debug_lunge_scan.py
```

---

## 输出可视化说明

### 图1：事件时间轴（`timeline.png`）

水平泳道图，展示分析窗口内双方各类事件的时间分布。

- 每条泳道代表一类事件（出手、冲刺、弓步、暂停、灯光）
- 垂直线标注关键时间点
- 顶部标注判决结果

### 图2：关节角度曲线（`joint_angles.png`）

4 个子图（从上到下）：

| 子图 | 内容 |
|------|------|
| L Attack | 左方肘角（实线）+ 肩角（点划线） |
| L Lunge  | 左方前膝角（实线）+ 大腿夹角（虚线）+ 步幅比（右轴） |
| R Attack | 右方肘角 + 肩角 |
| R Lunge  | 右方前膝角 + 大腿夹角 + 步幅比 |

- 虚横线：基线值（静止阶段）
- 点线：检测阈值（左侧 y 轴标注角度值）
- 竖线：baseline / ARM↑ / LAUNCH / LUNGE / LIGHT ON 时间点
- 右侧面板：所有阈值 + 双方基线数值

---

## 视频数据集说明

| 文件名规律 | 真实标签 |
|-----------|---------|
| `互中*左侧*` | LEFT_SCORES |
| `互中*右侧*` | RIGHT_SCORES |
| `同时*` | SIMULTANEOUS |

共 10 个标注视频，覆盖互中（5个）和同时（5个）两种场景。

---

## 主要参数（`config.py`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ATTACK_ELBOW_ABS` | 140° | 肘角绝对阈值，超过则判定出手 |
| `ATTACK_SHOULDER_ABS` | 60° | 肩角绝对阈值 |
| `LUNGE_FRONT_KNEE_ABS` | 150° | 前膝角绝对阈值，超过则判定弓步 |
| `LUNGE_THIGH_ANGLE_L` | 100° | 左方大腿夹角阈值 |
| `LUNGE_THIGH_ANGLE_R` | 120° | 右方大腿夹角阈值 |
| `LUNGE_WIDTH_RATIO` | 2.0× | 步幅与基线之比阈值 |
| `SIMULTANEOUS_ARM_THRESHOLD` | 3帧 | 手臂动作差 ≤ 此值视为同时 |
| `TARGET_FPS` | 30 | 统一采样帧率 |

---

## 依赖

- [ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv8-Pose
- OpenCV
- NumPy
- Matplotlib
- librosa（音频模块）
