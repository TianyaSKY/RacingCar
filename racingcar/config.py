# config.py
import math

# --- 窗口设置 ---
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600
FOV = 45

# --- 物理与控制 ---
CAR_BASE_SPEED = 0.8 # 稍微快一点
TURN_SPEED = 2.5     # 转向速度 (度/帧)
ACCELERATION = 0.02
FRICTION = 0.96      # 摩擦力稍微大一点，方便过弯

OBSTACLE_COUNT = 15

# --- 赛道生成 ---
ROAD_WIDTH = 10.0
OFF_ROAD_TOLERANCE = 2.0
TRACK_POINTS = 20    # 控制点数量
TRACK_RADIUS = 150   # 赛道大概半径
TRACK_VARIANCE = 60  # 赛道扭曲程度 (随机半径波动范围)

# --- 颜色定义 ---
COLOR_SKY = (0.5, 0.8, 1.0)
COLOR_GROUND = (0.1, 0.4, 0.1) # 纯色地面，简单点
COLOR_GROUND_LIGHT = (0.2, 0.5, 0.2) # 浅色地面
COLOR_GROUND_DARK = (0.1, 0.3, 0.1) # 深色地面
COLOR_ROAD = (0.2, 0.2, 0.2)
COLOR_ROAD_MARKING = (1.0, 1.0, 1.0) # 道路标记（白色）
COLOR_KERB_RED = (0.8, 0.1, 0.1) # 路肩红
COLOR_KERB_WHITE = (0.9, 0.9, 0.9) # 路肩白

COLOR_CAR_BODY = (1.0, 0.2, 0.2)
COLOR_CAR_WINDOW = (0.2, 0.2, 0.5)
COLOR_CAR_WHEEL = (0.1, 0.1, 0.1)

COLOR_OBSTACLE_BODY = (0.8, 0.6, 0.1)
COLOR_OBSTACLE_TOP = (0.5, 0.4, 0.0)

# --- UI ---
COLOR_TEXT_SCORE = (1, 1, 0)
COLOR_TEXT_GAME_OVER = (1, 0, 0)
COLOR_TEXT_INFO = (1, 1, 1)

# -- 奖励函数 ---
REWARD_STEP_PENALTY = -0.8
REWARD_COLLISION_PENALTY = -50  # 碰撞惩罚
REWARD_OUT_OF_BOUNDS_PENALTY = -100 # 出界惩罚
REWARD_SPEED_REWARD = 2 # 速度奖励
REWARD_TRACK_CENTER_REWARD = 0.05 # 保持在赛道上奖励