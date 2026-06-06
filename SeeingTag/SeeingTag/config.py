"""
配置加载模块
负责读取和解析 tag_config.json 配置文件
"""

import json
from typing import Dict
import numpy as np


def truncate_to_2_decimals(value: float) -> float:
    """
    截断浮点数到两位小数（直接舍弃第三位及以后，不四舍五入）
    例如：2.357 → 2.35, -1.239 → -1.23
    参数:
        value: 原始浮点数
    返回:
        截断后的浮点数
    """
    return int(value * 100) / 100.0


def load_config(path: str) -> dict:
    """
    加载配置文件
    参数:
        path: 配置文件路径
    返回:
        配置字典，包含以下字段：
            - tag_size_m: 标签物理尺寸（米）
            - min_tag_width: 最小标签像素宽度
            - unity_ip: Unity目标IP
            - unity_port: Unity目标端口
            - tag_positions: 固定标签世界坐标字典 {id: np.array([x,y,z])}
            - car_tag_id: 车载标签ID
            - homography_update_mode: H矩阵更新模式
            - homography_update_interval: H矩阵更新间隔
    """
    with open(path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    # 解析标签世界坐标
    tag_positions: Dict[int, np.ndarray] = {}
    for tid, pos in cfg["tag_world_positions"].items():
        tag_positions[int(tid)] = np.array([pos["x"], pos["y"], pos["z"]], dtype=np.float64)
    
    # 获取 H 矩阵更新模式配置，默认为 "interval"
    homography_mode = cfg.get("homography_update_mode", "interval")
    # 验证模式值是否合法
    valid_modes = ("once", "interval", "every_frame")
    if homography_mode not in valid_modes:
        print(f"[WARN] 无效的 homography_update_mode: {homography_mode}，使用默认值 'interval'")
        homography_mode = "interval"
    
    # 获取更新间隔，默认 30 帧
    homography_interval = cfg.get("homography_update_interval", 30)
    if homography_interval < 1:
        homography_interval = 30
    
    return {
        "tag_size_m": cfg["tag_size_m"],
        "min_tag_width": cfg.get("min_tag_width_pixels", 10),
        "unity_ip": cfg["unity_ip"],
        "unity_port": cfg["unity_port"],
        "tag_positions": tag_positions,
        "car_tag_id": cfg["car_tag_id"],
        "homography_update_mode": homography_mode,
        "homography_update_interval": homography_interval,
    }
