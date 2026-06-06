"""
摄像头管理模块
负责摄像头检测、选择、打开等操作
"""

import cv2
import sys
from typing import List, Optional


def list_available_cameras(max_test: int = 5) -> List[int]:
    """
    检测可用的摄像头索引
    参数:
        max_test: 最大测试的摄像头索引数
    返回:
        可用的摄像头索引列表
    """
    available = []
    print("[Camera] 正在检测可用摄像头...")
    
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # 尝试读取一帧来验证摄像头真正可用
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                available.append(i)
                print(f"  [{i}] 摄像头可用 - 分辨率: {w}x{h}")
            cap.release()
    
    if not available:
        print("[Camera] 未检测到可用摄像头!")
    else:
        print(f"[Camera] 共检测到 {len(available)} 个可用摄像头")
    
    return available


def select_camera() -> int:
    """
    自动检测摄像头并让用户选择
    返回:
        用户选择的摄像头索引
    """
    available = list_available_cameras()
    
    if not available:
        print("[ERROR] 没有可用的摄像头!")
        sys.exit(1)
    
    if len(available) == 1:
        print(f"[Camera] 只有一个摄像头，自动选择: {available[0]}")
        return available[0]
    
    # 多个摄像头，让用户选择
    print("\n请选择要使用的摄像头:")
    for i in available:
        print(f"  输入 {i} 选择摄像头 {i}")
    
    while True:
        try:
            choice = input("请输入摄像头索引: ").strip()
            choice = int(choice)
            if choice in available:
                print(f"[Camera] 已选择摄像头: {choice}")
                return choice
            else:
                print(f"[ERROR] 无效选择，请输入: {available}")
        except ValueError:
            print("[ERROR] 请输入数字!")
        except KeyboardInterrupt:
            print("\n[INFO] 用户取消")
            sys.exit(0)


def open_camera(camera_index: Optional[int] = None) -> cv2.VideoCapture:
    """
    打开本地 USB 摄像头
    参数:
        camera_index: 指定摄像头索引，None 则自动选择
    返回:
        cv2.VideoCapture 对象
    """
    # 如果没有指定摄像头索引，则自动检测并选择
    if camera_index is None:
        camera_index = select_camera()
    
    print(f"[Camera] 正在打开摄像头 {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"[ERROR] 无法打开摄像头 {camera_index}!")
        sys.exit(1)

    # 设置 MJPG 格式
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    # 尝试不同的分辨率（优先使用摄像头原生分辨率）
    resolutions = [
        (1920, 1080),  # 摄像头原生分辨率
        (1280, 720),
        (640, 480),
    ]
    
    for w, h in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w == w and actual_h == h:
            print(f"[Camera] 分辨率设置成功: {w}x{h}")
            break
    else:
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Camera] 使用摄像头默认分辨率: {actual_w}x{actual_h}")
    
    # 设置帧率
    cap.set(cv2.CAP_PROP_FPS, 30)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[Camera] 帧率: {actual_fps:.1f} FPS")
    
    # 设置缓冲区大小为1，减少延迟
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    return cap
