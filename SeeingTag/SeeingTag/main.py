"""
SeeingTag — ARUCO 智能车定位系统（Homography 平面映射版）
核心逻辑：检测标签 → 单应矩阵映射 → UDP 发给 Unity
特点：只需平面坐标 (X, Z)，无需相机内参标定
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import json
import socket
import time
import sys
from typing import Dict, List, Optional, Tuple


def load_config(path: str) -> dict:
    """
    加载配置文件
    参数:
        path: 配置文件路径
    返回:
        配置字典
    """
    with open(path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
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


def truncate_to_2_decimals(value: float) -> float:
    """
    截断浮点数到两位小数（直接舍弃第三位及以后，不四舍五入）
    例如：2.357 → 2.35, -1.239 → -1.23
    """
    return int(value * 100) / 100.0


class UdpSender:
    def __init__(self, target_ip: str, target_port: int):
        self.target = (target_ip, target_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0

    def send(self, x: float, z: float) -> bool:
        try:
            # 截断到两位小数
            x_truncated = truncate_to_2_decimals(x)
            z_truncated = truncate_to_2_decimals(z)
            
            data = {
                "type": "robot_position",
                "pos": [float(z_truncated), 0.0, float(x_truncated)],  # 修正：交换x和z的顺序
                "euler": [0.0, 0.0, 0.0],
                "seq": self.seq,
                "timestamp": time.time()
            }
            self.seq += 1
            self.sock.sendto(json.dumps(data).encode('utf-8'), self.target)
            print(f"[UDP] 发送成功 → {self.target[0]}:{self.target[1]}  pos=({z_truncated:.2f}, 0.0, {x_truncated:.2f})  seq={self.seq-1}")
            return True
        except Exception as e:
            print(f"[UDP] 发送失败: {e}")
            return False

    def close(self):
        self.sock.close()


class TagTracker:
    def __init__(self, cfg: dict):
        self.tag_size = cfg["tag_size_m"]
        self.min_tag_width = cfg["min_tag_width"]
        self.tag_positions = cfg["tag_positions"]
        self.car_tag_id = cfg["car_tag_id"]
        
        # H 矩阵更新模式配置
        self.homography_update_mode = cfg.get("homography_update_mode", "interval")
        self.homography_update_interval = cfg.get("homography_update_interval", 30)

        # ArUco 检测器
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters()
        self.parameters.adaptiveThreshWinSizeMin = 3
        self.parameters.adaptiveThreshWinSizeMax = 23
        self.parameters.adaptiveThreshWinSizeStep = 5
        self.parameters.adaptiveThreshConstant = 7
        self.parameters.perspectiveRemovePixelPerCell = 8
        self.parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self.parameters.cornerRefinementWinSize = 5
        self.detector = aruco.ArucoDetector(self.dictionary, self.parameters)

        # UDP 发送器
        self.udp: Optional[UdpSender] = UdpSender(cfg["unity_ip"], cfg["unity_port"])

        # 单应矩阵缓存
        self.H: Optional[np.ndarray] = None
        self.homography_valid = False
        # 标记 H 矩阵是否已初始化（用于 "once" 模式）
        self.homography_initialized = False

    @staticmethod
    def build_tag_corners_world(tag_center: np.ndarray, tag_size: float) -> np.ndarray:
        """返回标签4个角点在世界 XZ 平面上的坐标 (x, z)"""
        s2 = tag_size / 2
        local = np.array([
            [-s2, -s2, 0],
            [ s2, -s2, 0],
            [ s2,  s2, 0],
            [-s2,  s2, 0],
        ])
        corners_3d = tag_center + local
        return corners_3d[:, [0, 2]]  # 只保留 (x, z)

    def detect(self, gray: np.ndarray) -> List[dict]:
        """
        检测所有可见的 ARUCO 标签
        参数:
            gray: 灰度图像
        返回:
            检测结果列表，每个元素包含 id, corners, width_px
        """
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return []
        results = []
        for i, tag_id in enumerate(ids.flatten()):
            c = corners[i][0] if corners[i].shape[0] == 1 else corners[i]
            w = abs(c[0][0] - c[1][0])
            if w < self.min_tag_width:
                continue
            results.append({"id": int(tag_id), "corners": c.copy(), "width_px": w})
        return results

    def detect_car_only(self, gray: np.ndarray) -> Optional[dict]:
        """
        只检测车标签（优化性能）
        参数:
            gray: 灰度图像
        返回:
            车标签检测结果，未检测到则返回 None
        """
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return None
        
        for i, tag_id in enumerate(ids.flatten()):
            if tag_id == self.car_tag_id:
                c = corners[i][0] if corners[i].shape[0] == 1 else corners[i]
                w = abs(c[0][0] - c[1][0])
                if w >= self.min_tag_width:
                    return {"id": int(tag_id), "corners": c.copy(), "width_px": w}
        return None

    def compute_homography(self, detections: List[dict]) -> bool:
        """
        用固定标签的角点计算单应矩阵
        需要至少 4 个点（可以是 1 个完整标签或多个标签的组合）
        """
        src_pts = []  # 像素坐标 (u, v)
        dst_pts = []  # 世界坐标 (x, z)

        for d in detections:
            if d["id"] not in self.tag_positions:
                continue
            # 获取该标签的世界坐标角点
            world_corners_xz = self.build_tag_corners_world(
                self.tag_positions[d["id"]], self.tag_size
            )
            # 添加 4 个角点对应关系
            for j in range(4):
                src_pts.append(d["corners"][j].astype(np.float64))
                dst_pts.append(world_corners_xz[j])

        # 至少需要 4 个点才能计算单应矩阵
        if len(src_pts) < 4:
            self.homography_valid = False
            self.H = None
            return False

        src_pts = np.array(src_pts, dtype=np.float64)
        dst_pts = np.array(dst_pts, dtype=np.float64)

        # 使用 RANSAC 鲁棒估计单应矩阵
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

        if H is not None:
            self.H = H
            self.homography_valid = True
            inlier_count = int(np.sum(mask))
            print(f"[Homography] 计算成功: {len(src_pts)}个点, {inlier_count}个内点")
            return True
        else:
            self.homography_valid = False
            self.H = None
            return False

    def locate(self, detections: List[dict]) -> Tuple[Optional[float], Optional[float]]:
        """
        定位车标签，返回世界坐标 (x, z)
        此方法会重新计算 H 矩阵
        参数:
            detections: 所有检测到的标签列表
        返回:
            (x, z) 世界坐标，失败返回 (None, None)
        """
        # 第 1 步：更新单应矩阵
        self.compute_homography(detections)

        if not self.homography_valid or self.H is None:
            return None, None

        # 第 2 步：找到车标签的中心像素坐标
        car_det = next((d for d in detections if d["id"] == self.car_tag_id), None)
        if car_det is None:
            return None, None

        car_center_pixel = np.mean(car_det["corners"], axis=0)

        # 第 3 步：通过单应矩阵映射到世界坐标
        pixel_pt = np.array([[car_center_pixel[0], car_center_pixel[1]]], dtype=np.float64)
        world_pt = cv2.perspectiveTransform(pixel_pt.reshape(1, 1, 2), self.H)
        wx, wz = world_pt[0, 0, 0], world_pt[0, 0, 1]

        return float(wx), float(wz)

    def locate_with_cached_H(self, car_det: Optional[dict]) -> Tuple[Optional[float], Optional[float]]:
        """
        使用缓存的 H 矩阵定位车标签（不重新计算 H）
        参数:
            car_det: 车标签检测结果（来自 detect_car_only）
        返回:
            (x, z) 世界坐标，失败返回 (None, None)
        """
        # 检查 H 矩阵是否有效
        if not self.homography_valid or self.H is None:
            return None, None
        
        # 检查车标签是否检测到
        if car_det is None:
            return None, None

        # 计算车标签中心像素坐标
        car_center_pixel = np.mean(car_det["corners"], axis=0)

        # 通过缓存的 H 矩阵映射到世界坐标
        pixel_pt = np.array([[car_center_pixel[0], car_center_pixel[1]]], dtype=np.float64)
        world_pt = cv2.perspectiveTransform(pixel_pt.reshape(1, 1, 2), self.H)
        wx, wz = world_pt[0, 0, 0], world_pt[0, 0, 1]

        return float(wx), float(wz)

    def send_position(self, x: float, z: float):
        if self.udp is not None:
            self.udp.send(x, z)

    def draw_hud(self, frame: np.ndarray, detections: List[dict],
                 car_x: Optional[float], car_z: Optional[float], fps: float) -> np.ndarray:
        display = frame.copy()

        # 画标签边框
        for d in detections:
            color = (0, 255, 0) if d["id"] in self.tag_positions else (0, 128, 255)
            pts = d["corners"].astype(np.int32)
            cv2.polylines(display, [pts], True, color, 2)
            cx, cy = int(d['corners'][0][0]), int(d['corners'][0][1])
            label = "CAR" if d["id"] == self.car_tag_id else f"ID:{d['id']}"
            cv2.putText(display, label, (cx, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # HUD 信息
        h, w = frame.shape[:2]
        fixed_count = sum(1 for d in detections if d["id"] in self.tag_positions)
        car_visible = any(d["id"] == self.car_tag_id for d in detections)

        lines = []
        if car_x is not None and car_z is not None:
            # 截断到两位小数后显示
            car_x_display = truncate_to_2_decimals(car_x)
            car_z_display = truncate_to_2_decimals(car_z)
            lines.append(f"Car: X={car_x_display:.2f} Z={car_z_display:.2f}")
        lines.append(f"Fixed: {fixed_count}/{len(self.tag_positions)}  Car tag: {'YES' if car_visible else 'NO'}")
        if self.homography_valid:
            lines.append(f"Homography: OK")
        else:
            lines.append(f"Homography: Need >=4 points")

        for idx, line in enumerate(lines):
            cv2.putText(display, line, (10, 30 + idx * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 右下角 FPS
        cv2.putText(display, f"FPS: {fps:.0f}", (w - 110, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        return display

    def generate_bird_eye_view(self, frame: np.ndarray, detections: List[dict],
                               car_x: Optional[float], car_z: Optional[float]) -> Optional[np.ndarray]:
        """
        生成鸟瞰图（俯视图）：对原图进行逆透视变换
        将相机视角的图像映射到世界平面视角
        """
        if not self.homography_valid or self.H is None:
            return None

        h, w = frame.shape[:2]

        # 定义世界坐标系中的目标区域（根据场地大小）
        margin = 0.5  # 边距 0.5m
        world_min_x = -margin
        world_max_x = 5.0 + margin
        world_min_z = -margin
        world_max_z = 4.0 + margin

        # 计算缩放比例：让世界坐标适配到输出图像
        output_width = 800
        output_height = 640
        scale_x = output_width / (world_max_x - world_min_x)
        scale_z = output_height / (world_max_z - world_min_z)

        # 世界坐标的四个角点（齐次坐标）
        world_corners = np.array([
            [world_min_x, world_min_z, 1],
            [world_max_x, world_min_z, 1],
            [world_max_x, world_max_z, 1],
            [world_min_x, world_max_z, 1]
        ], dtype=np.float64)

        # H: 像素 → 世界，所以 H_inv: 世界 → 像素
        H_inv = np.linalg.inv(self.H)

        # 将世界坐标角点转换到像素坐标
        src_pts = []
        for corner in world_corners:
            pixel = H_inv @ corner
            pixel = pixel / pixel[2]  # 归一化
            src_pts.append([pixel[0], pixel[1]])
        src_pts = np.array(src_pts, dtype=np.float32)

        # 目标图像的四个角点（像素坐标）
        dst_pts = np.array([
            [0, 0],
            [output_width, 0],
            [output_width, output_height],
            [0, output_height]
        ], dtype=np.float32)

        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # 执行透视变换
        bird_eye = cv2.warpPerspective(frame, M, (output_width, output_height))

        # 绘制场地边界（红色矩形）
        boundary_color = (0, 0, 255)
        top_left = (int((0 - world_min_x) * scale_x), int((0 - world_min_z) * scale_z))
        top_right = (int((5.0 - world_min_x) * scale_x), int((0 - world_min_z) * scale_z))
        bottom_right = (int((5.0 - world_min_x) * scale_x), int((4.0 - world_min_z) * scale_z))
        bottom_left = (int((0 - world_min_x) * scale_x), int((4.0 - world_min_z) * scale_z))
        
        cv2.line(bird_eye, top_left, top_right, boundary_color, 2)
        cv2.line(bird_eye, top_right, bottom_right, boundary_color, 2)
        cv2.line(bird_eye, bottom_right, bottom_left, boundary_color, 2)
        cv2.line(bird_eye, bottom_left, top_left, boundary_color, 2)

        # 绘制固定标签位置（绿色圆圈）
        for tag_id, pos in self.tag_positions.items():
            bx = int((pos[0] - world_min_x) * scale_x)
            bz = int((pos[2] - world_min_z) * scale_z)
            cv2.circle(bird_eye, (bx, bz), 10, (0, 255, 0), -1)
            cv2.circle(bird_eye, (bx, bz), 12, (0, 200, 0), 2)
            cv2.putText(bird_eye, str(tag_id), (bx + 12, bz + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 绘制车标签位置（黄色圆圈）
        if car_x is not None and car_z is not None:
            bx = int((car_x - world_min_x) * scale_x)
            bz = int((car_z - world_min_z) * scale_z)
            # 确保坐标在图像范围内
            if 0 <= bx < output_width and 0 <= bz < output_height:
                cv2.circle(bird_eye, (bx, bz), 12, (0, 255, 255), -1)
                cv2.circle(bird_eye, (bx, bz), 14, (0, 200, 200), 2)
                cv2.putText(bird_eye, "CAR", (bx + 15, bz + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 添加标题
        cv2.putText(bird_eye, "Bird Eye View (Top-Down)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 添加坐标轴说明
        cv2.putText(bird_eye, f"X: 0-5m", (10, output_height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(bird_eye, f"Z: 0-4m", (10, output_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        return bird_eye

    def print_debug_info(self, detections: List[dict]):
        fixed = [d for d in detections if d["id"] in self.tag_positions]
        car_det = next((d for d in detections if d["id"] == self.car_tag_id), None)
        print(f"\n--- Debug ---")
        print(f"  Tags detected: {[d['id'] for d in detections]}")
        print(f"  Fixed tags: {len(fixed)}/{len(self.tag_positions)}")
        for d in fixed:
            print(f"    ID {d['id']}: corners[0]=({d['corners'][0][0]:.1f}, {d['corners'][0][1]:.1f}) w={d['width_px']}px")
        if car_det is not None:
            center = np.mean(car_det["corners"], axis=0)
            print(f"  Car tag(ID {self.car_tag_id}): center=({center[0]:.1f}, {center[1]:.1f}) w={car_det['width_px']}px")
        if self.homography_valid:
            print(f"  Homography: Valid")
        else:
            print(f"  Homography: Invalid (need >=4 points)")
        print("------------------------\n")

    def close(self):
        if self.udp is not None:
            self.udp.close()


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
        (1280, 720),   # 常用分辨率
        (640, 480),    # 备选分辨率
    ]
    
    for w, h in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        time.sleep(0.3)
        
        # 验证设置是否生效
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if actual_w == w and actual_h == h:
            print(f"[Camera] 成功设置分辨率: {w}x{h}")
            break
        else:
            print(f"[Camera] 分辨率 {w}x{h} 不支持，实际: {actual_w}x{actual_h}")

    # 获取最终设置
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join(chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4))
    print(f"[Camera] USB 摄像头 — {w}x{h} @ {fps:.0f}fps  编码={fourcc_str}")
    
    # 测试是否能正常读取帧
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print("[ERROR] 摄像头无法读取帧，请检查:")
        print("  1. 摄像头是否被其他程序占用")
        print("  2. 摄像头驱动是否正常")
        print("  3. 尝试重新插拔摄像头")
        cap.release()
        sys.exit(1)
    print(f"[Camera] 帧读取测试成功，图像尺寸: {test_frame.shape}")
    
    return cap


def run_calibration(cfg: dict):
    print("=== Calibration Mode — 不发 UDP，按 C 打印调试信息 ===")
    tracker = TagTracker(cfg)
    tracker.udp = None

    cap = open_camera()
    print(f"[Homography] 固定标签: {list(cfg['tag_positions'].keys())}")
    print(f"[Homography] 车标签ID: {cfg['car_tag_id']}")
    print(f"[Homography] 需要至少 4 个点（1个完整标签或多个标签组合）")

    fps_time, fps_count, cur_fps = time.time(), 0, 0.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            cv2.waitKey(10)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = tracker.detect(gray)
        car_x, car_z = tracker.locate(detections)

        fps_count += 1
        if time.time() - fps_time >= 1.0:
            cur_fps = fps_count
            fps_count = 0
            fps_time = time.time()

        display = tracker.draw_hud(frame, detections, car_x, car_z, cur_fps)
        
        # 生成鸟瞰图
        bird_eye = tracker.generate_bird_eye_view(frame, detections, car_x, car_z)

        # 同时显示两个窗口
        cv2.imshow("SeeingTag - Original View", display)
        if bird_eye is not None:
            cv2.imshow("SeeingTag - Bird Eye View", bird_eye)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            tracker.print_debug_info(detections)

    cap.release()
    cv2.destroyAllWindows()


def run_competition(cfg: dict):
    """
    比赛模式主循环
    支持三种 H 矩阵更新模式：
        - once: 启动时检测一次，之后不再更新
        - interval: 每 N 帧更新一次
        - every_frame: 每帧都更新
    """
    print("=== Competition Mode — 检测 → Homography映射 → UDP ===")
    print(f"  Unity → {cfg['unity_ip']}:{cfg['unity_port']}")
    
    # 获取 H 矩阵更新模式
    h_mode = cfg.get("homography_update_mode", "interval")
    h_interval = cfg.get("homography_update_interval", 30)
    print(f"  H矩阵更新模式: {h_mode}" + (f" (每{h_interval}帧)" if h_mode == "interval" else ""))

    tracker = TagTracker(cfg)
    cap = open_camera()
    print(f"[Homography] 固定标签: {list(cfg['tag_positions'].keys())}")
    print(f"[Homography] 车标签ID: {cfg['car_tag_id']}")
    
    # 等待 H 矩阵初始化（必须检测到全部 4 个固定标签）
    print(f"[Homography] 等待固定标签检测...")
    while not tracker.homography_valid:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = tracker.detect(gray)
        
        # 检查是否检测到全部 4 个固定标签
        fixed_ids = set(d["id"] for d in detections if d["id"] in tracker.tag_positions)
        fixed_count = len(fixed_ids)
        all_fixed_found = fixed_count == len(tracker.tag_positions)
        
        # 只有检测到全部固定标签才计算 H 矩阵
        if all_fixed_found:
            tracker.compute_homography(detections)
        
        # 显示等待状态（同时绘制检测框）
        display = frame.copy()
        
        # 绘制检测框
        for d in detections:
            color = (0, 255, 0) if d["id"] in tracker.tag_positions else (0, 128, 255)
            pts = d["corners"].astype(np.int32)
            cv2.polylines(display, [pts], True, color, 2)
            cx, cy = int(d['corners'][0][0]), int(d['corners'][0][1])
            label = "CAR" if d["id"] == tracker.car_tag_id else f"ID:{d['id']}"
            cv2.putText(display, label, (cx, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 显示等待提示
        status = "OK! Initializing..." if all_fixed_found else f"Waiting... ({fixed_count}/{len(tracker.tag_positions)})"
        cv2.putText(display, status, 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("SeeingTag - Original View", display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return
    
    print(f"[Homography] 初始化完成！开始追踪车标签...")
    tracker.homography_initialized = True

    fps_time, fps_count, cur_fps = time.time(), 0, 0.0
    frame_no = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            cv2.waitKey(10)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 根据模式决定检测策略
        if h_mode == "every_frame":
            # 模式1: 每帧都检测所有标签并更新 H 矩阵
            detections = tracker.detect(gray)
            car_x, car_z = tracker.locate(detections)
        else:
            # 模式2/3: 检查是否需要更新 H 矩阵
            need_update_H = False
            
            if h_mode == "once":
                # 模式2: 只在启动时更新一次，之后不再更新
                need_update_H = False
            elif h_mode == "interval":
                # 模式3: 每 N 帧更新一次
                need_update_H = (frame_no % h_interval == 0)
            
            if need_update_H:
                # 需要更新 H 矩阵：检测所有标签
                detections = tracker.detect(gray)
                tracker.compute_homography(detections)
                # 从检测结果中找车标签
                car_det = next((d for d in detections if d["id"] == tracker.car_tag_id), None)
            else:
                # 不需要更新 H 矩阵：只检测车标签
                car_det = tracker.detect_car_only(gray)
                detections = [car_det] if car_det else []
            
            # 使用缓存的 H 矩阵定位
            car_x, car_z = tracker.locate_with_cached_H(car_det)

        # 发送位置
        if car_x is not None and car_z is not None:
            tracker.send_position(car_x, car_z)
            if frame_no % 30 == 0:
                x_trunc = truncate_to_2_decimals(car_x)
                z_trunc = truncate_to_2_decimals(car_z)
                print(f"[Send] Car=(X={x_trunc:.2f}, Z={z_trunc:.2f})")

        fps_count += 1
        if time.time() - fps_time >= 1.0:
            cur_fps = fps_count
            fps_count = 0
            fps_time = time.time()

        display = tracker.draw_hud(frame, detections, car_x, car_z, cur_fps)
        
        # 添加模式显示
        mode_text = f"H-Mode: {h_mode}"
        if h_mode == "interval":
            mode_text += f" (N={h_interval})"
        cv2.putText(display, mode_text, (10, display.shape[0] - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 生成鸟瞰图
        bird_eye = tracker.generate_bird_eye_view(frame, detections, car_x, car_z)

        # 同时显示两个窗口
        cv2.imshow("SeeingTag - Original View", display)
        if bird_eye is not None:
            cv2.imshow("SeeingTag - Bird Eye View", bird_eye)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # 按 R 键重新校准 H 矩阵
            print("[Recalibrate] 重新检测固定标签...")
            tracker.homography_valid = False
            while not tracker.homography_valid:
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detections = tracker.detect(gray)
                tracker.compute_homography(detections)
                fixed_count = sum(1 for d in detections if d["id"] in tracker.tag_positions)
                display = frame.copy()
                cv2.putText(display, f"Recalibrating... ({fixed_count}/{len(tracker.tag_positions)})", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("SeeingTag - Original View", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    tracker.close()
                    cv2.destroyAllWindows()
                    return
            print("[Recalibrate] 完成！")

        frame_no += 1

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()


def main():
    mode = "calibration" if len(sys.argv) > 1 and sys.argv[1] in ("--calibrate", "-c") else "competition"
    cfg = load_config("tag_config.json")

    if mode == "calibration":
        run_calibration(cfg)
    else:
        run_competition(cfg)


if __name__ == "__main__":
    main()