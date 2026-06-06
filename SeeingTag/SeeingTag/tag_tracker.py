"""
标签追踪核心模块
负责ARUCO标签检测、位姿估计、坐标转换、可视化等功能
"""

import cv2
import cv2.aruco as aruco
import numpy as np
from typing import Dict, List, Optional, Tuple
from config import truncate_to_2_decimals
from udp_sender import UdpSender


class TagTracker:
    """ARUCO标签追踪器"""
    
    def __init__(self, cfg: dict):
        """
        初始化标签追踪器
        参数:
            cfg: 配置字典，包含标签尺寸、位置、UDP参数等
        """
        # 基本配置
        self.tag_size = cfg["tag_size_m"]
        self.min_tag_width = cfg["min_tag_width"]
        self.tag_positions = cfg["tag_positions"]
        self.car_tag_id = cfg["car_tag_id"]
        
        # H 矩阵更新模式配置
        self.homography_update_mode = cfg.get("homography_update_mode", "interval")
        self.homography_update_interval = cfg.get("homography_update_interval", 30)

        # 初始化 ArUco 检测器
        self._init_detector()

        # UDP 发送器
        self.udp: Optional[UdpSender] = UdpSender(cfg["unity_ip"], cfg["unity_port"])

        # 单应矩阵缓存
        self.H: Optional[np.ndarray] = None
        self.homography_valid = False
        # 标记 H 矩阵是否已初始化（用于 "once" 模式）
        self.homography_initialized = False

    def _init_detector(self):
        """初始化ARUCO检测器，设置检测参数"""
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters()
        
        # 自适应阈值参数
        self.parameters.adaptiveThreshWinSizeMin = 3
        self.parameters.adaptiveThreshWinSizeMax = 23
        self.parameters.adaptiveThreshWinSizeStep = 5
        self.parameters.adaptiveThreshConstant = 7
        
        # 透视变换参数
        self.parameters.perspectiveRemovePixelPerCell = 8
        
        # 角点精炼参数
        self.parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self.parameters.cornerRefinementWinSize = 5
        
        # 创建检测器
        self.detector = aruco.ArucoDetector(self.dictionary, self.parameters)

    @staticmethod
    def build_tag_corners_world(tag_center: np.ndarray, tag_size: float) -> np.ndarray:
        """
        返回标签4个角点在世界 XZ 平面上的坐标 (x, z)
        参数:
            tag_center: 标签中心世界坐标 [x, y, z]
            tag_size: 标签物理尺寸（米）
        返回:
            4个角点的世界坐标 (x, z)，shape=(4, 2)
        """
        s2 = tag_size / 2
        # 标签局部坐标系中的角点位置（以标签中心为原点）
        local = np.array([
            [-s2, -s2, 0],  # 左上
            [ s2, -s2, 0],  # 右上
            [ s2,  s2, 0],  # 右下
            [-s2,  s2, 0],  # 左下
        ])
        # 转换到世界坐标系
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
            w = abs(c[0][0] - c[1][0])  # 计算标签宽度（像素）
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
        参数:
            detections: 检测结果列表
        返回:
            计算成功返回True，失败返回False
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

    def compute_yaw_from_corners(self, corners: np.ndarray) -> Optional[float]:
        """
        通过标签角点计算偏航角
        原理：角点0→角点1的连线方向代表标签的"前方"，映射到世界坐标系后计算角度
        参数:
            corners: 标签4个角点的像素坐标，shape=(4, 2)
        返回:
            yaw角（度），失败返回None
        """
        if not self.homography_valid or self.H is None:
            return None
        
        # 获取角点0和角点1的像素坐标（角点顺序：左上、右上、右下、左下）
        # 角点0→角点1的方向代表标签的"前方"（X轴正方向）
        p0 = corners[0]  # 左上角
        p1 = corners[1]  # 右上角
        
        # 计算标签中心点
        center = np.mean(corners, axis=0)
        
        # 计算方向向量（从中心指向标签前方）
        # 使用角点0和角点1的中点作为参考
        front_mid = (p0 + p1) / 2.0
        direction_pixel = front_mid - center
        
        # 如果方向向量太小，使用角点0→角点1的连线方向
        if np.linalg.norm(direction_pixel) < 5:
            direction_pixel = p1 - p0
        
        # 归一化方向向量
        norm = np.linalg.norm(direction_pixel)
        if norm < 1e-6:
            return None
        direction_pixel = direction_pixel / norm
        
        # 通过单应矩阵映射方向向量到世界坐标系
        # 映射中心点和方向终点，然后计算世界坐标系中的方向
        center_h = np.array([[center[0], center[1]]], dtype=np.float64)
        end_point = center + direction_pixel * 100  # 延长方向向量
        end_h = np.array([[end_point[0], end_point[1]]], dtype=np.float64)
        
        # 映射到世界坐标
        center_world = cv2.perspectiveTransform(center_h.reshape(1, 1, 2), self.H)[0, 0]
        end_world = cv2.perspectiveTransform(end_h.reshape(1, 1, 2), self.H)[0, 0]
        
        # 计算世界坐标系中的方向向量
        direction_world = end_world - center_world
        direction_world = direction_world / (np.linalg.norm(direction_world) + 1e-9)
        
        # 计算yaw角（绕Y轴旋转角度）
        # 世界坐标系：X轴正方向为0度，逆时针为正
        # direction_world = [dx, dz]
        yaw_rad = np.arctan2(direction_world[1], direction_world[0])  # atan2(dz, dx)
        yaw_deg = np.degrees(yaw_rad)
        
        return float(yaw_deg)

    def locate(self, detections: List[dict]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        定位车标签，返回世界坐标 (x, z) 和偏航角 yaw
        此方法会重新计算 H 矩阵
        参数:
            detections: 所有检测到的标签列表
        返回:
            (x, z, yaw) 世界坐标和偏航角（度），失败返回 (None, None, None)
        """
        # 第 1 步：更新单应矩阵
        self.compute_homography(detections)

        if not self.homography_valid or self.H is None:
            return None, None, None

        # 第 2 步：找到车标签的中心像素坐标
        car_det = next((d for d in detections if d["id"] == self.car_tag_id), None)
        if car_det is None:
            return None, None, None

        car_center_pixel = np.mean(car_det["corners"], axis=0)

        # 第 3 步：通过单应矩阵映射到世界坐标
        pixel_pt = np.array([[car_center_pixel[0], car_center_pixel[1]]], dtype=np.float64)
        world_pt = cv2.perspectiveTransform(pixel_pt.reshape(1, 1, 2), self.H)
        wx, wz = world_pt[0, 0, 0], world_pt[0, 0, 1]

        # 第 4 步：计算偏航角
        yaw = self.compute_yaw_from_corners(car_det["corners"])

        return float(wx), float(wz), yaw

    def locate_with_cached_H(self, car_det: Optional[dict]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        使用缓存的 H 矩阵定位车标签（不重新计算 H）
        参数:
            car_det: 车标签检测结果（来自 detect_car_only）
        返回:
            (x, z, yaw) 世界坐标和偏航角（度），失败返回 (None, None, None)
        """
        # 检查 H 矩阵是否有效
        if not self.homography_valid or self.H is None:
            return None, None, None
        
        # 检查车标签是否检测到
        if car_det is None:
            return None, None, None

        # 计算车标签中心像素坐标
        car_center_pixel = np.mean(car_det["corners"], axis=0)

        # 通过缓存的 H 矩阵映射到世界坐标
        pixel_pt = np.array([[car_center_pixel[0], car_center_pixel[1]]], dtype=np.float64)
        world_pt = cv2.perspectiveTransform(pixel_pt.reshape(1, 1, 2), self.H)
        wx, wz = world_pt[0, 0, 0], world_pt[0, 0, 1]

        # 计算偏航角
        yaw = self.compute_yaw_from_corners(car_det["corners"])

        return float(wx), float(wz), yaw

    def send_position(self, x: float, z: float, yaw: float = 0.0):
        """
        发送位置和偏航角数据
        参数:
            x: 世界坐标X（米）
            z: 世界坐标Z（米）
            yaw: 偏航角（度）
        """
        if self.udp is not None:
            self.udp.send(x, z, yaw)

    def draw_hud(self, frame: np.ndarray, detections: List[dict],
                 car_x: Optional[float], car_z: Optional[float], 
                 car_yaw: Optional[float], fps: float) -> np.ndarray:
        """
        在画面上绘制HUD信息
        参数:
            frame: 原始图像
            detections: 检测结果列表
            car_x: 车辆世界坐标X
            car_z: 车辆世界坐标Z
            car_yaw: 车辆偏航角（度）
            fps: 当前帧率
        返回:
            绘制了HUD的图像
        """
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
            # 显示偏航角
            if car_yaw is not None:
                car_yaw_display = truncate_to_2_decimals(car_yaw)
                lines.append(f"Yaw: {car_yaw_display:.1f} deg")
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
                               car_x: Optional[float], car_z: Optional[float],
                               car_yaw: Optional[float] = None) -> Optional[np.ndarray]:
        """
        生成鸟瞰图（俯视图）：对原图进行逆透视变换
        将相机视角的图像映射到世界平面视角
        参数:
            frame: 原始图像
            detections: 检测结果列表
            car_x: 车辆世界坐标X
            car_z: 车辆世界坐标Z
            car_yaw: 车辆偏航角（度）
        返回:
            鸟瞰图图像，失败返回None
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

        # 绘制车标签位置（黄色圆圈 + 朝向箭头）
        if car_x is not None and car_z is not None:
            bx = int((car_x - world_min_x) * scale_x)
            bz = int((car_z - world_min_z) * scale_z)
            # 确保坐标在图像范围内
            if 0 <= bx < output_width and 0 <= bz < output_height:
                cv2.circle(bird_eye, (bx, bz), 12, (0, 255, 255), -1)
                cv2.circle(bird_eye, (bx, bz), 14, (0, 200, 200), 2)
                cv2.putText(bird_eye, "CAR", (bx + 15, bz + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # 绘制朝向箭头（如果有yaw角）
                if car_yaw is not None:
                    arrow_length = 30  # 箭头长度（像素）
                    yaw_rad = np.radians(car_yaw)
                    # 计算箭头终点
                    end_x = int(bx + arrow_length * np.cos(yaw_rad))
                    end_z = int(bz + arrow_length * np.sin(yaw_rad))
                    cv2.arrowedLine(bird_eye, (bx, bz), (end_x, end_z), (0, 255, 255), 3, tipLength=0.4)

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
        """
        打印调试信息
        参数:
            detections: 检测结果列表
        """
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
        """关闭资源"""
        if self.udp is not None:
            self.udp.close()
