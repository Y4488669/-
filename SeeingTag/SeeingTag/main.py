"""
SeeingTag — ARUCO 智能车定位系统
主程序入口

模块结构：
    - config.py: 配置加载
    - udp_sender.py: UDP通信
    - tag_tracker.py: 标签追踪核心
    - camera.py: 摄像头管理
    - main.py: 主程序入口
"""

import cv2
import time
import argparse
from config import load_config, truncate_to_2_decimals
from tag_tracker import TagTracker
from camera import open_camera


def run_calibration(cfg: dict):
    """
    校准模式主循环
    不发送UDP数据，用于现场调试和参数调整
    参数:
        cfg: 配置字典
    """
    print("=== Calibration Mode — 仅显示，不发送UDP ===")
    
    tracker = TagTracker(cfg)
    cap = open_camera()
    
    print(f"[Homography] 固定标签: {list(cfg['tag_positions'].keys())}")
    print(f"[Homography] 车标签ID: {cfg['car_tag_id']}")
    print("[提示] 按 C 键打印调试信息，按 Q 键退出")

    fps_count = 0
    fps_time = time.time()
    cur_fps = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            cv2.waitKey(10)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = tracker.detect(gray)
        car_x, car_z, car_yaw = tracker.locate(detections)

        fps_count += 1
        if time.time() - fps_time >= 1.0:
            cur_fps = fps_count
            fps_count = 0
            fps_time = time.time()

        display = tracker.draw_hud(frame, detections, car_x, car_z, car_yaw, cur_fps)
        bird_eye = tracker.generate_bird_eye_view(frame, detections, car_x, car_z, car_yaw)

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
    参数:
        cfg: 配置字典
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
    print("[提示] 按 R 键重新校准H矩阵，按 Q 键退出")

    # 等待H矩阵初始化
    print("\n[Init] 等待检测固定标签以初始化H矩阵...")
    while not tracker.homography_valid:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = tracker.detect(gray)
        tracker.compute_homography(detections)
        
        fixed_count = sum(1 for d in detections if d["id"] in tracker.tag_positions)
        display = frame.copy()
        cv2.putText(display, f"Initializing... ({fixed_count}/{len(tracker.tag_positions)})", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("SeeingTag - Original View", display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            tracker.close()
            cv2.destroyAllWindows()
            return
    
    print("[Init] H矩阵初始化完成！开始追踪...\n")

    fps_count = 0
    fps_time = time.time()
    cur_fps = 0
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
            car_x, car_z, car_yaw = tracker.locate(detections)
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
            car_x, car_z, car_yaw = tracker.locate_with_cached_H(car_det)

        # 发送位置
        if car_x is not None and car_z is not None:
            yaw_to_send = car_yaw if car_yaw is not None else 0.0
            tracker.send_position(car_x, car_z, yaw_to_send)
            if frame_no % 30 == 0:
                x_trunc = truncate_to_2_decimals(car_x)
                z_trunc = truncate_to_2_decimals(car_z)
                yaw_trunc = truncate_to_2_decimals(yaw_to_send)
                print(f"[Send] Car=(X={x_trunc:.2f}, Z={z_trunc:.2f}, Yaw={yaw_trunc:.1f}°)")

        fps_count += 1
        if time.time() - fps_time >= 1.0:
            cur_fps = fps_count
            fps_count = 0
            fps_time = time.time()

        display = tracker.draw_hud(frame, detections, car_x, car_z, car_yaw, cur_fps)
        
        # 添加模式显示
        mode_text = f"H-Mode: {h_mode}"
        if h_mode == "interval":
            mode_text += f" (N={h_interval})"
        cv2.putText(display, mode_text, (10, display.shape[0] - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        bird_eye = tracker.generate_bird_eye_view(frame, detections, car_x, car_z, car_yaw)

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
    """主函数"""
    parser = argparse.ArgumentParser(description="SeeingTag — ARUCO智能车定位系统")
    parser.add_argument("--calibrate", action="store_true", help="校准模式（不发送UDP）")
    parser.add_argument("--config", type=str, default="tag_config.json", help="配置文件路径")
    args = parser.parse_args()

    # 加载配置
    cfg = load_config(args.config)

    # 运行对应模式
    if args.calibrate:
        run_calibration(cfg)
    else:
        run_competition(cfg)


if __name__ == "__main__":
    main()
