"""
UDP通信模块
负责将定位数据通过UDP协议发送到Unity端
"""

import socket
import json
import time
from config import truncate_to_2_decimals


class UdpSender:
    """UDP数据发送器"""
    
    def __init__(self, target_ip: str, target_port: int):
        """
        初始化UDP发送器
        参数:
            target_ip: 目标IP地址
            target_port: 目标端口号
        """
        self.target = (target_ip, target_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0  # 序列号，用于追踪数据包

    def send(self, x: float, z: float, yaw: float = 0.0) -> bool:
        """
        通过UDP发送位置和偏航角数据
        参数:
            x: 世界坐标X（米）
            z: 世界坐标Z（米）
            yaw: 偏航角（度），范围 -180 ~ 180
        返回:
            发送成功返回True，失败返回False
        """
        try:
            # 截断到两位小数
            x_truncated = truncate_to_2_decimals(x)
            z_truncated = truncate_to_2_decimals(z)
            yaw_truncated = truncate_to_2_decimals(yaw)
            
            # 构建JSON数据包
            data = {
                "type": "robot_position",
                "pos": [float(z_truncated), 0.0, float(x_truncated)],  # Unity坐标系：交换x和z的顺序
                "euler": [0.0, 0.0, float(yaw_truncated)],  # yaw角（绕Y轴旋转）
                "seq": self.seq,
                "timestamp": time.time()
            }
            
            # 发送数据
            self.seq += 1
            self.sock.sendto(json.dumps(data).encode('utf-8'), self.target)
            print(f"[UDP] 发送成功 → {self.target[0]}:{self.target[1]}  pos=({z_truncated:.2f}, 0.0, {x_truncated:.2f})  yaw={yaw_truncated:.1f}°  seq={self.seq-1}")
            return True
            
        except Exception as e:
            print(f"[UDP] 发送失败: {e}")
            return False

    def close(self):
        """关闭UDP套接字"""
        self.sock.close()
