import numpy as np
import yaml
import os
import joblib
from scipy.spatial.transform import Rotation as R

class Navigation:
    def __init__(self, project_root):
        self.target_pos = None
        self.target_yaw = None
        self.project_root = project_root
        self._load_target()
        self.counter = 0

    def _load_target(self):
        try:
            # 假设 MotionTracking 在 policy/MotionTracking
            mt_config_path = os.path.join(self.project_root, "policy", "MotionTracking", "config", "MotionTracking.yaml")
            
            if os.path.exists(mt_config_path):
                with open(mt_config_path, "r") as f:
                    mt_config = yaml.load(f, Loader=yaml.FullLoader)
                    motion_file = mt_config.get("motion_file")
                    if motion_file:
                        motion_path = os.path.join(self.project_root, "policy", "MotionTracking", motion_file)
                        motion_idx = mt_config.get("motion_index", 0)
                        if os.path.exists(motion_path):
                            data = joblib.load(motion_path)
                            # 假设是单个动作或字典中的第一个
                            if isinstance(data, dict):
                                # 通常第一个key是动作名
                                motion_id = list(data.keys())[motion_idx]
                                motion = data[motion_id]
                            else:
                                motion = data
                            
                            if "body_pos_w" in motion and "body_quat_w" in motion:
                                self.target_pos = motion["body_pos_w"][0, 0] # frame 0, body 0 (root)
                                target_quat = motion["body_quat_w"][0, 0] # wxyz
                                # 转换为 yaw
                                r = R.from_quat([target_quat[1], target_quat[2], target_quat[3], target_quat[0]])
                                self.target_yaw = r.as_euler('xyz')[2]
                                print(f"[Navigation] Loaded target pos {self.target_pos}, yaw {self.target_yaw:.3f}")
                            else:
                                print("[Navigation] Motion data missing body_pos_w or body_quat_w")
                        else:
                            print(f"[Navigation] Motion file not found: {motion_path}")
            else:
                print(f"[Navigation] Config not found: {mt_config_path}")
        except Exception as e:
            print(f"[Navigation] Failed to load target: {e}")

    def get_action(self, current_pos, current_quat):
        """
        计算导航速度指令
        current_pos: [x, y, z]
        current_quat: [w, x, y, z]
        return: [vx, vy, wz]
        """
        self.counter += 1
        if self.target_pos is None:
            return np.zeros(3, dtype=np.float32)
            
        # 计算当前 yaw
        try:
            r = R.from_quat([current_quat[1], current_quat[2], current_quat[3], current_quat[0]])
            current_yaw = r.as_euler('xyz')[2]
        except:
            return np.zeros(3, dtype=np.float32)
        
        # 位置误差 (世界坐标系)
        err_pos = self.target_pos - current_pos
        dist = np.linalg.norm(err_pos[:2])
        
        # 策略：
        # 1. 如果距离较远 (>0.5m)，目标朝向是"指向目标点"
        # 2. 如果距离较近 (<=0.5m)，目标朝向是"最终目标朝向"
        
        if dist > 0.5:
            desired_yaw = np.arctan2(err_pos[1], err_pos[0])
        else:
            desired_yaw = self.target_yaw if self.target_yaw is not None else current_yaw

        err_yaw = desired_yaw - current_yaw
        # 归一化到 [-pi, pi]
        err_yaw = (err_yaw + np.pi) % (2 * np.pi) - np.pi
        
        # 转换位置误差到 body frame
        cos_yaw = np.cos(current_yaw)
        sin_yaw = np.sin(current_yaw)
        
        err_x_body = err_pos[0] * cos_yaw + err_pos[1] * sin_yaw
        err_y_body = -err_pos[0] * sin_yaw + err_pos[1] * cos_yaw

        kp_pos = 1.0
        kp_ang = 1.5 
        
        # 计算指令
        # 如果角度误差很大，减小前进速度，优先转向
        turn_factor = np.clip(1.0 - abs(err_yaw) / 1.0, 0.0, 1.0) # 误差大于1弧度时停止前进
        
        vx = np.clip(kp_pos * err_x_body, -0.5, 0.5) * turn_factor
        vy = np.clip(kp_pos * err_y_body, -0.3, 0.3) * turn_factor 
        wz = np.clip(kp_ang * err_yaw, -0.8, 0.8)
        
        # 停止条件：位置和角度都满足
        if dist < 0.1 and abs(err_yaw) < 0.1:
             return np.zeros(3, dtype=np.float32)

        if self.counter % 400 == 0:
            print(f"[Nav] Step {self.counter}: Cur Pos {current_pos}, Target {self.target_pos}, Dist {dist:.3f}, Err Yaw {err_yaw:.3f}")
        return np.array([vx, vy, wz], dtype=np.float32)
