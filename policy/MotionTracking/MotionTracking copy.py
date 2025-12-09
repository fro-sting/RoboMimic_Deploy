from common.path_config import PROJECT_ROOT

from FSM.FSMState import FSMStateName, FSMState
from common.ctrlcomp import StateAndCmd, PolicyOutput
import numpy as np
import yaml
from common.utils import FSMCommand, progress_bar
import onnx
import onnxruntime
import os
import joblib
from scipy.spatial.transform import Rotation as R

"""
G1 29自由度关节顺序 (0-28):
0:  left_hip_pitch_joint
1:  left_hip_roll_joint
2:  left_hip_yaw_joint
3:  left_knee_joint
4:  left_ankle_pitch_joint
5:  left_ankle_roll_joint
6:  right_hip_pitch_joint
7:  right_hip_roll_joint
8:  right_hip_yaw_joint
9:  right_knee_joint
10: right_ankle_pitch_joint
11: right_ankle_roll_joint
12: waist_yaw_joint
13: waist_roll_joint
14: waist_pitch_joint
15: left_shoulder_pitch_joint
16: left_shoulder_roll_joint
17: left_shoulder_yaw_joint
18: left_elbow_joint
19: left_wrist_roll_joint
20: left_wrist_pitch_joint
21: left_wrist_yaw_joint
22: right_shoulder_pitch_joint
23: right_shoulder_roll_joint
24: right_shoulder_yaw_joint
25: right_elbow_joint
26: right_wrist_roll_joint
27: right_wrist_pitch_joint
28: right_wrist_yaw_joint

模型19维输出对应:
.*hip.*: 6个 (0,1,2,6,7,8)
.*knee.*: 2个 (3,9)
.*ankle_pitch.*: 2个 (4,10)
waist_yaw_joint: 1个 (12)
.*shoulder.*: 6个 (15,16,17,22,23,24)
.*elbow.*: 2个 (18,25)
总计: 19个
 
keypoint_body (12个，用于body_pos观测):
- left_hip_pitch_link (1), right_hip_pitch_link (7)
- left_knee_link (4), right_knee_link (10)
- left_ankle_roll_link (6), right_ankle_roll_link (12)
- left_shoulder_roll_link (16), right_shoulder_roll_link (23)
- left_elbow_link (18), right_elbow_link (25)
- left_wrist_yaw_link (21), right_wrist_yaw_link (28)
"""


class MotionLibrary:
    """动作库 - 加载和管理参考动作数据（支持多个动作）"""
    
    def __init__(self, motion_file: str = None, motion_index: int = 0):
        
        self.motion_file = motion_file
        self.all_motions = None 
        self.motion_names = []   
        self.current_motion_index = 0
        self.current_motion_name = ""

        self.joint_pos = None
        self.joint_vel = None
        self.body_pos_w = None
        self.body_quat_w = None
        self.body_lin_vel_w = None
        self.body_ang_vel_w = None
        self.num_frames = 0
        self.dt = 0.02  # 50Hz
        self.motion_length = 0.0
        
        if motion_file and os.path.exists(motion_file):
            self.load_motion_file(motion_file)
            self.select_motion(motion_index)
    
    def load_motion_file(self, motion_file: str):
        """加载动作文件（可能包含多个动作）"""
        print(f"Loading motion file: {motion_file}...")
        data = joblib.load(motion_file)
        
        if isinstance(data, dict):
            self.all_motions = data
            self.motion_names = list(data.keys())
            print(f"Found {len(self.motion_names)} motions:")
            for i, name in enumerate(self.motion_names):
                print(f"  [{i}] {name}")
        else:
            # 单个动作，包装成字典
            self.all_motions = {"motion_0": data}
            self.motion_names = ["motion_0"]
            print("Found 1 motion (unnamed)")
    
    def select_motion(self, index: int = 0):
        """选择并加载指定索引的动作"""
        if self.all_motions is None or len(self.motion_names) == 0:
            print("No motions loaded!")
            return False

        index = max(0, min(index, len(self.motion_names) - 1))
        self.current_motion_index = index
        self.current_motion_name = self.motion_names[index]
        
        motion = self.all_motions[self.current_motion_name]
        
        self.joint_pos = np.array(motion["joint_pos"], dtype=np.float32)
        self.joint_vel = np.array(motion["joint_vel"], dtype=np.float32)
        self.body_pos_w = np.array(motion["body_pos_w"], dtype=np.float32)
        self.body_quat_w = np.array(motion["body_quat_w"], dtype=np.float32)
        self.body_lin_vel_w = np.array(motion["body_lin_vel_w"], dtype=np.float32)
        self.body_ang_vel_w = np.array(motion["body_ang_vel_w"], dtype=np.float32)
        
        self.num_frames = self.joint_pos.shape[0]
        self.motion_length = self.num_frames * self.dt

        self.root_pos_w = self.body_pos_w[:, 0]
        self.root_quat_w = self.body_quat_w[:, 0]
        self.root_lin_vel_w = self.body_lin_vel_w[:, 0]
        self.root_ang_vel_w = self.body_ang_vel_w[:, 0]
        
        print(f"\n>>> Selected motion [{index}]: {self.current_motion_name}")
        print(f"    Frames: {self.num_frames}, Duration: {self.motion_length:.2f}s")
        print(f"    joint_pos shape: {self.joint_pos.shape}")
        print(f"    body_pos_w shape: {self.body_pos_w.shape}")
        return True
    
    def next_motion(self):
        """切换到下一个动作"""
        next_idx = (self.current_motion_index + 1) % len(self.motion_names)
        return self.select_motion(next_idx)
    
    def prev_motion(self):
        """切换到上一个动作"""
        prev_idx = (self.current_motion_index - 1) % len(self.motion_names)
        return self.select_motion(prev_idx)
    
    def get_motion_count(self):
        return len(self.motion_names)
    
    def get_frame(self, frame_idx: int):
        """获取指定帧的数据"""
        if self.num_frames == 0:
            return None
        frame_idx = min(frame_idx, self.num_frames - 1)
        return {
            "joint_pos": self.joint_pos[frame_idx],
            "joint_vel": self.joint_vel[frame_idx],
            "body_pos_w": self.body_pos_w[frame_idx],
            "body_quat_w": self.body_quat_w[frame_idx],
            "body_lin_vel_w": self.body_lin_vel_w[frame_idx],
            "body_ang_vel_w": self.body_ang_vel_w[frame_idx],
            "root_pos_w": self.root_pos_w[frame_idx],
            "root_quat_w": self.root_quat_w[frame_idx],
        }

class MotionTracking(FSMState):
    """
    MotionTracking 策略 
    
    13个 keypoint body (用于body_pos观测，与训练一致):
    动作数据中的body索引（不含world，pelvis=0）:
    - pelvis (0)
    - left_hip_pitch_link (1), right_hip_pitch_link (7)
    - left_knee_link (4), right_knee_link (10)
    - left_ankle_roll_link (6), right_ankle_roll_link (12)
    - left_shoulder_roll_link (17), right_shoulder_roll_link (24)
    - left_elbow_link (19), right_elbow_link (26)
    - left_wrist_yaw_link (22), right_wrist_yaw_link (29)
    """
    
    # 13个关键点body的名称
    KEYPOINT_BODY_NAMES = [
        "pelvis",                # 0
        "left_hip_pitch_link",   # 1
        "right_hip_pitch_link",  # 2
        "left_knee_link",        # 3
        "right_knee_link",       # 4
        "left_ankle_roll_link",  # 5
        "right_ankle_roll_link", # 6
        "left_shoulder_roll_link", # 7
        "right_shoulder_roll_link", # 8
        "left_elbow_link",       # 9
        "right_elbow_link",      # 10
        "left_wrist_yaw_link",   # 11
        "right_wrist_yaw_link",  # 12
    ]
    
    def __init__(self, state_cmd:StateAndCmd, policy_output:PolicyOutput):
        super().__init__()
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.name = FSMStateName.SKILL_MotionTracking
        self.name_str = "skill_motiontracking"
        self.counter_step = 0
        self.mujoco_data = None  
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "MotionTracking.yaml")
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.onnx_path = os.path.join(current_dir, "model", config["onnx_path"])
            self.kps = np.array(config["kps"], dtype=np.float32)
            self.kds = np.array(config["kds"], dtype=np.float32)
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)
            self.tau_limit = np.array(config["tau_limit"], dtype=np.float32)
            self.num_actions = config["num_actions"]  # 19
            self.num_obs_robot = config["num_obs_robot"]  # 123
            self.num_obs_ref_motion = config["num_obs_ref_motion"]  # 120
            self.num_obs_priv = config["num_obs_priv"]  # 40
            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = config["action_scale"]
            self.motion_length = config.get("motion_length", 10.0) 
            
            # 加载动作库
            motion_file = config.get("motion_file", None)
            motion_index = config.get("motion_index", 0) 
            if motion_file:
                motion_path = os.path.join(current_dir, motion_file)
                self.motion_lib = MotionLibrary(motion_path, motion_index)
                # 使用动作库的实际长度
                if self.motion_lib.num_frames > 0:
                    self.motion_length = self.motion_lib.motion_length
            else:
                self.motion_lib = MotionLibrary()
            
            # 19个输出关节对应29自由度的索引
            # hip(6) + knee(2) + ankle_pitch(2) + waist_yaw(1) + shoulder(6) + elbow(2) = 19
            self.action_to_dof29_index = np.array([
                0, 1, 2,    # left_hip_pitch, left_hip_roll, left_hip_yaw
                6, 7, 8,    # right_hip_pitch, right_hip_roll, right_hip_yaw
                3, 9,       # left_knee, right_knee
                4, 10,      # left_ankle_pitch, right_ankle_pitch
                12,         # waist_yaw
                15, 16, 17, # left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw
                22, 23, 24, # right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw
                18, 25      # left_elbow, right_elbow
            ], dtype=np.int32)
            
            # 从29自由度中提取19个控制关节的索引 (用于观测)
            self.dof19_index = self.action_to_dof29_index.copy()
            
            # 13个keypoint body在参考动作数据中的索引
            # 动作数据body索引（不含world，pelvis=0）
            self.keypoint_body_indices = np.array([
                0,   # pelvis
                1,   # left_hip_pitch_link
                7,   # right_hip_pitch_link
                4,   # left_knee_link
                10,  # right_knee_link
                6,   # left_ankle_roll_link
                12,  # right_ankle_roll_link
                17,  # left_shoulder_roll_link
                24,  # right_shoulder_roll_link
                19,  # left_elbow_link
                26,  # right_elbow_link
                22,  # left_wrist_yaw_link
                29,  # right_wrist_yaw_link
            ], dtype=np.int32)
            
            self.num_keypoints = len(self.keypoint_body_indices)  # 13
            
            # 初始化观测和动作缓存
            self.obs_robot = np.zeros(self.num_obs_robot, dtype=np.float32)
            self.obs_ref_motion = np.zeros(self.num_obs_ref_motion, dtype=np.float32)
            self.obs_priv = np.zeros(self.num_obs_priv, dtype=np.float32)
            self.action = np.zeros(self.num_actions, dtype=np.float32)
            self.prev_action = np.zeros(self.num_actions, dtype=np.float32)
            
            # 上一帧的body位置，用于计算速度
            self.prev_body_pos = np.zeros((self.num_keypoints, 3), dtype=np.float32)
            
            # 加载ONNX模型
            self.onnx_model = onnx.load(self.onnx_path)
            self.ort_session = onnxruntime.InferenceSession(self.onnx_path)
            
                    
            print("MotionTracking policy initializing ...")
    
    def enter(self):
        """进入状态时的初始化"""
        self.counter_step = 0
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.prev_action = np.zeros(self.num_actions, dtype=np.float32)
        self.obs_robot = np.zeros(self.num_obs_robot, dtype=np.float32)
        self.obs_ref_motion = np.zeros(self.num_obs_ref_motion, dtype=np.float32)
        self.obs_priv = np.zeros(self.num_obs_priv, dtype=np.float32)
        self.prev_body_pos = np.zeros((self.num_keypoints, 3), dtype=np.float32)
    
    def set_mujoco_data(self, mj_data, mj_model):
        """设置MuJoCo数据引用，用于获取body位置"""
        self.mujoco_data = mj_data
        self.mujoco_model = mj_model
        # 获取keypoint body的ID（MuJoCo中的ID）
        self.keypoint_body_ids = []
        for name in self.KEYPOINT_BODY_NAMES:
            try:
                body_id = mj_model.body(name).id
                self.keypoint_body_ids.append(body_id)
            except KeyError:
                print(f"Warning: Body '{name}' not found in MuJoCo model")
                self.keypoint_body_ids.append(-1)
        self.keypoint_body_ids = np.array(self.keypoint_body_ids, dtype=np.int32)
        print(f"Keypoint body IDs (MuJoCo): {self.keypoint_body_ids}")
    
        
    def _get_body_positions(self):
        """从MuJoCo获取13个keypoint body的位置（在body frame下，相对于pelvis）"""
        if self.mujoco_data is None:
            return np.zeros((self.num_keypoints, 3), dtype=np.float32)
        
        # pelvis位置和四元数 (root)
        pelvis_pos = self.mujoco_data.body("pelvis").xpos.copy()
        pelvis_quat = self.mujoco_data.body("pelvis").xquat.copy()  # wxyz
        
        body_pos = np.zeros((self.num_keypoints, 3), dtype=np.float32)
        for i, body_id in enumerate(self.keypoint_body_ids):
            if body_id >= 0:
                world_pos = self.mujoco_data.xpos[body_id]
                # 1. 平移到pelvis原点
                pos_rel = world_pos - pelvis_pos
                # 2. 旋转到body frame（用pelvis四元数的逆）
                body_pos[i] = self._quat_rotate_inverse(pelvis_quat, pos_rel)
        return body_pos
    
    def _compute_ref_motion_obs(self, frame_idx: int):
        """
        计算参考动作观测 (120维)
        
        结构:
        - ref_qpos: 29 (参考关节位置)
        - ref_kp_pos_gap: 13×3 = 39 (关键点位置差，使用subtract_frame_transforms)
        - ref_kp_quat: 13×4 = 52 (关键点四元数差，使用subtract_frame_transforms)
        总计: 29 + 39 + 52 = 120
        """
        obs = np.zeros(self.num_obs_ref_motion, dtype=np.float32)
        
        if self.motion_lib.num_frames == 0:
            return obs
        
        frame = self.motion_lib.get_frame(frame_idx)
        if frame is None:
            return obs
        
        idx = 0
        
        # 1. ref_qpos (29) - 参考关节位置（原始值）
        ref_joint_pos = frame["joint_pos"]
        obs[idx:idx+29] = ref_joint_pos
        idx += 29
        # 2. ref_kp_pos_gap (39) - 使用subtract_frame_transforms计算位置差
        # 获取当前body的位置和四元数（世界坐标系）
        cur_body_pos_w = np.zeros((self.num_keypoints, 3), dtype=np.float32)
        cur_body_quat_w = np.zeros((self.num_keypoints, 4), dtype=np.float32)
        
        if self.mujoco_data is not None:
            for i, body_id in enumerate(self.keypoint_body_ids):
                if body_id >= 0:
                    cur_body_pos_w[i] = self.mujoco_data.xpos[body_id].copy()
                    cur_body_quat_w[i] = self.mujoco_data.xquat[body_id].copy()  # wxyz

        ref_body_pos_all = frame["body_pos_w"]  # 30×3
        ref_body_quat_all = frame["body_quat_w"]  # 30×4
        
        # 提取13个keypoint的参考数据
        ref_body_pos_w = np.zeros((self.num_keypoints, 3), dtype=np.float32)
        ref_body_quat_w = np.zeros((self.num_keypoints, 4), dtype=np.float32)
        for i, body_idx in enumerate(self.keypoint_body_indices):
            if body_idx < ref_body_pos_all.shape[0]:
                ref_body_pos_w[i] = ref_body_pos_all[body_idx]
                ref_body_quat_w[i] = ref_body_quat_all[body_idx]
        
        # 使用subtract_frame_transforms计算位置差（在当前body frame下）
        pos_gap, quat_gap = self._subtract_frame_transforms(
            cur_body_pos_w, cur_body_quat_w,
            ref_body_pos_w, ref_body_quat_w
        )
        
        obs[idx:idx+39] = pos_gap.flatten()
        idx += 39
        
        # 3. ref_kp_quat (52) - 四元数差
        obs[idx:idx+52] = quat_gap.flatten()
        idx += 52
        
        return obs

    def _quat_rotate_inverse(self, quat, vec):
        """用四元数的逆旋转向量（将世界坐标转换到body frame）
        quat: (4,) wxyz格式
        """
        # 将wxyz转换为xyzw格式（scipy使用xyzw）
        quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float32)
        rot = R.from_quat(quat_xyzw)
        # 使用逆旋转
        rotated = rot.inv().apply(vec)
        return rotated
    
    def _subtract_frame_transforms(self, pos0, quat0, pos1, quat1):
        """
        计算两个frame之间的相对变换
        
        Args:
            pos0: 当前位置 (N, 3)
            quat0: 当前四元数 (N, 4) wxyz
            pos1: 参考位置 (N, 3)
            quat1: 参考四元数 (N, 4) wxyz
        
        Returns:
            pos_diff: 位置差（在frame0坐标系下） (N, 3)
            quat_diff: 四元数差（相对旋转） (N, 4) wxyz
        """
        # 位置差：将pos1转换到pos0的局部坐标系
        pos_diff = pos1 - pos0
        
        # 对每个关键点进行旋转变换
        pos_diff_local = np.zeros_like(pos_diff)
        quat_diff = np.zeros_like(quat0)
        
        for i in range(len(pos0)):
            # 将wxyz转换为xyzw格式（scipy使用xyzw）
            quat0_xyzw = np.array([quat0[i][1], quat0[i][2], quat0[i][3], quat0[i][0]])
            quat1_xyzw = np.array([quat1[i][1], quat1[i][2], quat1[i][3], quat1[i][0]])
            
            rot0 = R.from_quat(quat0_xyzw)
            rot1 = R.from_quat(quat1_xyzw)
            
            # 位置差旋转到frame0的坐标系
            pos_diff_local[i] = rot0.inv().apply(pos_diff[i])
            
            # 四元数差：quat_diff = quat0^-1 * quat1
            rot_diff = rot0.inv() * rot1
            quat_diff_xyzw = rot_diff.as_quat()
            # 转回wxyz格式
            quat_diff[i] = np.array([quat_diff_xyzw[3], quat_diff_xyzw[0], 
                                     quat_diff_xyzw[1], quat_diff_xyzw[2]], dtype=np.float32)
        
        return pos_diff_local, quat_diff
    
    def _compute_priv_obs(self, dt=0.02):
        """
        计算特权观测 (40维)
        
        结构:
        - root_height: 1
        - root_linvel_b: 3 
        - body_vel: 12×3 = 36 (不含pelvis，只有12个关键点的速度)
        总计: 1 + 3 + 36 = 40
        """
        obs = np.zeros(self.num_obs_priv, dtype=np.float32)
        
        if self.mujoco_data is None:
            return obs
        
        idx = 0
        
        # root_height (1) - pelvis高度
        pelvis_pos = self.mujoco_data.body("pelvis").xpos.copy()
        obs[idx] = pelvis_pos[2]  # z坐标
        idx += 1
        
        # root_linvel_b (3) - body frame下的root线速度
        try:
            root_vel = self.mujoco_data.body("pelvis").cvel[3:6].copy()  # 线速度部分
            obs[idx:idx+3] = root_vel
        except:
            obs[idx:idx+3] = 0.0
        idx += 3
        
        # body_vel (36) - 12个keypoint的速度（不含pelvis，通过差分估计）
        cur_body_pos = self._get_body_positions()  # (13, 3)
        cur_body_pos_12 = cur_body_pos[1:]  # (12, 3)
        prev_body_pos_12 = self.prev_body_pos[1:]  # (12, 3)
        body_vel = (cur_body_pos_12 - prev_body_pos_12) / dt
        self.prev_body_pos = cur_body_pos.copy()
        obs[idx:idx+36] = body_vel.flatten()
        idx += 36
        
        return obs
        
    def run(self):
        """主运行循环"""
        gravity_orientation = self.state_cmd.gravity_ori.reshape(-1)  # 3
        qj = self.state_cmd.q.reshape(-1)  # 29
        dqj = self.state_cmd.dq.reshape(-1)  # 29
        ang_vel = self.state_cmd.ang_vel.reshape(-1)  # 3
        
        # 获取base四元数
        base_quat = self.state_cmd.base_quat.reshape(-1)  # 4 (wxyz)

        
        # ==================== 构建robot观测 (123维) ====================
        # 根据训练配置:
        # root_quat_w: 4 (完整四元数 wxyz)
        # root_angvel_b: 3
        # projected_gravity_b: 3
        # joint_pos: 29 (全部关节)
        # joint_vel: 29
        # prev_actions: 19 (steps=1)
        # body_pos: 12 bodies × 3 = 36 (不含pelvis)
        # 总计: 4+3+3+29+29+19+36 = 123
        
        idx = 0
        # root_quat_w (4) -  (wxyz格式)
        self.obs_robot[idx:idx+4] = base_quat
        idx += 4
        
        # root_angvel_b (3)
        self.obs_robot[idx:idx+3] = ang_vel
        idx += 3
        
        # projected_gravity_b (3)
        self.obs_robot[idx:idx+3] = gravity_orientation
        idx += 3
        
        # joint_pos (29) - 全部关节
        self.obs_robot[idx:idx+29] = qj
        idx += 29
        
        # joint_vel (29) - 全部关节
        self.obs_robot[idx:idx+29] = dqj
        idx += 29
        
        # prev_actions (19)
        self.obs_robot[idx:idx+19] = self.prev_action
        idx += 19
        
        # body_pos (36) - 12个keypoint body的位置（不含pelvis，相对于pelvis）
        body_pos = self._get_body_positions()  # (13, 3)
        body_pos_12 = body_pos[1:]  # (12, 3)
        self.obs_robot[idx:idx+36] = body_pos_12.flatten()
        idx += 36
        
        # 剩余维度填零 (如果有的话)
        if idx < self.num_obs_robot:
            self.obs_robot[idx:] = 0.0
        
        # ==================== 构建ref_motion观测 (120维) ====================
        self.obs_ref_motion = self._compute_ref_motion_obs(self.counter_step)
        
        # ==================== 构建priv观测 (40维) ====================
        self.obs_priv = self._compute_priv_obs()

        # ==================== 详细调试信息 ====================
        if self.counter_step % 50 == 0:
            print(f"\n{'='*60}")
            print(f"Step {self.counter_step}:")
            print(f"  Robot obs shape: {self.obs_robot.shape}, range: [{self.obs_robot.min():.3f}, {self.obs_robot.max():.3f}]")
            print(f"  Ref motion obs shape: {self.obs_ref_motion.shape}, range: [{self.obs_ref_motion.min():.3f}, {self.obs_ref_motion.max():.3f}]")
            print(f"  Priv obs shape: {self.obs_priv.shape}, range: [{self.obs_priv.min():.3f}, {self.obs_priv.max():.3f}]")
            print(f"  Base quat: {base_quat}")
            print(f"  Ang vel: {ang_vel}")
            print(f"  Joint pos (first 5): {qj[:5]}")
            print(f"  Joint pos scaled (first 5): {qj_full_scaled[:5]}")
            print(f"  Prev action: {self.prev_action}")
        
        
        # ==================== 模型推理 ====================
        obs_robot_tensor = self.obs_robot.reshape(1, -1).astype(np.float32)
        obs_ref_tensor = self.obs_ref_motion.reshape(1, -1).astype(np.float32)  
        obs_priv_tensor = self.obs_priv.reshape(1, -1).astype(np.float32)

        
        obs_robot_tensor_zero = np.zeros_like(obs_robot_tensor)
        obs_ref_tensor_zero = np.zeros_like(obs_ref_tensor) 
        obs_priv_tensor_zero = np.zeros_like(obs_priv_tensor)
        
        # DEBUG: 使用全0动作测试
        # self.action = np.zeros(self.num_actions, dtype=np.float32)
        
        output = self.ort_session.run(None, {
            "priv": obs_priv_tensor_zero,
            "ref_motion_": obs_ref_tensor_zero,
            "robot": obs_robot_tensor_zero
            
        })[0]
        if self.counter_step % 100 == 0:
            print(f"\nStep {self.counter_step}: Model output (raw): {output}")
        self.action = np.squeeze(output)
        
        if np.any(np.isnan(self.action)) or np.any(np.isinf(self.action)):
            print(f"ERROR: Model output contains NaN or Inf at step {self.counter_step}")
            self.action = np.zeros(self.num_actions, dtype=np.float32)
        else:
            self.action = np.clip(self.action, -10., 10.)
        
        
        # 保存当前动作用于下一步的prev_actions
        self.prev_action = self.action.copy()
        
        # 将19维动作映射到29自由度
        target_dof_pos = self.default_angles.copy()
        for i, dof_idx in enumerate(self.action_to_dof29_index):
            target_dof_pos[dof_idx] = self.action[i] * self.action_scale + self.default_angles[dof_idx]
        
        # 输出到policy_output
        self.policy_output.actions = target_dof_pos
        self.policy_output.kps = self.kps
        self.policy_output.kds = self.kds
        
        # 更新计数器并显示进度
        self.counter_step += 1
        motion_time = self.counter_step * 0.02
        motion_time_display = min(motion_time, self.motion_length)
        print(progress_bar(motion_time_display, self.motion_length), end="", flush=True)
    
    def exit(self):
        """退出状态时的清理"""
        self.counter_step = 0
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.prev_action = np.zeros(self.num_actions, dtype=np.float32)
        print()
    
    def checkChange(self):
        """检查状态转换"""
        if self.state_cmd.skill_cmd == FSMCommand.LOCO:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.SKILL_COOLDOWN
        elif self.state_cmd.skill_cmd == FSMCommand.PASSIVE:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.PASSIVE
        elif self.state_cmd.skill_cmd == FSMCommand.POS_RESET:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.FIXEDPOSE
        else:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.SKILL_MotionTracking
