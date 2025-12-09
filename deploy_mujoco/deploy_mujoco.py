import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from common.path_config import PROJECT_ROOT

import time
import mujoco.viewer
import mujoco
import numpy as np
import yaml
import os
from common.ctrlcomp import *
from FSM.FSM import *
from common.utils import get_gravity_orientation
from common.keyboard import KeyBoard, KeyboardKey



def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mujoco_yaml_path = os.path.join(current_dir, "config", "mujoco.yaml")
    with open(mujoco_yaml_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        xml_path = os.path.join(PROJECT_ROOT, config["xml_path"])
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]
        
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    mj_per_step_duration = simulation_dt * control_decimation
    num_joints = m.nu
    policy_output_action = np.zeros(num_joints, dtype=np.float32)
    kps = np.zeros(num_joints, dtype=np.float32)
    kds = np.zeros(num_joints, dtype=np.float32)
    sim_counter = 0
    
    state_cmd = StateAndCmd(num_joints)
    policy_output = PolicyOutput(num_joints)
    FSM_controller = FSM(state_cmd, policy_output)
    
    # 设置MuJoCo数据引用，用于MotionTracking等需要body位置的策略
    FSM_controller.set_mujoco_data(d, m)
    
    keyboard = KeyBoard()
    Running = True
    with mujoco.viewer.launch_passive(m, d) as viewer:
        sim_start_time = time.time()
        while viewer.is_running() and Running:
            try:
                keyboard.update()
                
                # 9: 退出
                if keyboard.is_key_pressed(KeyboardKey.KEY_9):
                    Running = False

                # 2: PASSIVE (阻尼保护)
                if keyboard.is_key_released(KeyboardKey.KEY_2):
                    state_cmd.skill_cmd = FSMCommand.PASSIVE
                    print("[键盘] 切换到: PASSIVE (阻尼保护)")
                
                # 3: POS_RESET (位控复位)
                if keyboard.is_key_released(KeyboardKey.KEY_3):
                    state_cmd.skill_cmd = FSMCommand.POS_RESET
                    print("[键盘] 切换到: POS_RESET (位控复位)")
                
                # 4: LOCO (行走模式)
                if keyboard.is_key_released(KeyboardKey.KEY_4):
                    state_cmd.skill_cmd = FSMCommand.LOCO
                    print("[键盘] 切换到: LOCO (行走模式)")
                
                # 5: SKILL_1 (舞蹈)
                if keyboard.is_key_released(KeyboardKey.KEY_5):
                    state_cmd.skill_cmd = FSMCommand.SKILL_1
                    print("[键盘] 切换到: SKILL_1 (舞蹈)")
                
                # 6: SKILL_2 (武术)
                if keyboard.is_key_released(KeyboardKey.KEY_6):
                    state_cmd.skill_cmd = FSMCommand.SKILL_2
                    print("[键盘] 切换到: SKILL_2 (武术)")
                
                # 7: SKILL_3 (武术2)
                if keyboard.is_key_released(KeyboardKey.KEY_7):
                    state_cmd.skill_cmd = FSMCommand.SKILL_3
                    print("[键盘] 切换到: SKILL_3 (武术2)")
                
                # 8: SKILL_4 (踢腿)
                if keyboard.is_key_released(KeyboardKey.KEY_8):
                    state_cmd.skill_cmd = FSMCommand.SKILL_4
                    print("[键盘] 切换到: SKILL_4 (踢腿)")
                
                # T: SKILL_5 (MotionTracking)
                if keyboard.is_key_released(KeyboardKey.KEY_T):
                    state_cmd.skill_cmd = FSMCommand.SKILL_5
                    print("[键盘] 切换到: SKILL_5 (MotionTracking)")
                
                # 获取速度命令
                vel_x, vel_y, vel_yaw = keyboard.get_velocity()
                state_cmd.vel_cmd[0] = vel_x
                state_cmd.vel_cmd[1] = vel_y
                state_cmd.vel_cmd[2] = vel_yaw
                
                step_start = time.time()
                
                tau = pd_control(policy_output_action, d.qpos[7:], kps, np.zeros_like(kps), d.qvel[6:], kds)
                d.ctrl[:] = tau
                mujoco.mj_step(m, d)
                sim_counter += 1
                if sim_counter % control_decimation == 0:
                    
                    qj = d.qpos[7:]
                    dqj = d.qvel[6:]
                    quat = d.qpos[3:7]
                    
                    omega = d.qvel[3:6] 
                    gravity_orientation = get_gravity_orientation(quat)
                    
                    state_cmd.q = qj.copy()
                    state_cmd.dq = dqj.copy()
                    state_cmd.gravity_ori = gravity_orientation.copy()
                    state_cmd.base_quat = quat.copy()
                    state_cmd.ang_vel = omega.copy()
                    
                    FSM_controller.run()
                    policy_output_action = policy_output.actions.copy()
                    kps = policy_output.kps.copy()
                    kds = policy_output.kds.copy()
            except ValueError as e:
                print(str(e))
            
            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    keyboard.stop()