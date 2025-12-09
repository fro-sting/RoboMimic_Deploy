#!/usr/bin/env python3
"""
调试脚本 - 诊断ONNX模型输出问题
帮助判断是模型问题还是代码问题
"""

import numpy as np
import onnxruntime
import onnx
import os
import yaml
import joblib

def check_onnx_model(onnx_path):
    """检查ONNX模型结构"""
    print(f"\n{'='*60}")
    print("1. 检查ONNX模型结构")
    print(f"{'='*60}")
    
    try:
        model = onnx.load(onnx_path)
        print(f"✓ ONNX模型加载成功: {onnx_path}")
        
        # 检查输入
        print("\n模型输入:")
        for inp in model.graph.input:
            print(f"  - {inp.name}: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")
        
        # 检查输出
        print("\n模型输出:")
        for out in model.graph.output:
            print(f"  - {out.name}: {[d.dim_value for d in out.type.tensor_type.shape.dim]}")
        
        return True
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return False

def test_model_inference(onnx_path, num_obs_robot, num_obs_ref_motion, num_obs_priv, num_actions):
    """测试模型推理"""
    print(f"\n{'='*60}")
    print("2. 测试模型推理（多组输入）")
    print(f"{'='*60}")
    
    try:
        session = onnxruntime.InferenceSession(onnx_path)
        print(f"✓ ONNX Runtime会话创建成功")
        
        test_cases = [
            ("全零输入", np.zeros),
            ("全1输入", np.ones),
            ("小随机输入 [-0.1, 0.1]", lambda shape: np.random.uniform(-0.1, 0.1, shape)),
            ("正态分布输入 N(0,1)", lambda shape: np.random.randn(*shape)),
        ]
        
        results = []
        
        for test_name, input_gen in test_cases:
            print(f"\n测试: {test_name}")
            
            obs_robot = input_gen((1, num_obs_robot)).astype(np.float32)
            obs_ref = input_gen((1, num_obs_ref_motion)).astype(np.float32)
            obs_priv = input_gen((1, num_obs_priv)).astype(np.float32)
            
            try:
                output = session.run(None, {
                    "robot": obs_robot,
                    "ref_motion_": obs_ref,
                    "priv": obs_priv
                })[0]
                
                output_squeezed = np.squeeze(output)
                
                # 检查输出
                has_nan = np.any(np.isnan(output_squeezed))
                has_inf = np.any(np.isinf(output_squeezed))
                
                print(f"  输出形状: {output_squeezed.shape}")
                print(f"  输出范围: [{output_squeezed.min():.4f}, {output_squeezed.max():.4f}]")
                print(f"  输出均值: {output_squeezed.mean():.4f}")
                print(f"  输出标准差: {output_squeezed.std():.4f}")
                print(f"  包含NaN: {has_nan}")
                print(f"  包含Inf: {has_inf}")
                
                if not has_nan and not has_inf:
                    print(f"  ✓ 输出正常")
                else:
                    print(f"  ✗ 输出异常!")
                
                results.append({
                    'name': test_name,
                    'output': output_squeezed,
                    'valid': not (has_nan or has_inf)
                })
                
                # 显示前几个输出值
                print(f"  前5个输出值: {output_squeezed[:5]}")
                
            except Exception as e:
                print(f"  ✗ 推理失败: {e}")
                results.append({'name': test_name, 'valid': False, 'error': str(e)})
        
        return results
        
    except Exception as e:
        print(f"✗ 创建推理会话失败: {e}")
        return []

def test_with_motion_data(onnx_path, motion_file, config):
    """使用真实动作数据测试"""
    print(f"\n{'='*60}")
    print("3. 使用真实动作数据测试")
    print(f"{'='*60}")
    
    if not os.path.exists(motion_file):
        print(f"✗ 动作文件不存在: {motion_file}")
        return
    
    try:
        # 加载动作数据
        print(f"加载动作数据: {motion_file}")
        data = joblib.load(motion_file)
        
        if isinstance(data, dict):
            motion_names = list(data.keys())
            motion = data[motion_names[0]]
            print(f"✓ 加载动作: {motion_names[0]}")
        else:
            motion = data
            print(f"✓ 加载单个动作")
        
        # 打印动作数据结构
        print(f"\n动作数据结构:")
        for key in motion.keys():
            if isinstance(motion[key], np.ndarray):
                print(f"  {key}: shape={motion[key].shape}, dtype={motion[key].dtype}")
        
        # 提取第一帧数据
        joint_pos = np.array(motion["joint_pos"][0], dtype=np.float32)
        joint_vel = np.array(motion["joint_vel"][0], dtype=np.float32)
        body_pos_w = np.array(motion["body_pos_w"][0], dtype=np.float32)
        
        print(f"\n第一帧数据:")
        print(f"  joint_pos: {joint_pos[:5]} ... (显示前5个)")
        print(f"  joint_vel: {joint_vel[:5]} ... (显示前5个)")
        print(f"  body_pos_w shape: {body_pos_w.shape}")
        
        # 构建观测（简化版）
        num_obs_robot = config["num_obs_robot"]
        num_obs_ref_motion = config["num_obs_ref_motion"]
        num_obs_priv = config["num_obs_priv"]
        
        # 使用参考数据构建观测
        obs_robot = np.zeros(num_obs_robot, dtype=np.float32)
        obs_ref = np.zeros(num_obs_ref_motion, dtype=np.float32)
        obs_priv = np.zeros(num_obs_priv, dtype=np.float32)
        
        # 填充一些基本数据
        # robot obs: base_quat(4) + ang_vel(3) + gravity(3) + joint_pos(29) + joint_vel(29) + ...
        obs_robot[0:4] = [1.0, 0.0, 0.0, 0.0]  # base quaternion
        obs_robot[7:10] = [0.0, 0.0, -1.0]  # gravity
        obs_robot[10:39] = joint_pos  # joint positions
        obs_robot[39:68] = joint_vel  # joint velocities
        
        # ref motion obs: 包含参考joint pos
        obs_ref[0:29] = joint_pos  # ref_qpos
        
        print(f"\n构建观测:")
        print(f"  obs_robot shape: {obs_robot.shape}, range: [{obs_robot.min():.3f}, {obs_robot.max():.3f}]")
        print(f"  obs_ref shape: {obs_ref.shape}, range: [{obs_ref.min():.3f}, {obs_ref.max():.3f}]")
        print(f"  obs_priv shape: {obs_priv.shape}, range: [{obs_priv.min():.3f}, {obs_priv.max():.3f}]")
        
        # 推理
        session = onnxruntime.InferenceSession(onnx_path)
        output = session.run(None, {
            "robot": obs_robot.reshape(1, -1),
            "ref_motion_": obs_ref.reshape(1, -1),
            "priv": obs_priv.reshape(1, -1)
        })[0]
        
        output_squeezed = np.squeeze(output)
        
        print(f"\n模型输出:")
        print(f"  形状: {output_squeezed.shape}")
        print(f"  范围: [{output_squeezed.min():.4f}, {output_squeezed.max():.4f}]")
        print(f"  完整输出:\n{output_squeezed}")
        
        # 应用action_scale
        action_scale = config["action_scale"]
        default_angles = np.array(config["default_angles"], dtype=np.float32)
        
        # 19个控制关节的索引
        action_to_dof29_index = np.array([
            0, 1, 2, 6, 7, 8, 3, 9, 4, 10, 12,
            15, 16, 17, 22, 23, 24, 18, 25
        ], dtype=np.int32)
        
        target_dof_pos = default_angles.copy()
        for i, dof_idx in enumerate(action_to_dof29_index):
            target_dof_pos[dof_idx] = output_squeezed[i] * action_scale + default_angles[dof_idx]
        
        print(f"\n应用action_scale后的目标位置:")
        print(f"  目标位置 (前10个): {target_dof_pos[:10]}")
        print(f"  与默认位置的差异 (前10个): {target_dof_pos[:10] - default_angles[:10]}")
        
        # 检查是否有过大的变化
        max_diff = np.abs(target_dof_pos - default_angles).max()
        print(f"  最大位置变化: {max_diff:.4f}")
        if max_diff > 1.0:
            print(f"  ⚠ 警告: 位置变化过大! 可能导致机器人不稳定")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def check_observation_ranges(config):
    """检查观测数据的预期范围"""
    print(f"\n{'='*60}")
    print("4. 观测数据范围检查")
    print(f"{'='*60}")
    
    print(f"\n配置参数:")
    print(f"  ang_vel_scale: {config.get('ang_vel_scale', 1.0)}")
    print(f"  dof_pos_scale: {config.get('dof_pos_scale', 1.0)}")
    print(f"  dof_vel_scale: {config.get('dof_vel_scale', 1.0)}")
    print(f"  action_scale: {config.get('action_scale', 1.0)}")
    
    print(f"\n观测维度:")
    print(f"  num_obs_robot: {config['num_obs_robot']}")
    print(f"  num_obs_ref_motion: {config['num_obs_ref_motion']}")
    print(f"  num_obs_priv: {config['num_obs_priv']}")
    print(f"  num_actions: {config['num_actions']}")
    
    print(f"\nPD控制参数:")
    kps = np.array(config['kps'])
    kds = np.array(config['kds'])
    print(f"  kps 范围: [{kps.min():.1f}, {kps.max():.1f}]")
    print(f"  kds 范围: [{kds.min():.2f}, {kds.max():.2f}]")
    
    tau_limit = np.array(config['tau_limit'])
    print(f"  tau_limit 范围: [{tau_limit.min():.1f}, {tau_limit.max():.1f}]")

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config", "MotionTracking.yaml")
    
    print(f"{'='*60}")
    print("ONNX模型诊断工具")
    print(f"{'='*60}")
    
    # 加载配置
    try:
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        print(f"✓ 配置加载成功: {config_path}")
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        return
    
    onnx_path = os.path.join(current_dir, "model", config["onnx_path"])
    motion_file = config.get("motion_file", None)
    if motion_file:
        motion_path = os.path.join(current_dir, motion_file)
    else:
        motion_path = None
    
    num_obs_robot = config["num_obs_robot"]
    num_obs_ref_motion = config["num_obs_ref_motion"]
    num_obs_priv = config["num_obs_priv"]
    num_actions = config["num_actions"]
    
    # 1. 检查模型结构
    if not check_onnx_model(onnx_path):
        return
    
    # 2. 测试模型推理
    results = test_model_inference(onnx_path, num_obs_robot, num_obs_ref_motion, num_obs_priv, num_actions)
    
    # 3. 使用真实数据测试
    if motion_path and os.path.exists(motion_path):
        test_with_motion_data(onnx_path, motion_path, config)
    else:
        print(f"\n跳过真实数据测试 (motion_file未配置或不存在)")
    
    # 4. 检查观测范围
    check_observation_ranges(config)
    
    # 总结
    print(f"\n{'='*60}")
    print("诊断总结")
    print(f"{'='*60}")
    
    all_valid = all(r.get('valid', False) for r in results)
    
    if all_valid:
        print("✓ 模型推理测试全部通过")
        print("\n建议检查:")
        print("  1. 观测数据的构建是否正确 (特别是body_pos计算)")
        print("  2. 四元数格式是否一致 (wxyz vs xyzw)")
        print("  3. 坐标系转换是否正确")
        print("  4. action_scale是否合适")
        print("  5. PD控制器参数是否合理")
    else:
        print("✗ 模型推理测试发现问题")
        print("\n可能的原因:")
        print("  1. ONNX模型本身有问题 (导出时出错)")
        print("  2. 模型输入维度不匹配")
        print("  3. 模型权重未正确导出")

if __name__ == "__main__":
    main()
