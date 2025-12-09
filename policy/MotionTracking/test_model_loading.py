#!/usr/bin/env python3
"""
测试脚本：验证 MotionTracking 策略的模型加载功能
"""

import numpy as np
import sys
import os

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.insert(0, project_root)

def test_model_loading(model_type="onnx"):
    """测试模型加载"""
    print(f"\n{'='*60}")
    print(f"测试 {model_type.upper()} 模型加载")
    print(f"{'='*60}\n")
    
    # 临时修改配置
    import yaml
    config_path = os.path.join(current_dir, "config", "MotionTracking.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    original_model_type = config.get("model_type", "onnx")
    config["model_type"] = model_type
    
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    try:
        from common.ctrlcomp import StateAndCmd, PolicyOutput
        from policy.MotionTracking.MotionTracking import MotionTracking
        
        # 创建测试对象
        state_cmd = StateAndCmd()
        policy_output = PolicyOutput()
        
        # 初始化策略
        print("正在初始化策略...")
        policy = MotionTracking(state_cmd, policy_output)
        
        # 创建测试数据
        print("\n创建测试数据...")
        state_cmd.gravity_ori = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        state_cmd.q = np.zeros(29, dtype=np.float32)
        state_cmd.dq = np.zeros(29, dtype=np.float32)
        state_cmd.ang_vel = np.zeros(3, dtype=np.float32)
        state_cmd.base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # 进入状态
        print("进入状态...")
        policy.enter()
        
        # 运行几步测试
        print("\n运行推理测试...")
        for i in range(5):
            policy.run()
            print(f"  步骤 {i+1}: 动作输出形状 = {policy_output.actions.shape}, "
                  f"范围 = [{policy_output.actions.min():.4f}, {policy_output.actions.max():.4f}]")
        
        # 退出状态
        policy.exit()
        
        print(f"\n✅ {model_type.upper()} 模型测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ {model_type.upper()} 模型测试失败！")
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # 恢复原始配置
        config["model_type"] = original_model_type
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"\n配置已恢复为: {original_model_type}")


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("MotionTracking 模型加载测试")
    print("="*60)
    
    results = {}
    
    # 测试 ONNX 模型
    print("\n[1/2] 测试 ONNX 模型...")
    results["onnx"] = test_model_loading("onnx")
    
    # 检查 PyTorch 是否可用
    try:
        import torch
        pytorch_available = True
    except ImportError:
        pytorch_available = False
        print("\n⚠️  PyTorch 未安装，跳过 PyTorch 模型测试")
    
    # 测试 PyTorch 模型（如果可用）
    if pytorch_available:
        print("\n[2/2] 测试 PyTorch 模型...")
        
        # 检查 PyTorch 模型文件是否存在
        pt_model_path = os.path.join(current_dir, "model", "model.pt")
        if os.path.exists(pt_model_path):
            results["pytorch"] = test_model_loading("pytorch")
        else:
            print(f"\n⚠️  PyTorch 模型文件不存在: {pt_model_path}")
            print("跳过 PyTorch 模型测试")
            results["pytorch"] = None
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    for model_type, result in results.items():
        if result is True:
            status = "✅ 通过"
        elif result is False:
            status = "❌ 失败"
        else:
            status = "⏭️  跳过"
        print(f"{model_type.upper():10s}: {status}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
