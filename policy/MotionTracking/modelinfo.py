import onnx
import onnxruntime
import numpy as np

model_path = "model/model2.onnx"
model = onnx.load(model_path)

print("=" * 50)
print("模型输入信息:")
for inp in model.graph.input:
    print(f"  名称: {inp.name}")
    shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in inp.type.tensor_type.shape.dim]
    print(f"  形状: {shape}")

print("=" * 50)
print("模型输出信息:")
for out in model.graph.output:
    print(f"  名称: {out.name}")
    shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in out.type.tensor_type.shape.dim]
    print(f"  形状: {shape}")

# 使用 onnxruntime 实际推理来获取真实维度
print("=" * 50)
print("实际推理测试:")
ort_session = onnxruntime.InferenceSession(model_path)

# 获取所有输入信息
print("所有输入:")
for inp in ort_session.get_inputs():
    print(f"  {inp.name}: {inp.shape}")

# 获取输出信息
output_info = ort_session.get_outputs()[0]
print(f"输出: {output_info.name}: {output_info.shape}")

# 做一次实际推理 - 需要提供所有3个输入
dummy_robot = np.zeros((1, 123), dtype=np.float32)
dummy_ref_motion = np.zeros((1, 120), dtype=np.float32)
dummy_priv = np.zeros((1, 40), dtype=np.float32)

output = ort_session.run(None, {
    "robot": dummy_robot,
    "ref_motion_": dummy_ref_motion,
    "priv": dummy_priv
})[0]

print("=" * 50)
print(f"实际输出形状: {output.shape}")
print(f"num_actions = {output.shape[1]}")
print("=" * 50)
print("\n配置文件需要的参数:")
print(f"  num_obs_robot: 123")
print(f"  num_obs_ref_motion: 120")
print(f"  num_obs_priv: 40")
print(f"  num_actions: {output.shape[1]}")