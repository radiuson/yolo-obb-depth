import h5py
import numpy as np
from pathlib import Path
import imageio
from tqdm import tqdm

# 加载 .mat 文件（v7.3 格式）
mat_path = "/home/ihpc/code/yolo/yolo-obb-depth/dataset/nyudepthv2/nyu_depth_v2_labeled.mat"
f = h5py.File(mat_path, 'r')

# 创建输出目录
out_rgb = Path("./nyu_rgb")
out_depth = Path("./nyu_depth")
out_rgb.mkdir(parents=True, exist_ok=True)
out_depth.mkdir(parents=True, exist_ok=True)
print("✅ 文件变量键：", list(f.keys()))
print("✅ 图像 shape:", f['images'].shape)   # (3, 640, 480, 1449)
print("✅ 深度 shape:", f['depths'].shape)   # (640, 480, 1449)

# 获取变量
imgs = f['images']       # shape: (3, 640, 480, 1449)
depths = f['depths']     # shape: (640, 480, 1449)

n = imgs.shape[-1]       # 1449

for i in tqdm(range(n)):
    rgb = f['images'][i]  # shape: (3, 640, 480)
    print(f"\n🔍 第 {i} 张图像原始 shape: {rgb.shape}")

    rgb = np.transpose(rgb, (1, 2, 0))  # (640, 480, 3)
    print(f"✅ 转置后 shape: {rgb.shape}")  # 应该是 (640, 480, 3)

    print(f"🧪 RGB 像素范围: min={rgb.min()} max={rgb.max()} dtype={rgb.dtype}")
    rgb = np.ascontiguousarray(rgb).astype(np.uint8)

    depth = f['depths'][i]  # shape: (640, 480)
    depth = np.array(depth)
    print(f"✅ 深度 shape: {depth.shape} range: {depth.min():.3f}~{depth.max():.3f} m")

    # 保存图像
    imageio.imwrite(out_rgb / f"rgb_{i:04d}.png", rgb)
    depth_mm = (depth * 1000).astype(np.uint16)
    imageio.imwrite(out_depth / f"depth_{i:04d}.png", depth_mm)
