import torch
import numpy as np
import random
import torch.optim as optim


from model import YOLOv8OBBDepthModel
import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
import cv2
from torch.utils.data import DataLoader
import torch.nn.functional as F

def scale_invariant_loss(pred, target, valid_mask=None):
    """
    pred, target: (B, 1, H, W)
    valid_mask: (B, 1, H, W) or None
    """
    if valid_mask is not None:
        pred = pred[valid_mask]
        target = target[valid_mask]

    diff = pred - target
    diff2 = diff ** 2
    first_term = diff2.mean()
    second_term = 0.5 * (diff.sum() ** 2) / (diff.numel() ** 2)
    return first_term - second_term


class OBBDepthDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir, depth_dir, img_size=640):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.depth_dir = Path(depth_dir)

        self.image_files = sorted(self.image_dir.glob("*.png"))
        self.img_size = img_size

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        # 加载 RGB 图像
        img_path = self.image_files[index]
        name = img_path.stem
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # 加载标签
        label_path = self.label_dir / f"{name}.txt"
        labels = []
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = list(map(float, line.strip().split()))
                    class_id = int(parts[0])
                    points = parts[1:]  # 8 个点坐标
                    labels.append([class_id] + points)
        labels = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 9), dtype=torch.float32)

        # 加载深度图并缩放
        depth_name = f"depth_{name.split('_', 1)[1]}.png"
        depth_path = self.depth_dir / depth_name
        if not depth_path.exists():
            depth_img = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        else:
            depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            depth_img = cv2.resize(depth_img, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # 生成 3 尺寸下采样深度图
        depth_80 = cv2.resize(depth_img, (80, 80), interpolation=cv2.INTER_NEAREST)
        depth_40 = cv2.resize(depth_img, (40, 40), interpolation=cv2.INTER_NEAREST)
        depth_20 = cv2.resize(depth_img, (20, 20), interpolation=cv2.INTER_NEAREST)

        # 转 tensor 并加通道维度
        depth_80 = torch.from_numpy(depth_80).unsqueeze(0).float()  # shape: 1x80x80
        depth_40 = torch.from_numpy(depth_40).unsqueeze(0).float()
        depth_20 = torch.from_numpy(depth_20).unsqueeze(0).float()

        return {
            "img": img,             # (3, 640, 640)
            "label": labels,        # (N, 9) -> [class_id, x1, y1, ..., x4, y4]
            "depth": [depth_80, depth_40, depth_20],  # list of 3 tensors
            "name": name
        }
    

def train(image_dir, label_dir, depth_dir,nc=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型
    model = YOLOv8OBBDepthModel(nc=nc).to(device)
    model.train()

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=1e-2)

    # 数据集
    train_dataset = OBBDepthDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        depth_dir=depth_dir
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4,collate_fn=custom_collate_fn)

    # 开始训练
    num_epochs = 1000
    for epoch in range(num_epochs):
        total_depth_loss = 0.0
        for batch in train_loader:
            model.train()
            imgs = batch["img"].to(device)              # (B, 3, 640, 640)
            depths_gt = [d.to(device) for d in batch["depth"]]  # [(B,1,80,80), ...]

            optimizer.zero_grad()
            _, depth_preds = model(imgs)  # depth_preds: list of 3 tensors

            # 计算多尺度 Scale-Invariant Loss
            loss = 0
            for pred, gt in zip(depth_preds, depths_gt):
                loss += scale_invariant_loss(pred, gt)
            loss.backward()
            optimizer.step()

            total_depth_loss += loss.item()

        print(f"[Epoch {epoch+1}] Depth Loss: {total_depth_loss:.4f}")

        val_loader = get_val_loader_from_train(train_dataset, num_samples=10, batch_size=1)
        model.eval()
        with torch.no_grad():
            total_rmse = 0.0
            for batch in val_loader:
                img = batch["img"].to(device)
                gt_depths = [d.to(device) for d in batch["depth"]]
                _, pred_depths = model(img)
                depth_pred_80 = pred_depths[0]  # 只取模型输出中的 80x80 输出

                rmse_vals = compute_rmse(depth_pred_80, gt_depths)
                total_rmse += np.mean(rmse_vals)
        print(f"[Epoch {epoch+1}] Validation RMSE: {total_rmse/len(val_loader):.4f}")
    # 保存模型
        torch.save(model.state_dict(), "obb_depth_model.pth")

def compute_rmse(pred_depths, gt_depths):
    """
    计算每个尺度下的 RMSE（Root Mean Squared Error）。

    Args:
        pred_depths (list[torch.Tensor]): 预测深度图列表 [B, 1, H, W]
        gt_depths (list[torch.Tensor]):   GT 深度图列表 [B, 1, H, W]

    Returns:
        List[float]: 每个尺度的 RMSE 值
    """
    rmse_list = []

    for pred, gt in zip(pred_depths, gt_depths):
        # valid_mask = gt > 0
        # pred = pred[valid_mask]
        # gt = gt[valid_mask]

        if pred.numel() == 0:
            rmse_list.append(float("nan"))
            continue

        mse = torch.mean((pred - gt) ** 2)
        rmse = torch.sqrt(mse).item()
        rmse_list.append(rmse)

    return rmse_list

def custom_collate_fn(batch):
    imgs = torch.stack([b["img"] for b in batch])
    labels = [b["label"] for b in batch]  # 保持为 list of tensors
    depths = [b["depth"] for b in batch]  # list of list (3 tensors)
    names = [b["name"] for b in batch]
    
    # 深度图 [B, 1, H, W] * 3
    depths_80 = torch.stack([d[0] for d in depths])
    depths_40 = torch.stack([d[1] for d in depths])
    depths_20 = torch.stack([d[2] for d in depths])

    return {
        "img": imgs,
        "label": labels,
        "depth": [depths_80, depths_40, depths_20],
        "name": names,
    }
from torch.utils.data import Subset, DataLoader
import random

def get_val_loader_from_train(train_dataset, num_samples=10, batch_size=1, seed=42):
    """
    从训练集中随机采样 num_samples 个样本构建验证 DataLoader

    Args:
        train_dataset (Dataset): 原始训练集
        num_samples (int): 验证样本数量
        batch_size (int): 验证时的 batch size
        seed (int): 随机种子，确保可复现

    Returns:
        DataLoader: 构建好的验证集加载器
    """
    random.seed(seed)
    indices = random.sample(range(len(train_dataset)), num_samples)
    val_subset = Subset(train_dataset, indices)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    return val_loader

if __name__ == "__main__":
    model = YOLOv8OBBDepthModel(nc=3) 
    
    # tomato leaf 数据集路径
    # image_dir="/home/ihpc/code/yolo/yolo-obb-depth/dataset/tomato_leaf_split/train/images"
    # label_dir="/home/ihpc/code/yolo/yolo-obb-depth/dataset/tomato_leaf_split/train/labels"
    # depth_dir="/home/ihpc/code/yolo/yolo-obb-depth/dataset/tomato_leaf_split/train/depth"
    # x = torch.randn(16, 3, 640, 640)
    # det,depth = model(x)
    # model.train()
    # train(image_dir, label_dir, depth_dir,nc=3)

    image_dir="/home/ihpc/code/yolo/yolo-obb-depth/dataset/nyudepthv2/nyu_preprocessed/nyu_rgb"
    label_dir="/home/ihpc/code/yolo/yolo-obb-depth/dataset/nyudepthv2/nyu_preprocessed/dummy_label"
    depth_dir="/home/ihpc/code/yolo/yolo-obb-depth/dataset/nyudepthv2/nyu_preprocessed/nyu_depth"
    x = torch.randn(16, 3, 640, 640)
    det,depth = model(x)
    model.train()
    train(image_dir, label_dir, depth_dir,nc=3)


