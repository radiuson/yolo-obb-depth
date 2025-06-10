import torch
import torch.nn as nn
import numpy as np
import random
import torch.optim as optim
import datetime 

from loss import scale_invariant_loss, batch_obb_loss_fn, xyxyxyxy2xywhr, BboxLoss
from taskal import RotatedTaskAlignedAssigner, make_anchors

from model import YOLOv8OBBDepthModel
import os
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
import cv2
from torch.utils.data import DataLoader
import torch.nn.functional as F


def analyze_structure(var, name="var", indent=0):
    """递归分析变量结构（支持 Tensor、list、tuple、dict 等）"""
    prefix = " " * indent
    if isinstance(var, torch.Tensor):
        print(f"{prefix}{name}: Tensor | shape={tuple(var.shape)} | dtype={var.dtype} | device={var.device}")
    elif isinstance(var, (list, tuple)):
        print(f"{prefix}{name}: {type(var).__name__} | len={len(var)}")
        for i, item in enumerate(var):
            analyze_structure(item, name=f"[{i}]", indent=indent + 2)
    elif isinstance(var, dict):
        print(f"{prefix}{name}: dict | keys={list(var.keys())}")
        for k, v in var.items():
            analyze_structure(v, name=f'["{k}"]', indent=indent + 2)
    else:
        print(f"{prefix}{name}: {type(var).__name__} | value={var}")

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

        mse = torch.mean((pred/1000 - gt/1000) ** 2)
        rmse = torch.sqrt(mse).item()
        rmse_list.append(rmse)

    return rmse_list


def custom_collate_fn_rotated_depth(batch, img_size=(640, 640), max_gt=60):
    imgs = torch.stack([b["img"] for b in batch])               # [B, 3, H, W]
    labels_raw = [b["label"] for b in batch]                    # list of [N, 9]
    depths = [b["depth"] for b in batch]                        # list of list(3)
    names = [b["name"] for b in batch]

    B = len(batch)
    gt_labels = torch.zeros((B, max_gt, 1), dtype=torch.long)
    gt_bboxes = torch.zeros((B, max_gt, 5), dtype=torch.float32)
    mask_gt = torch.zeros((B, max_gt, 1), dtype=torch.bool)
    batch_idx_list = []
    cls_list = []
    rboxes_list = []
    
    for i, labels in enumerate(labels_raw):
        
        if len(labels) == 0:
            continue
        labels = np.array(list(labels), dtype=np.float32)
        classes = labels[:, 0]
    
        rboxes = torch.tensor(xyxyxyxy2xywhr(labels[:, 1:9]), dtype=torch.float32) # 转为 [cx, cy, w, h, theta] 格式

        batch_idx_list.extend([i] * len(classes))
        n = min(len(classes), max_gt)
        gt_labels[i, :n, 0] = torch.tensor(classes[:n], dtype=torch.long)
        gt_bboxes[i, :n] = rboxes[:n]
        mask_gt[i, :n, 0] = 1
    batch_idx_list = torch.tensor(batch_idx_list, dtype=torch.long)  # [N,]
    # 深度图 3 级尺度：list of [B, 1, H, W]
    depths_80 = torch.stack([d[0] for d in depths])
    depths_40 = torch.stack([d[1] for d in depths])
    depths_20 = torch.stack([d[2] for d in depths])

    return {
        "img": imgs,
        "batch_idx": batch_idx_list,
        "label": labels_raw,
        "depth": [depths_80, depths_40, depths_20],
        "name": names,
        "gt_labels": gt_labels,
        "gt_bboxes": gt_bboxes,
        "mask_gt": mask_gt
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




def train(image_dir, label_dir, depth_dir, nc=2, save_name=None, weight_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reg_max = 16
    # 自动生成模型保存名
    if save_name is None:
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"obb_depth_{now}.pth"

    # 模型
    model = YOLOv8OBBDepthModel(nc=nc).to(device)
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path, map_location=device))

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=1e-2)

    # 数据集
    train_dataset = OBBDepthDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        depth_dir=depth_dir
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=custom_collate_fn_rotated_depth)

    # Assigner
    assigner = RotatedTaskAlignedAssigner(
        topk=13,
        num_classes=nc,
        alpha=1.0,
        beta=6.0,
        eps=1e-9,
    )
    bce = nn.BCEWithLogitsLoss(reduction="none")
    bbox_loss = BboxLoss(reg_max).to(device)

    num_epochs = 500
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_det_loss = 0.0
        total_depth_loss = 0.0

        for batch in train_loader:
            model.train()
            imgs = batch["img"].to(device)
            labels = batch["label"]
            depths_gt = [d.to(device) for d in batch["depth"]]
            gt_labels = batch["gt_labels"].to(device)
            gt_bboxes = batch["gt_bboxes"].to(device)
            mask_gt = batch["mask_gt"].to(device)

            optimizer.zero_grad()
            det_out, depth_preds = model(imgs)
            
            obb_loss = batch_obb_loss_fn(det_out,batch,device,assigner,bce,bbox_loss)

            # ===== Depth Loss =====
            depth_loss = sum(scale_invariant_loss(pred, gt, gt > 0) for pred, gt in zip(depth_preds, depths_gt))

            # ===== Backward =====
            depth_train = False
            loss = depth_loss + obb_loss
            loss.backward()
            optimizer.step()


        # ===== 打印 epoch 统计 =====
        print(f"[Epoch {epoch+1}] Total Loss: {total_loss:.4f} | Detection: {total_det_loss:.4f} | Depth: {total_depth_loss:.4f}")
        if depth_train is True:
            # ===== 每轮验证 RMSE（可选）=====
            val_loader = get_val_loader_from_train(train_dataset, num_samples=1, batch_size=1)
            model.eval()
            with torch.no_grad():
                total_rmse = 0.0
                for batch in val_loader:
                    img = batch["img"].to(device)
                    gt_depths = [d.to(device) for d in batch["depth"]]
                    _, pred_depths = model(img)
                    depth_pred_80 = pred_depths[0]
                    rmse_vals = compute_rmse(depth_pred_80, gt_depths)
                    total_rmse += np.mean(rmse_vals)
            print(f"[Epoch {epoch+1}] Validation RMSE: {total_rmse/len(val_loader):.4f}")

        # ===== 保存模型 =====
        torch.save(model.state_dict(), save_name)

    def analyze_structure(var, name="var", indent=0):
            """递归分析变量结构（支持 Tensor、list、tuple、dict 等）"""
            prefix = " " * indent
            if isinstance(var, torch.Tensor):
                print(f"{prefix}{name}: Tensor | shape={tuple(var.shape)} | dtype={var.dtype} | device={var.device}")
            elif isinstance(var, (list, tuple)):
                print(f"{prefix}{name}: {type(var).__name__} | len={len(var)}")
                for i, item in enumerate(var):
                    analyze_structure(item, name=f"[{i}]", indent=indent + 2)
            elif isinstance(var, dict):
                print(f"{prefix}{name}: dict | keys={list(var.keys())}")
                for k, v in var.items():
                    analyze_structure(v, name=f'["{k}"]', indent=indent + 2)
            else:
                print(f"{prefix}{name}: {type(var).__name__} | value={var}")
if __name__ == "__main__":
    model = YOLOv8OBBDepthModel(nc=2) 
    
    # tomato leaf 数据集路径
    image_dir="/home/ihpc/code/yolo/yolo-obb-depth/dataset/tomato_leaf_split/train/images"
    label_dir="/home/ihpc/code/yolo/yolo-obb-depth/dataset/tomato_leaf_split/train/labels"
    depth_dir="/home/ihpc/code/yolo/yolo-obb-depth/dataset/tomato_leaf_split/train/depth"
    # train_dataset = OBBDepthDataset(
    #         image_dir=image_dir,
    #         label_dir=label_dir,
    #         depth_dir=depth_dir
    #     )


    # train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=custom_collate_fn_rotated_depth)
    # train_iter = iter(train_loader)
    # batch = next(train_iter)
    
    model.train()
    train(image_dir, label_dir, depth_dir,nc=2, save_name="yolo_obb_depth_tomato_leaf.pth")

    # NYU Depth V2 数据集路径
    # image_dir="/home/ihpc/code/yolo/yolo-obb-depth/dataset/nyudepthv2/nyu_preprocessed/nyu_rgb"
    # label_dir="/home/ihpc/code/yolo/yolo-obb-depth/dataset/nyudepthv2/nyu_preprocessed/dummy_label"
    # depth_dir="/home/ihpc/code/yolo/yolo-obb-depth/dataset/nyudepthv2/nyu_preprocessed/nyu_depth"
    
    # DOTA8 数据集路径
    # image_dir="/home/ihpc/code/yolo/datasets/dota8/images"
    # label_dir="/home/ihpc/code/yolo/datasets/dota8/labels"
    # depth_dir="/home/ihpc/code/yolo/datasets/dota8/depth"
        
    
    # model.train()
    # train(image_dir, label_dir, depth_dir,nc=2, save_name="tomato_leaf.pth")


