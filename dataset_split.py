import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import math

# 类别映射
class_name_to_id = {
    "tomato": 0,
    "leaf": 1
}

def robndbox_to_points(cx, cy, w, h, angle):
    # 计算 4 个角点
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    dx = w / 2
    dy = h / 2

    corners = [
        (-dx, -dy),
        ( dx, -dy),
        ( dx,  dy),
        (-dx,  dy)
    ]

    points = []
    for x, y in corners:
        rx = x * cos_a - y * sin_a + cx
        ry = x * sin_a + y * cos_a + cy
        points.append((rx, ry))

    return points  # [(x1,y1), ..., (x4,y4)]

def convert_robndbox(xml_file, output_txt_dir, img_dir):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    filename = root.find('filename').text + '.png'
    img_path = Path(img_dir) / filename

    image = Image.open(img_path)
    img_width, img_height = image.size

    yolo_obb_lines = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in class_name_to_id:
            continue
        class_id = class_name_to_id[class_name]

        robndbox = obj.find('robndbox')
        if robndbox is None:
            continue

        cx = float(robndbox.find('cx').text)
        cy = float(robndbox.find('cy').text)
        w = float(robndbox.find('w').text)
        h = float(robndbox.find('h').text)
        angle = float(robndbox.find('angle').text)

        points = robndbox_to_points(cx, cy, w, h, angle)
        normalized = [(x / img_width, y / img_height) for x, y in points]
        line = f"{class_id} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in normalized)
        yolo_obb_lines.append(line)

    out_file = Path(output_txt_dir) / (Path(xml_file).stem + ".txt")
    with open(out_file, "w") as f:
        f.write("\n".join(yolo_obb_lines))

def split_dataset(
    rgb_dir,
    depth_dir,
    output_dir,
    train_ratio=0.8,
    val_ratio=0.2,
    test_ratio=0.0,
    seed=42
):
    total = train_ratio + val_ratio + test_ratio
    assert abs(total - 1.0) < 1e-5, "train+val+test 必须加起来为 1.0"

    image_files = sorted(f.stem for f in Path(rgb_dir).glob("*.png"))
    print(f"共找到 {len(image_files)} 张图像")

    random.seed(seed)
    random.shuffle(image_files)

    n_total = len(image_files)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val

    splits = {
        "train": image_files[:n_train],
        "val": image_files[n_train:n_train + n_val]
    }
    if test_ratio > 0:
        splits["test"] = image_files[n_train + n_val:]

    for split, names in splits.items():
        rgb_out = Path(output_dir) / "images" / split
        label_out = Path(output_dir) / "labels" / split
        depth_out = Path(output_dir) / "depth" / split
        os.makedirs(rgb_out, exist_ok=True)
        os.makedirs(label_out, exist_ok=True)
        os.makedirs(depth_out, exist_ok=True)

        for name in names:
            shutil.copy(Path(rgb_dir) / f"{name}.png", rgb_out / f"{name}.png")
            xml_file = Path(rgb_dir) / f"{name}.xml"
            if xml_file.exists():
                convert_robndbox(xml_file, label_out, rgb_dir)

            depth_name = f"depth_{name.split('_', 1)[1]}.png"
            depth_file = Path(depth_dir) / depth_name
            if depth_file.exists():
                shutil.copy(depth_file, depth_out / depth_name)

        print(f"✅ {split}: {len(names)} 张")

if __name__ == "__main__":
    split_dataset(
        rgb_dir="./dataset/tomato_leaf/rgb0508_anno",
        depth_dir="./dataset/tomato_leaf/depth0508_anno",
        output_dir="./dataset/tomato_leaf_split",
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42
    )