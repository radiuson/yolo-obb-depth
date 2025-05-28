# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.engine.results import OBBDepth, Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops

class OBBDPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on an Oriented Bounding Box (OBB) and Depth model.

    This predictor handles oriented bounding box detection tasks and depth estimation.

    Attributes:
        args (namespace): Configuration arguments for the predictor.
        model (torch.nn.Module): The loaded YOLO OBBD model.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.obb_depth import OBBDPredictor
        >>> args = dict(model="yolo11n-obbd.pt", source=ASSETS)
        >>> predictor = OBBDPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes OBBDPredictor with optional model and data configuration overrides."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "obb-depth"
    def postprocess(self, preds, img, orig_imgs):
        """
        Post-process predictions into Results objects with rotated boxes and single-scale depth.

        Args:
            preds (tuple): A tuple containing (preds, raw_x, depth)
            img (torch.Tensor): Preprocessed input image (B, 3, H, W)
            orig_imgs (List[np.ndarray]): List of original images

        Returns:
            List[Results]: Processed result per image
        """
        preds, raw_x, depth = preds

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):  # B 批次中的每一张图片
            orig_img = orig_imgs[i]
            img_shape = img.shape[2:]
            orig_shape = orig_img.shape[:2]

            # 1. 分离预测
            xywh = pred[:, :4]
            conf = pred[:, 4:5]
            cls = pred[:, 5:6]
            angle = pred[:, 6:7]

            # 2. 缩放 xywh，组合 rboxes
            xywh = ops.scale_boxes(img_shape, xywh, orig_shape, xywh=True)
            rboxes = ops.regularize_rboxes(torch.cat([xywh, angle], dim=-1))  # [x, y, w, h, θ]
            obb = torch.cat([rboxes, conf, cls], dim=-1)  # [x, y, w, h, θ, conf, cls]

            # 3. 深度图
            depth_i = depth[i].squeeze(0)  # (H, W)

            # 4. 包装成 Results
            results.append(Results(
                orig_img,
                path=self.batch[0][i],
                names=self.model.names,
                obb=obb,
                obb_depth=OBBDepth(obb, orig_shape=orig_shape, angle=angle, depth=[depth_i])
            ))

        return results
