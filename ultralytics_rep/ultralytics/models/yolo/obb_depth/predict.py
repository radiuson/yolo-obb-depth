# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.engine.results import DEPTH
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class OBBDPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on an Oriented Bounding Box (OBB) and Depth model.

    This predictor handles oriented bounding box detection tasks and depth estimation, processing images and returning results with rotated
    bounding boxes and depth.

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
        self.args.task = "obbd"

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        results = DEPTH(preds)
        
        


        return results
        # preds = ops.non_max_suppression(
        #     preds,
        #     self.args.conf,
        #     self.args.iou,
        #     agnostic=self.args.agnostic_nms,
        #     max_det=self.args.max_det,
        #     nc=len(self.model.names),
        #     classes=self.args.classes,
        #     rotated=True,
        # )

        # if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
        #     orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        # results = []
        # for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
        #     rboxes = ops.regularize_rboxes(torch.cat([pred[:, :4], pred[:, -1:]], dim=-1))
        #     rboxes[:, :4] = ops.scale_boxes(img.shape[2:], rboxes[:, :4], orig_img.shape, xywh=True)
        #     # xywh, r, conf, cls
        #     obb = torch.cat([rboxes, pred[:, 4:6]], dim=-1)
        #     results.append(Results(orig_img, path=img_path, names=self.model.names, obb=obb))
        # return results
