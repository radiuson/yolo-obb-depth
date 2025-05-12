import torch
import torch.nn as nn
import yaml
import os
from copy import copy


from ultralytics import YOLO
from ultralytics.models import yolo
from ultralytics.nn.tasks import OBBModel

from ultralytics.models.yolo.obb import OBBTrainer, OBBPredictor, OBBValidator
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import IterableSimpleNamespace, YAML

DEFAULT_CFG_DICT = YAML.load("default.yaml")
DEFAULT_CFG = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
# PyTorch Multi-GPU DDP Constants

# Load the model configuration from a YAML file
# model_dict = yaml_model_load('yolo-obb-depth.yaml')
# model, sorted_savelist = parse_model(model_dict, ch=3, verbose=True)  # model_dict['model'] is a list of dicts
# print(model)  # Print the model architecture
# print(type(model))

# class MyValidator(BaseValidator):
#     def __init__(self, cfg, model, overrides=None, _callbacks=None):
#         super().__init__(cfg, overrides, _callbacks)
#         self.model = model
#         self.shuffle = self.args.shuffle
#     def get_dataloader(self, batch_size = 64, rank = 1,  mode="train" ):
#         return DataLoader(trainer.valset, batch_size=self.batch_size, shuffle=self.shuffle)

# class MyTrainer(BaseTrainer):
#     def __init__(self, cfg,  model, overrides=None, _callbacks=None):
#         super().__init__(cfg, overrides, _callbacks)
#         self.model = model
#         self.shuffle = self.args.shuffle
#     def get_dataloader(self, batch_size = 64, rank = 1,  mode="train" ):
#         return DataLoader(trainer.trainset, batch_size=self.batch_size, shuffle=self.shuffle)
    

class YODTrainer(OBBTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Return OBBModel initialized with specified config and weights.

        Args:
            cfg (str | dict | None): Model configuration. Can be a path to a YAML config file, a dictionary
                containing configuration parameters, or None to use default configuration.
            weights (str | Path | None): Path to pretrained weights file. If None, random initialization is used.
            verbose (bool): Whether to display model information during initialization.

        Returns:
            (OBBModel): Initialized OBBModel with the specified configuration and weights.

        Examples:
            >>> trainer = OBBTrainer()
            >>> model = trainer.get_model(cfg="yolo11n-obb.yaml", weights="yolo11n-obb.pt")
        """
        model = OBBModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return an instance of OBBValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.obb.OBBValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )


model = YOLO('yolo-obb-depth.yaml')  # Load a model

model.train(data="/home/ihpc/code/yolo/yolo-obb-depth/data.yaml")