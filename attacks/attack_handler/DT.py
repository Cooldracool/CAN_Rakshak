from evaluate import evaluation_metrics
from ..DT_RF_attack.DT_attack import DT_attack
from .base import *

import os
from datetime import datetime


class DT(Attack):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def apply(self):

        cfg = self.cfg
        attack_name = cfg["adv_attack"].lower()

        print(f"Selected adversarial attack: {attack_name}")

        surrogate_model_path = (
            os.path.join(cfg["dir_path"], "..", "models", cfg["surrogate_model"])
            if cfg["surrogate_model"] else None
        )

        target_model_path = (
            os.path.join(cfg["dir_path"], "..", "models", cfg["target_model"])
            if cfg["adv_attack_type"] == "blackbox"
            else surrogate_model_path
        )

        print("Making call to the attack :", attack_name)

        preds, labels, output_path = DT_attack(
            surrogate_model_path,
            target_model_path,
            cfg
        )

        evaluation_metrics(preds, labels, cfg)

        return output_path