import csv
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from amp.config import KL_ANNEALRATE, MAX_KL, MAX_LENGTH, MAX_TEMPERATURE, MIN_KL, MIN_TEMPERATURE, TAU_ANNEALRATE
from amp.data_utils import sequence
from amp.models.model import Model
from amp.utils.basic_model_serializer import BasicModelSerializer
from keras import backend as K
from keras.callbacks import Callback


class VAECallback(Callback):
    def __init__(
            self,
            encoder,
            decoder,
            amp_classifier,
            mic_classifier,
            output_layer,
            kl_annealrate: float = KL_ANNEALRATE,
            max_kl: float = MAX_KL,
            kl_weight=K.variable(MIN_KL, name="kl_weight"),
            tau=K.variable(MAX_TEMPERATURE, name="temperature "),
            tau_annealrate=TAU_ANNEALRATE,
            min_tau=MIN_TEMPERATURE,
            max_length: int = MAX_LENGTH,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.output_layer = output_layer
        self.kl_annealrate = kl_annealrate
        self.max_kl = max_kl
        self.kl_weight = kl_weight
        self.tau = tau
        self.tau_annealrate = tau_annealrate
        self.min_tau = min_tau

    def on_epoch_end(self, epoch, logs={}):
        new_kl = np.min([K.get_value(self.kl_weight) * np.exp(self.kl_annealrate * epoch), self.max_kl])
        K.set_value(self.kl_weight, new_kl)

        new_tau = np.max([K.get_value(self.tau) * np.exp(- self.tau_annealrate * epoch), self.min_tau])
        K.set_value(self.tau, new_tau)

        print("Current KL weight is " + str(K.get_value(self.kl_weight)))
        print("Current temperature is " + str(K.get_value(self.tau)))


class SaveModelCallback(Callback):

    def __init__(self, model: Model, model_save_path, name):
        """
        model : amp.models.Model instance (that corresponds to trained keras.Model instance)
        model_save_path: location for model root directory
        name: name of the model (experiment), this is also the name of model root directory
        """
        super().__init__()
        self.model_st = model
        self.model_save_path = model_save_path
        self.name = name
        self.root_dir = os.path.join(model_save_path, name)
        self.serializer = BasicModelSerializer()
        os.path.join(model_save_path, self.name)
        Path(self.root_dir).mkdir(parents=True, exist_ok=True)

        self.metrics_file_path = os.path.join(self.root_dir, "metrics.csv")
        self.metrics_initialized = False
        self.metric_order = None

    def _initialize_metrics_doc(self, metrics_names: List[str]):
        col_names = ["epoch_no"] + metrics_names
        with open(self.metrics_file_path, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(col_names)
        self.metric_order = metrics_names
        self.metrics_initialized = True

    def _save_metrics(self, logs: Dict[str, Any], epoch_no: int):
        unboxed_metrics = [logs[m_name] for m_name in self.metric_order]
        row = [epoch_no] + unboxed_metrics
        with open(self.metrics_file_path, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def on_epoch_end(self, epoch, logs=None):
        if not self.metrics_initialized:
            self._initialize_metrics_doc(list(logs.keys()))
        save_path = os.path.join(self.model_save_path, self.name, str(epoch))
        self.serializer.save_model(self.model_st, save_path)
        self._save_metrics(logs, epoch)
