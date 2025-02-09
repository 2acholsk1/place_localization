import logging

import torch
from pytorch_metric_learning.distances import BaseDistance
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils.inference import CustomKNN
from torchmetrics import Metric

from place_localization.metrics.precisions_at_x import ModifiedAccuracyCalculator

class MultiMetric(Metric):
    def __init__(self, distance: BaseDistance):
        super().__init__()
        self.count = 0

        knn = CustomKNN(distance, batch_size=256)
        self.calculator = ModifiedAccuracyCalculator(include=(
                                            'mean_average_precision',
                                            'precision_at_1',
                                            'precision_at_5',
                                            'precision_at_10',
                                            'precision_at_25',
                                            'recall_at_1',
                                            'recall_at_5',
                                            'recall_at_10',
                                            'recall_at_25'
                                            ), 
                                             k=25,
                                             device=torch.device('cpu'),
                                             knn_func=knn)
        self.metric_names = self.calculator.get_curr_metrics()

        for metric_name in self.metric_names:
            self.add_state(metric_name, default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx='sum')

        self.add_state('count', default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx='sum')

    def update(self, vectors, labels):
        vectors = vectors.detach().cpu() if vectors.requires_grad else vectors.cpu()
        labels = labels.detach().cpu() if labels.requires_grad else labels.cpu()
        results = self.calculator.get_accuracy(
            query=vectors,
            reference=None,
            query_labels=labels,
            reference_labels=None,
            ref_includes_query=False,
            include=self.metric_names
        )


        for metric_name, metric_value in results.items():
            metric_state = getattr(self, metric_name)
            metric_state += metric_value

        self.count += 1

    def compute(self):
        return {
            metric_name: getattr(self, metric_name) / self.count for metric_name in self.metric_names
        }