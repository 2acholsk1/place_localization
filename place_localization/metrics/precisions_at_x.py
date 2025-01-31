from pytorch_metric_learning.utils import accuracy_calculator

class ModifiedAccuracyCalculator(accuracy_calculator.AccuracyCalculator):
    def calculate_precision_at_5(self, knn_labels, query_labels, **kwargs):
        return accuracy_calculator.precision_at_k(
            knn_labels,
            query_labels[:, None],
            5,
            self.avg_of_avgs,
            self.return_per_class,
            self.label_comparison_fn
            )

    def calculate_precision_at_10(self, knn_labels, query_labels, **kwargs):
        return accuracy_calculator.precision_at_k(
            knn_labels,
            query_labels[:, None],
            10,
            self.avg_of_avgs,
            self.return_per_class,
            self.label_comparison_fn
            )

    def calculate_precision_at_25(self, knn_labels, query_labels, **kwargs):
        return accuracy_calculator.precision_at_k(
            knn_labels,
            query_labels[:, None],
            25,
            self.avg_of_avgs,
            self.return_per_class,
            self.label_comparison_fn
            )

    def requires_knn(self):
        return super().requires_knn() + ["precision_at_5", "precision_at_10", "precision_at_25"]