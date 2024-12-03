from torchmetrics.functional import accuracy, label_ranking_loss
from torchmetrics.functional import auroc, f1_score, specificity, precision, confusion_matrix
import torch
from torchmetrics import Recall, Precision, AUROC, Specificity, ConfusionMatrix, Accuracy, F1Score
#from sklearn.metrics import f1_score, auc, precision_score, recall_score, accuracy_score,


class PGU_Metrics:
    def __init__(self, num_classes = 2, threshold = 0.5, positive_class = 1):
        self.num_classes = num_classes
        self.threshold = threshold
        self.positive_class = positive_class
        self.metrics_list = {
        'accuracy': Accuracy(threshold=self.threshold, num_classes=self.num_classes),
        'auroc': AUROC(num_classes=self.num_classes,pos_label=self.positive_class),
        'f1_score': F1Score(num_classes=self.num_classes, threshold=self.threshold,average="macro"),
        'specificity':Specificity(num_classes=self.num_classes, threshold=self.threshold, average="macro"),
        'precision':Precision(num_classes=self.num_classes, threshold=self.threshold, average="macro"),
        'confusion_matrix':ConfusionMatrix(num_classes=self.num_classes, threshold=self.threshold),
    }


    def calculate_metrics(self,added_metrics, data_type, real_value=None, predicted_value=None):
        metrics = ['accuracy']
        if len(added_metrics) > 0:
            metrics.extend(added_metrics)

        num_classes = len(set(real_value))
        real = torch.tensor(real_value)
        predict = torch.tensor(predicted_value)
        results = {}
        for meter in metrics:
            if meter in self.metrics_list.keys():
                fun = self.metrics_list[meter]
                #print(f'fun = {fun} and meter = {meter}')
                res = fun(preds=predict, target=real)
                if data_type == 'train':
                    results['train_' + meter] = res
                elif data_type == 'validation':
                    results['val_' + meter] = res
                else:
                    results['test_' + meter] = res
        return results
