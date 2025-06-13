import torch, time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score,\
    classification_report,roc_auc_score, confusion_matrix

from sklearn.preprocessing import label_binarize




class AbsMetric(object):
    r"""An abstract class for the performance metrics of a task.

    Attributes:
        record (list): A list of the metric scores in every iteration.
        bs (list): A list of the number of data in every iteration.
    """

    def __init__(self):
        self.record = []
        self.bs = []

    @property
    def update_fun(self, pred, gt):
        r"""Calculate the metric scores in every iteration and update :attr:`record`.

        Args:
            pred (torch.Tensor): The prediction tensor.
            gt (torch.Tensor): The ground-truth tensor.
        """
        pass

    @property
    def score_fun(self):
        r"""Calculate the final score (when an epoch ends).

        Return:
            list: A list of metric scores.
        """
        pass

    def reinit(self):
        r"""Reset :attr:`record` and :attr:`bs` (when an epoch ends).
        """
        self.record = []
        self.bs = []


# accuracy
class AccMetric(AbsMetric):
    r"""Calculate the accuracy.
    """

    def __init__(self):
        super(AccMetric, self).__init__()

    def update_fun(self, pred, gt):
        r"""
        """
        pred = F.softmax(pred, dim=-1).max(-1)[1]
        self.record.append(gt.eq(pred).sum().item())
        self.bs.append(pred.size()[0])

    def score_fun(self):
        r"""
        """
        return [(sum(self.record) / sum(self.bs))]


# L1 Error
class L1Metric(AbsMetric):
    r"""Calculate the Mean Absolute Error (MAE).
    """

    def __init__(self):
        super(L1Metric, self).__init__()

    def update_fun(self, pred, gt):
        r"""
        """
        abs_err = torch.abs(pred - gt)
        self.record.append(abs_err.item())
        self.bs.append(pred.size()[0])

    def score_fun(self):
        r"""
        """
        records = np.array(self.record)
        batch_size = np.array(self.bs)
        return [(records * batch_size).sum() / (sum(batch_size))]


class CombinedMetric(AbsMetric):
    def __init__(self, metrics):
        super(CombinedMetric, self).__init__()
        self.metrics = metrics

    def update_fun(self, pred, gt):
        for metric in self.metrics:
            metric.update_fun(pred, gt)

    def score_fun(self):
        scores = []
        for metric in self.metrics:
            scores.extend(metric.score_fun())
        return scores

    def reinit(self):
        for metric in self.metrics:
            metric.reinit()

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        for metric in self.metrics:
            if hasattr(metric, attr):
                return getattr(metric, attr)
        raise AttributeError(f"'CombinedMetric' object has no attribute '{attr}'")




class PRSMetric(AbsMetric):
    def __init__(self, task_type, average='macro'):
        super(PRSMetric, self).__init__()
        self.task_type = task_type
        self.average = average
        self.pred_labels = []
        self.true_labels = []

    def update_fun(self, pred, gt):
        pred_label = torch.argmax(pred, dim=1)
        self.pred_labels.append(pred_label)
        self.true_labels.append(gt)

    def score_fun(self):
        pred_labels = torch.cat(self.pred_labels).cpu().numpy()
        true_labels = torch.cat(self.true_labels).cpu().numpy()

        #print("Predicted labels:", pred_labels)
        #print("True labels:", true_labels)

        report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)

        precision = report['macro avg']['precision']
        recall = report['macro avg']['recall']
        f1 = report['macro avg']['f1-score']

        cm = confusion_matrix(true_labels, pred_labels)

        if cm.shape[0] == 2:  # Binary classification problem
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp)
        else:  # Multi-class classification problem
            specificities = []
            for i in range(len(cm)):
                tn = np.delete(cm, i, axis=0).sum() - cm[i, :i].sum() - cm[i, i + 1:].sum()
                fp = cm[:i, i].sum() + cm[i + 1:, i].sum()
                specificity = tn / (tn + fp) if tn + fp > 0 else 0
                specificities.append(specificity)
            specificity = np.mean(specificities)

        return [precision, recall, specificity, f1]

    def reinit(self):
        self.pred_labels = []
        self.true_labels = []
###############

class AUCMetric(AbsMetric):
    def __init__(self, task_type):
        super(AUCMetric, self).__init__()
        self.task_type = task_type
        self.pred_probs = []
        self.true_labels = []

    def update_fun(self, pred, gt):
        if self.task_type == 'binary':
            pred_prob = torch.softmax(pred, dim=1)[:, 1]
        else:
            pred_prob = torch.softmax(pred, dim=1)
        self.pred_probs.append(pred_prob)
        self.true_labels.append(gt)

    def score_fun(self):
        pred_probs = torch.cat(self.pred_probs)
        true_labels = torch.cat(self.true_labels)

        if self.task_type == 'binary':
            if len(torch.unique(true_labels)) == 1:
                return -1  # Return -1 to indicate AUC cannot be calculated
            auc = roc_auc_score(true_labels.cpu().numpy(), pred_probs.cpu().numpy())
        else:
            if len(torch.unique(true_labels)) == 1:
                return -1  # Return -1 to indicate AUC cannot be calculated
            auc = roc_auc_score(true_labels.cpu().numpy(), pred_probs.cpu().numpy(), multi_class='ovr')

        return auc

    def reinit(self):
        self.pred_probs = []
        self.true_labels = []


class AccAUCPRSMetric(AbsMetric):
    def __init__(self, task_type, average='macro'):
        super(AccAUCPRSMetric, self).__init__()
        self.task_type = task_type
        self.acc_metric = AccMetric()
        self.auc_metric = AUCMetric(task_type)
        self.prs_metric = PRSMetric(task_type, average)

    def update_fun(self, pred, gt):
        self.acc_metric.update_fun(pred, gt)
        self.auc_metric.update_fun(pred, gt)
        self.prs_metric.update_fun(pred, gt)

    def score_fun(self):
        acc_score = self.acc_metric.score_fun()
        auc_score = self.auc_metric.score_fun()
        prs_score = self.prs_metric.score_fun()

        if auc_score == -1:
            auc_score = [-1]  # Convert -1 to list format [-1]
        else:
            auc_score = [auc_score]  # Convert AUC score to list format

        return acc_score + auc_score + prs_score

    def reinit(self):
        self.acc_metric.reinit()
        self.auc_metric.reinit()
        self.prs_metric.reinit()