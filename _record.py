import torch, time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import count_improvement
import logging

class _PerformanceMeter(object):
    """
    Class for recording and displaying performance metrics in multi-task learning.

    Parameters:
    - task_dict: A dictionary containing the configuration for each task, such as weights, loss functions, and metric functions.
    - multi_input: Indicates whether to use multiple inputs.
    - base_result: Initial baseline result used for calculating improvements.
    """

    def __init__(self, task_dict, multi_input, base_result=None):
        self.task_dict = task_dict
        self.multi_input = multi_input
        self.task_num = len(self.task_dict)
        self.task_name = list(self.task_dict.keys())

        # Initialize task weights, best results, and improvements
        self.weight = {task: self.task_dict[task]['weight'] for task in self.task_name}
        self.base_result = base_result
        self.best_result = {'improvement': -1e+2, 'epoch': 0, 'result': 0}
        self.improvement = None

        # Initialize task loss functions and metric functions
        self.losses = {task: self.task_dict[task]['loss_fn'] for task in self.task_name}
        self.metrics = {task: self.task_dict[task]['metrics_fn'] for task in self.task_name}

        # Initialize task result records
        self.results = {task:[] for task in self.task_name}
        self.loss_item = np.zeros(self.task_num)

        self.has_val = False

        self._init_display()

    def record_time(self, mode='begin'):
        """
        Record the start or end time.

        Parameters:
        - mode: Time recording mode, can be 'begin' (start) or 'end' (end).
        """
        if mode == 'begin':
            self.beg_time = time.time()
        elif mode == 'end':
            self.end_time = time.time()
        else:
            raise ValueError('No support time mode {}'.format(mode))

    def update(self, preds, gts, task_name=None):
        """
        Update metrics using predictions and ground truth labels.

        Parameters:
        - preds: Dictionary of prediction results, with task names as keys and predictions as values.
        - gts: Dictionary of ground truth labels, with task names as keys and ground truth as values.
        - task_name: Optional parameter to specify a particular task to update.
        """
        with torch.no_grad():
            # if task_name is not None:
            #         self.metrics[task_name].update_fun(preds, gts)
            # else:
            #     self.metrics.update_fun(preds, gts)
            if task_name is None:
                for tn, task in enumerate(self.task_name):
                    self.metrics[task].update_fun(preds[task], gts[task])
            else:
                self.metrics[task_name].update_fun(preds, gts)

        # self.metrics[task_name].update_fun(preds, gts)

    def get_score(self):
        """
        Calculate loss and metric scores for each task.
        """
        with torch.no_grad():
            for tn, task in enumerate(self.task_name):
                self.results[task] = self.metrics[task].score_fun()
                self.loss_item[tn] = self.losses[task]._average_loss()

    def _init_display(self):
        """
        Initialize display format.
        """
        logging.info('=' * 100)
        header = f"{'LOG FORMAT':<15}|"
        for task in self.task_name:
            header += f" {task + '_LOSS':<10}"
            for m in self.task_dict[task]['metrics']:
                header += f"{m:<10}"
            header += " |"
        header += f" {'TIME':<10}"
        logging.info(header)
        logging.info('=' * 100)

    def display(self, mode, epoch):
        """
        Display performance metrics for training, validation, or testing.

        Parameters:
        - mode: Display mode, can be 'train', 'val', or 'test'.
        - epoch: Current epoch number.
        """
        if epoch is not None:
            # Update best result
            if epoch == 0 and self.base_result is None and mode == ('val' if self.has_val else 'test'):
                self.base_result = self.results
            if mode == 'train':
                logging.info(f"Epoch: {epoch}")

            if not self.has_val and mode == 'test':
                self._update_best_result(self.results, epoch)
            if self.has_val and mode != 'train':
                self._update_best_result_by_val(self.results, epoch, mode)

        mode_dict = {'train': 'TRAIN', 'val': 'VAL', 'test': 'TEST'}
        p_mode = mode_dict[mode]

        # Find the longest mode name length
        max_mode_len = max(len(m) for m in mode_dict.values())

        # Find the longest task name length
        max_task_name_len = max(len(task) for task in self.task_name)

        for tn, task in enumerate(self.task_name):
            # Use string formatting to align mode names, task names, and vertical bars
            output = f"{p_mode:<{max_mode_len}}_{task:<{max_task_name_len}} | "
            output += f"{self.loss_item[tn]:<10.4f}"
            for result in self.results[task]:
                output += f"{result:<10.4f}"
            output += " |"
            logging.info(output)

        time_taken = self.end_time - self.beg_time
        logging.info(f"Time taken: {time_taken:.2f}s")

        if mode == 'test':
            logging.info('-' * 100)

    def display_best_result(self):
        """
        Display the best result.
        """
        logging.info('='*40)
        logging.info('Best Result: Epoch {}, result {}'.format(self.best_result['epoch'], self.best_result['result']))
        logging.info('='*40)

    def _update_best_result_by_val(self, new_result, epoch, mode):
        """
        Update best result based on validation set results.

        Parameters:
        - new_result: New result.
        - epoch: Current epoch number.
        - mode: Current mode, can be 'train', 'val', or 'test'.
        """
        if mode == 'val':
            improvement = count_improvement(self.base_result, new_result, self.weight)
            self.improvement = improvement
            if improvement > self.best_result['improvement']:
                self.best_result['improvement'] = improvement
                self.best_result['epoch'] = epoch
        else:
            if epoch == self.best_result['epoch']:
                self.best_result['result'] = new_result

    def _update_best_result(self, new_result, epoch):
        """
        Update the best result.

        Parameters:
        - new_result: New result.
        - epoch: Current epoch number.
        """
        improvement = count_improvement(self.base_result, new_result, self.weight)
        self.improvement = improvement
        if improvement > self.best_result['improvement']:
            self.best_result['improvement'] = improvement
            self.best_result['epoch'] = epoch
            self.best_result['result'] = new_result

    def reinit(self):
        """
        Reinitialize loss functions and metric functions.
        """
        for task in self.task_name:
            self.losses[task]._reinit()
            self.metrics[task].reinit()
        self.loss_item = np.zeros(self.task_num)
        self.results = {task:[] for task in self.task_name}