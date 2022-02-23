import math
import sys


class EarlyStoppingCallback:

    def __init__(self, patience):
        self.best_value = sys.float_info.max
        self.count = 0
        self.patience = patience

    def step(self, current_loss):
        # check whether the current loss is lower than the previous best value.
        # if not count up for how long there was no progress
        if current_loss < self.best_value:
            self.best_value = current_loss
            self.count = 0
        else:
            self.count += 1

    def should_stop(self):
        # check whether the duration of where there was no progress is larger or equal to the patience
        return self.count >= self.patience

