import torch

ACCURACY = "acc"
LOSS = "loss"
AUC = "auc"

class CallBack:
    def __init__(self, save_the_best = True):
        super(CallBack, self, )

    def run_call_backs(self,):
        pass

class SaveTheBestCallBack(CallBack):

    def __init__(self, model, target_file, less_is_better = False):
        super(SaveTheBestCallBack, self, )
        self.less_is_better = less_is_better
        self.best_existing_criterion = None
        self.target_file = target_file
        self.model = model

    def save_model(self):
        torch.save(self.model.state_dict(), self.target_file)

    def save_the_best_model(self, current_criterion):

        if self.best_existing_criterion is None:
            self.best_existing_criterion = current_criterion
        elif self.less_is_better:
            if current_criterion < self.best_existing_criterion:
                self.save_model()
        else:
            if current_criterion > self.best_existing_criterion:
                self.save_model()

    def run(self, current_criterion):
        self.save_the_best_model(current_criterion=current_criterion)