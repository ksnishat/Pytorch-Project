import torch as t
from sklearn.metrics import f1_score
import numpy as np


class Trainer:
    def __init__(self,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimiser
                 train_dl=None,  # Training data set
                 val_dl=None,  # Validation (or test) data set
                 cuda=True,  # Whether to use the GPU
                 early_stopping_cb=None,
                 PREDICTION_THRESHOLD=0.5):  # The stopping criterion.
        self.accuracy_class0 = []
        self.accuracy_class1 = []
        self.accuracy = []
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_dl = val_dl
        self._cuda = cuda
        self._early_stopping_cb = early_stopping_cb
        self.pred_threshold = PREDICTION_THRESHOLD

        self._checkpoint = 10**50
        self.early_stop = False

        # transfer the batch to the gpu if given
        if self._cuda:
            self._model.cuda()
            self._crit.cuda()
            self.device = t.device('cuda:0')

        self.f1_scores = []

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenghth axes
                                    'output': {0: 'batch_size'}})

    def train_step(self, _inputs, _labels):
        # TODO: perform following steps:
        # -reset the gradients
        self._optim.zero_grad()

        # -propagate through the network
        outputs = self._model(_inputs)

        # -calculate the loss
        loss = self._crit(outputs, _labels)

        # -compute gradient by backward propagation
        loss.backward()

        # -update weights
        self._optim.step()

        # -return the loss
        return loss.item()

    def train_epoch(self):
        # set training mode
        self.mode = 'train'
        running_loss = []
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        dataloader = t.utils.data.DataLoader(self._train_dl, batch_size=32,
                                             shuffle=True, num_workers=1)
        # iterate through the training set
        for data in dataloader:
            # perform a training step
            _inputs, _labels = data
            if self._cuda:
                _inputs = _inputs.to(self.device)
                _labels = _labels.to(self.device)
            loss = self.train_step(_inputs, _labels)
            running_loss.append(loss)
            # print(loss)

        # calculate the average loss for the epoch and return it
        return np.mean(running_loss)

    def val_test_step(self, _inputs, _labels):
        # predict
        # propagate through the network and calculate the loss and predictions
        predictions = self._model(_inputs)
        loss = self._crit(predictions, _labels)
        # return the loss and the predictions
        return loss.item(), t.nn.Sigmoid()(predictions)

    def val_test(self):
        # set eval mode
        self.mode = 'test'
        running_loss = []
        loss_list = []
        predictions_list = np.ndarray((1, 2))
        labels_list = np.ndarray((1, 2))
        # disable gradient computation
        with t.no_grad():
            for data in self._val_dl:  # iterate through the validation set
                _inputs, _labels = data
                if self._cuda:
                    _inputs = _inputs.to(self.device)
                    _labels = _labels.to(self.device)

                loss, predictions = self.val_test_step(_inputs, _labels)  # perform a validation step
                predictions = (predictions > self.pred_threshold).int()
                # save the predictions and the labels for each batch
                labels_list = np.vstack((labels_list, _labels.cpu().numpy()))
                predictions_list = np.vstack((predictions_list, predictions.cpu().numpy()))

                loss_list.append(loss)
                # You might want to calculate these metrics in designated functions
                running_loss.append(loss)

        # calculate the average loss and average metrics of your choice.
        predictions_list = np.delete(predictions_list, 0, axis=0)
        labels_list = np.delete(labels_list, 0, axis=0)
        f1 = f1_score(y_true=labels_list, y_pred=predictions_list, average='macro')
        self.f1_scores.append(f1)
        # print("$$$$$$$$$$ F1: %.3f" %f1, '$$$$$$$$$$')
        # TODO: return the loss and print the calculated metrics
        return np.mean(running_loss)

    def fit(self, epochs=-1):
        assert self._early_stopping_cb is not None or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch
        # TODO
        loss_train_list = []
        loss_val_list = []
        stop_at = epochs
        for e in range(epochs):
            print('=====epoch ', e,' starts=====')
            # stop by epoch number
            # train for a epoch and then calculate the loss and metrics on the validation set
            l_train = self.train_epoch()
            l_val = self.val_test()

            # append the losses to the respective lists
            loss_train_list.append(l_train)
            loss_val_list.append(l_val)

            # use the save_checkpoint function to save the model for each epoch
            self.save_checkpoint(e)

            print('training_loss= ', loss_train_list)
            print('val_loss= ', loss_val_list)
            print('F1= ', self.f1_scores)
            print('F1 max: ', np.max(self.f1_scores))
            # print('====epoch ', e, ' finished===')

            # check whether early stopping should be performed using the early stopping callback and stop if so
            if self.should_save_checkpoint(e, loss_val_list):
                self.save_checkpoint(e)
            self._early_stopping_cb.step(l_val)
            if self._early_stopping_cb.should_stop():
                self.early_stop, stop_at = True, e
                break

        self._checkpoint = min(stop_at, epochs)
        self.save_checkpoint(stop_at)
        # return the loss lists for both training and validation
        return loss_train_list, loss_val_list
        # TODO

    def get_last_checkpoint(self):
        return self. early_stop, self._checkpoint

    def get_accuracy(self):
        return self.accuracy_class0, self.accuracy_class1

    def should_save_checkpoint(self, e, _list):

        if (self.f1_scores[-1] == np.max(self.f1_scores)):
            return True

        if (_list[-1] == np.min(_list)):
            return True

        return False