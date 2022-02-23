import torch as t
import numpy as np
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt


class Trainer_test:
    f1_scores = []

    def __init__(self,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimiser
                 train_dl=None,  # Training data set
                 val_dl=None,  # Validation (or test) data set
                 cuda=True,  # Whether to use the GPU
                 early_stopping_cb=None,
                 PREDICTION_THRESHOLD=0.5,
                 save_location="checkpoints",
                 name="run"):  # The stopping criterion.
        self.device = None
        self._model = model
        self._name = name
        self._save_location = save_location
        self._f1_threshold = PREDICTION_THRESHOLD
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_dl = val_dl
        self._cuda = cuda
        self._early_stopping_cb = early_stopping_cb

        if cuda:
            self._model.cuda()
            self._crit.cuda()
            self.device = t.device('cuda:0')

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

    def val_test_step(self, x, y):
        # predict # propagate through the network and calculate the loss and predictions # return the loss and the predictions with t.no_grad():
            outputs =self._model(x)
            loss =self._crit(outputs, y)

            return loss.item(), t.nn.Sigmoid()(outputs)

    # def train_step(self, x, y):
    #     # perform following steps: # -reset the gradients # -propagate through the network # -calculate the loss # -compute gradient by backward propagation # -update weights # -return the loss
    #
    #     self._optim.zero_grad()
    #     outputs = self._model(x)
    #     loss = self._crit(outputs, y)
    #     loss.backward()
    #     self._optim.step()
    #
    #     return loss.item()

    # def train_epoch(self):
    #     # set training mode # iterate through the training set # transfer the batch to "cuda()" -> the gpu if a gpu is given # perform a training step # calculate the average loss for the epoch and return it
    #     running_loss = []
    #     self._model.mode ="train"
    #     dataloader = self._train_dl
    #     for data in dataloader:
    #         images, labels = data
    #         # self.imshow(tv.utils.make_grid(images))
    #         if self._cuda:
    #             cuda = t.device("cuda:0")
    #             images = images.to(cuda)
    #             labels = labels.to(cuda)
    #         loss =self.train_step(images, labels)
    #         running_loss.append(loss)
    #         # print('[%d] Training loss: %.5f' % (i, loss))
    #     return np.mean(running_loss)

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
        dataloader = t.utils.data.DataLoader(self._train_dl, batch_size=32,
                                             shuffle=True, num_workers=1)
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # iterate through the training set
        for data in dataloader:
            # perform a training step
            _inputs, _labels = data
            if self._cuda:
                cuda = t.device("cuda:0")
                _inputs = _inputs.to(cuda)
                _labels = _labels.to(cuda)
            loss = self.train_step(_inputs, _labels)
            running_loss.append(loss)
            # print(loss)

        # calculate the average loss for the epoch and return it
        return np.mean(running_loss)


    def val_test(self):
        running_loss = []
        labels_list = []
        predictions_list = []
        labels_array =None
        predictions_array =None
        dataloader = self._val_dl
        with t.no_grad():
            for data in dataloader:
                images, labels = data
                if self._cuda:
                    cuda = t.device("cuda:0")
                    images = images.to(cuda)
                    labels = labels.to(cuda)

                loss, predictions = self.val_test_step(images, labels)
                predictions = (predictions > self._f1_threshold).int()
                if labels_array is None:
                    labels_array = labels.cpu().numpy()
                    predictions_array = predictions.cpu().numpy()
                else:
                    labels_array = np.vstack((labels_array, labels.cpu().numpy()))
                    predictions_array = np.vstack((predictions_array, predictions.cpu().numpy()))

                labels_list.append(labels.cpu().numpy())
                predictions_list.append(predictions.cpu().numpy())
                running_loss.append(loss)

        f = f1_score(labels_array, predictions_array, average="macro")
        print("Actual F1: %.3f" % f)
        self.f1_scores.append(f)
        return np.mean(running_loss)

    def fit(self, epochs=-1):
        assert self._early_stopping_cb is not None or epochs >0
        epoch =0
        train_loss = []
        val_loss = []
        while True:
            if epochs != -1 and epoch >= epochs:
                break
            print('Epoch: [%d]------------------------' % epoch)
            t_loss = self.train_epoch()
            v_loss = self.val_test()
            train_loss.append(t_loss)
            val_loss.append(v_loss)

            # self.save_checkpoint(epoch)

            if self._early_stopping_cb is not None:
                self._early_stopping_cb.step(v_loss)
                if self._early_stopping_cb.should_stop():
                    break
            epoch +=1
            print('training loss: ', train_loss)
            print('val loss: ', val_loss)

        loc = self._save_location +"/" + self._name +".onnx"
        self.save_onnx(loc)


        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.arange(len(self.f1_scores)), self.f1_scores, label='F1 Scores')
        ax.legend()
        loc = self._save_location +'/' +'metrics-{}.png'.format(self._name)
        fig.savefig(loc)

        return train_loss, val_loss

    def imshow(self, img):
        img = img /2 +0.5 # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()