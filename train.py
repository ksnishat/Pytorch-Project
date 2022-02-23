import torch as t
from data import get_train_dataset, get_validation_dataset
from stopping import EarlyStoppingCallback
from trainer import Trainer
from trainer_test import Trainer_test
from matplotlib import pyplot as plt
import numpy as np
from model import Resnet_test
from model import Resnet

if __name__ == '__main__':
    # ==========HYPER PARAMETERS===========
    BATCH_SIZE = 10
    F1_THRESHOLD = 0.5
    NEPOCHS = 60
    STOP_PATIENCE = 40
    LEARNING_RATE = 0.0002
    # =====================================
    # set up data loading for the training and validation set using t.utils.data.DataLoader and the methods implemented in data.py
    train_set = get_train_dataset()
    val_set = get_validation_dataset()
    # train_dl = t.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE)
    val_dl = t.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE)
    train_dl = train_set
    # val_dl = val_set

    # set up your model
    model_Resnet_test = Resnet_test.ResNet18()
    model_Resnet = Resnet.ResNet18()
    model = model_Resnet

    # set up loss (you can find pre-implemented loss functions in t.nn) use the pos_weight parameter to ease convergence
    # BCE
    w = train_set.pos_weight()
    loss_function = t.nn.BCEWithLogitsLoss(pos_weight=w)

    # set up optimizer (see t.optim);
    # optim = t.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    optim = t.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # initialize the early stopping callback implemented in stopping.py and create a object of type Trainer
    early_stopping_cb = EarlyStoppingCallback(STOP_PATIENCE)

    # go, go, go... call fit on trainer

    cuda = t.cuda.is_available()
    trainer = Trainer(model=model, crit=loss_function, optim=optim, train_dl=train_dl,
                            val_dl=val_dl, cuda=cuda, early_stopping_cb=early_stopping_cb)
    #trainer.restore_checkpoint(40)
    res = trainer.fit(NEPOCHS)
    F1 = trainer.f1_scores
    # acc = trainer.get_accuracy()

    # plot the results
    plt.plot(np.arange(len(res[0])), res[0], label='train loss')
    plt.plot(np.arange(len(res[1])), res[1], label='val loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig('losses.png')

    # save model
    print('===========TRAINING FINISHED==========')
    # print(F1)
    print('F1 max at: ', np.argmax(F1), '--- val loss =', res[1][np.argmax(F1)])
    print('------')
    # print(res[1])
    print('val loss min at: ', np.argmin(res[1]), '--- F1 = ', F1[np.argmin(res[1])])
    earlystop, checkpoint = trainer.get_last_checkpoint()
    print('====== enter checkpoint to save model: ======')
    #checkpoint = 32
    checkpoint = input()
    trainer.restore_checkpoint(checkpoint)
    trainer.save_onnx('checkpoint_{:03d}.onnx'.format(checkpoint))
    if earlystop:
        print('====STOPPED EARLY at: ', checkpoint, '======')
    else:
        print('FULL EPOCH')

    # 19:12 5/02/2020
