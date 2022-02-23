import torch as t
from trainer import Trainer
import sys
import torchvision as tv
from model import Alexnet
from model import Resnet

epoch = int(sys.argv[1])
#TODO: Enter your model here
model = Alexnet.AlexNet()

crit = t.nn.BCEWithLogitsLoss()
trainer = Trainer(model, crit)
trainer.restore_checkpoint(epoch)
trainer.save_onnx('checkpoint_{:03d}.onnx'.format(epoch))
