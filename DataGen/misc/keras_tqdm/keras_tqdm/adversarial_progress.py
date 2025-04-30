from sys import stderr

import numpy as np
import six
from tqdm import tqdm_notebook
from tensorflow.keras.callbacks import Callback

class AdversarialProgress(Callback):
    def __init__(self,loops,gen_total, disc_total):
        self.loops = loops
        self.gen_total = gen_total
        self.disc_total = disc_total        
        self.update="{phase} Epoch: {epoch} - {metrics}"

    def create(self, description, total, leave, file=stderr, initial=0):
        return tqdm_notebook(desc=description, total=total, leave=leave, file=stderr,initial=0)

    def createAll(self):
        self.createOuter()
        self.createGen()
        self.createDisc()

    def createOuter(self):
        self.outer = self.create("Training",self.loops, leave=False)

    def createGen(self):
        self.gen = self.create("Generator Epoch: 0",self.gen_total, True)

    def createDisc(self):
        self.disc = self.create("Discriminator Epoch: 0",self.disc_total, True)

    def null1(self,logs={}):
        pass

    def null2(self,epoch, logs={}):
        pass

    def doneGenTrain(self,logs={}):
        self.gen.update(-1 * self.gen.n)
        self.gen.desc = "Finished Generator Loop"

    def doneDiscTrain(self,logs={}):
        self.disc.update(-1 * self.disc.n)
        self.disc.desc = "Finished Discriminator Loop"
        self.outer.update(1)

    def gen_epoch_end(self, epoch, logs={}):
        metrics = self.format_metrics(logs)
        desc = self.update.format(phase="Generator",epoch=epoch, metrics=metrics)
        self.gen.desc = desc
        self.gen.update(1)

    def disc_epoch_end(self, epoch, logs={}):
        metrics = self.format_metrics(logs)
        desc = self.update.format(phase="Discriminator",epoch=epoch, metrics=metrics)
        self.disc.desc = desc
        self.disc.update(1)

    def format_metrics(self, logs):
        s = []
        for m,v in logs.items():
            s.append("{}:{:.3f}".format(m,v))
        return " - ".join(s)