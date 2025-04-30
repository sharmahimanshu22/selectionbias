from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import numpy as np
import pdb


def sortedplot(*args, **kwargs):
    if "ax" in kwargs:
        ax = kwargs.pop('ax')
    else:
        ax = plt
    if len(args) == 1:
        ax.plot(*args, **kwargs)
    else:
        #pdb.set_trace()
        i = 0
        newargs = list()
        ix = []
        for x in args:
            if np.remainder(i, 3) == 2:
                newargs.append(x)
            else:
                if np.remainder(i, 3) == 0:
                    #ix = np.argsort(x[:, 0], axis=0)
                    ix = np.argsort(x, axis=0).flatten()
                #newargs.append(x[ix, :])
                newargs.append(x[ix])
            i = i + 1
        # print(newargs[0].shape)
        ax.plot(*newargs, **kwargs)


def mysorted(*args, **kwargs):
    x = args[0]
    ix = np.argsort(x.flatten())
    if "ax" in kwargs:
        ax = kwargs["ax"]
    else:
        ax = plt
    for n,v in kwargs.items():
        if n != "ax":
            vs = v[ix]
            pdb.set_trace()
            ax.plot(x[ix],vs,label=n)
    ax.legend()


