import matplotlib.pyplot as plt
from options import opt
from pathlib import Path
import numpy as np


def plt_loss(loss):
    plt.figure(figsize=[8,6])
    plt.plot(loss,linewidth=3.0)
    plt.legend(['Training Loss'],fontsize=18)
    plt.xlabel('Epochs',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)
    fig_path =  Path(opt.save_path).joinpath('figure/loss_curve.jpg')
    plt.savefig(str(fig_path))