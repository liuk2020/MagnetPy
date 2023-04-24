import mpy
import numpy as np
import matplotlib.pyplot as plt
from ._plotPoincare import plotCase
from ..vmec2spec.misc import mu0
from ..misc.print import print_progress
from typing import List


class SuccessCase:

    def __init__(self, dataList: List[mpy.SPECOut], nameList: List[str]) -> None:
        assert len(dataList) == len(nameList)
        self.data = dataList
        self.name = nameList
        self.len = len(dataList)

    @classmethod
    def readSuccessTxt(cls, txtFile: str):
        dataList = list()
        nameList = list()
        with open(txtFile, 'r') as f:
            for file in f:
                file = file.strip("\n")
                file += ".h5"
                try:
                    dataList.append(mpy.SPECOut(file))
                    nameList.append(file)
                except:
                    print("Cannot open file: " + file + "! ")
        return cls(dataList, nameList)

    def plotPoincare(self):
        nFigs = self.len//6 + 1
        for i in range(nFigs-1):
            fig, ax = plt.subplots(2, 3, figsize=(15,10))
            for j in range(6):
                _ax = ax[j//3, j%3]
                index = i*6 + j
                self.data[index].plot_kam_surface(ns=[1], ax=_ax)
                plotCase(self.data[index], ax=_ax)
                _ax.set_title(self.name[index])
                _ax.set_xlim(xmin=1.0, xmax=1.8)
                _ax.set_ylim(ymin=-0.4, ymax=0.4)
            fig.tight_layout()
            fig.savefig("poincare"+str(i)+".pdf")
        fig, ax = plt.subplots(2, 3, figsize=(15,10))
        for i in range(6*(nFigs-1), self.len):
            _ax = ax[(i-6*(nFigs-1))//3, (i-6*(nFigs-1))%3]
            # self.data[i].plot_poincare(ax=_ax)
            self.data[i].plot_kam_surface(ns=[1], ax=_ax)
            plotCase(self.data[i], ax=_ax)
            _ax.set_title(self.name[i])
            _ax.set_xlim(xmin=1.0, xmax=1.8)
            _ax.set_ylim(ymin=-0.4, ymax=0.4)
        fig.tight_layout()
        fig.savefig("poincare"+str(nFigs-1)+".pdf")


if __name__ == "__main__":
    pass
