#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# poincare.py


import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class Poincare:

    def __init__(self, phi: float, rArr: np.ndarray, zArr: np.ndarray) -> matplotlib.collections.PathCollection:
        self.phi = phi
        assert rArr.size == zArr.size
        self.nums = rArr.size
        self.rArr = rArr
        self.zArr = zArr

    def plot(self, ax=None, fontsize=15, **kwargs) -> None:
        if ax is None:
            fig, ax = plt.subplots()
        plt.sca(ax)
        if kwargs.get("marker") == None:
            kwargs.update({"marker": "."})
        if kwargs.get("s") == None:
            kwargs.update({"s": 1.0})
        dots = ax.scatter(self.rArr, self.zArr, **kwargs)
        plt.xlabel("R/m", fontsize=fontsize)
        plt.ylabel("Z/m", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        return dots


if __name__ == "__main__":
    pass
