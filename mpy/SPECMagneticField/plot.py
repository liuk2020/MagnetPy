#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# plot.py


import matplotlib.pyplot as plt
from .fieldLine import FieldLine
from typing import List


def plotPoincare(lines: List[FieldLine], toroidalIdx: int=0, ax=None, **kwargs):

    if ax is None:
        fig, ax = plt.subplots()
    plt.sca(ax)
    if kwargs.get("marker") == None:
        kwargs.update({"marker": "."})
    if kwargs.get("s") == None:
        kwargs.update({"s": 1.4})

    for line in lines:
        assert line.equalZeta
        rArr = list()
        zArr = list()
        for i in range(len(line.rArr)):
            if (i-toroidalIdx) % line.nZeta == 0:
                rArr.append(line.rArr[i])
                zArr.append(line.zArr[i])
        dots = ax.scatter(rArr, zArr, **kwargs)
    plt.axis("equal")

    return


if __name__ == "__main__":
    pass