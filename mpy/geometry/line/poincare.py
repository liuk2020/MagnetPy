#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# poincare.py 


import numpy as np
import matplotlib.pyplot as plt
from .line import Line
from typing import List, Tuple


def plotPoincare(lines: List[Line], toroidalIdx: int=0, ax=None, **kwargs) -> Tuple[np.ndarray]:
    """
    Returns:
        rArr, zArr
    """
    
    if ax is None:
        fig, ax = plt.subplots()
    plt.sca(ax)
    if kwargs.get("marker") == None:
        kwargs.update({"marker": "."})
    if kwargs.get("s") == None:
        kwargs.update({"s": 2.0})

    if isinstance(lines, Line):
        lines = [lines]

    for line in lines:
        rArr = list()
        zArr = list()
        for i in range(len(line.rArr)):
            if (i-toroidalIdx) % line.phiNums == 0:
                rArr.append(line.rArr[i])
                zArr.append(line.zArr[i])
        dots = ax.scatter(rArr, zArr, **kwargs)
    plt.axis("equal")

    return 


if __name__ == "__main__":
    pass
