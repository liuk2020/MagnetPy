#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# line.py


import numpy as np


class BaseLine:

    def __init__(self, rArr: np.ndarray, zArr: np.ndarray, phiArr: np.ndarray) -> None:
        assert rArr.shape == zArr.shape == phiArr.shape
        self.rArr = rArr
        self.zArr = zArr
        self.phiArr = phiArr


class Line(BaseLine):

    def __init__(self, rArr: np.ndarray, zArr: np.ndarray, phiNums: int) -> None:
        assert rArr.shape == zArr.shape 
        self.rArr = rArr
        self.zArr = zArr
        self.phiNums = phiNums

    @property
    def phiArr(self):
        return np.arange(0,self.rArr.size,1) * 2*np.pi/self.phiNums


if __name__ == "__main__": 
    pass
