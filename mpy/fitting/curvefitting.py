#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# curvefitting.py


import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import OptimizeResult
from typing import Tuple


def fitCurve(thetaArr: np.ndarray, sArr: np.ndarray, mpol: int, debug: bool=False, **kwargs) -> Tuple[np.ndarray] or OptimizeResult:
    """
    Use the least squares method to fit the closed curve! 
    if `debug` is false, return xm, coeffSin, coeffCos 
        s = \sum(coeffSin*sin(xm*theta) + coeffCos*cos(xm*theta))
    else if `debug` is true, return class `scipy.optimize.OptimizeResult` 
    """

    assert thetaArr.shape == sArr.shape
    assert 2*(mpol+1) < thetaArr.size
    # `verbose = 1`: display a termination report.(0: work silently; 2: display progress during iterations (not supported by 'lm' method))
    if kwargs.get("verbose") is None:
        kwargs.update({"verbose": 1}) 
    xm = np.arange(mpol+1)
    
    def getS(coeffArr: np.ndarray, thetaArr: np.ndarray) -> np.ndarray:
        coeffSin = coeffArr[0: mpol+1]
        coeffCos = coeffArr[mpol+1: 2*(mpol+1)]
        angleMat = np.dot(xm.reshape(-1,1), thetaArr.reshape(1,-1))
        return (
            np.dot(coeffSin.reshape(1,-1), np.sin(angleMat)) + 
            np.dot(coeffCos.reshape(1,-1), np.cos(angleMat))
        ).flatten()
    
    def getErr(coeffArr: np.ndarray, angleArr: np.ndarray, s: np.ndarray) -> np.ndarray:
        return s - getS(coeffArr, angleArr)
    
    optimizeRes = least_squares(getErr, np.zeros(2*(mpol+1)), args=(thetaArr, sArr), bounds=(-np.inf, np.inf), **kwargs)

    if debug:
        return optimizeRes
    else:
        assert optimizeRes.success
        return xm, optimizeRes.x[0: mpol+1], optimizeRes.x[mpol+1: 2*(mpol+1)]
    

if __name__ == "__main__":
    pass
