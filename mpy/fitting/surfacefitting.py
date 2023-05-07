#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# surfacefitting.py


import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import OptimizeResult
from typing import Tuple


def fitSurface(thetaArr: np.ndarray, zetaArr: np.ndarray, sArr: np.ndarray, mpol: int, ntol: int, nfp: int=1, debug: bool=False, **kwargs) -> Tuple[np.ndarray] or OptimizeResult:
    """
    Use the least squares method to fit the toroidal surface!  
    if `debug` is false, return xm, xn, coeffSin, coeffCos 
        s = \sum(coeffSin*sin(xm*theta-nfp*xn*zeta) + coeffCos*cos(xm*theta-nfp*xn*zeta))
    else if `debug` is true, return class `scipy.optimize.OptimizeResult` 
    """
    
    assert thetaArr.shape == zetaArr.shape == sArr.shape
    thetaArr = thetaArr.flatten()
    zetaArr = zetaArr.flatten()
    sArr = sArr.flatten()
    mnLen = mpol*(2*ntol+1)+ntol+1
    angleLen = thetaArr.size
    assert 2*mnLen < thetaArr.size
    # `verbose = 1`: display a termination report.(0: work silently; 2: display progress during iterations (not supported by 'lm' method))
    if kwargs.get("verbose") is None:
        kwargs.update({"verbose": 1}) 
    xm, xn = getMN(mpol, ntol)
    
    def getS(coeffArr: np.ndarray, angleArr: np.ndarray) -> np.ndarray:
        coeffSin = coeffArr[0: mnLen]
        coeffCos = coeffArr[mnLen: 2*mnLen]
        thetaArr = angleArr[0: angleLen]
        zetaArr = angleArr[angleLen: 2*angleLen]
        angleMat = (
            np.dot(xm.reshape(-1,1), thetaArr.reshape(1,-1)) - 
            nfp * np.dot(xn.reshape(-1,1), zetaArr.reshape(1,-1))
        )
        return (
            np.dot(coeffSin.reshape(1,-1), np.sin(angleMat)) + 
            np.dot(coeffCos.reshape(1,-1), np.cos(angleMat))
        ).flatten()


    def getErr(coeffArr: np.ndarray, angleArr: np.ndarray, s: np.ndarray) -> np.ndarray:
        return s - getS(coeffArr, angleArr)
    
    optimizeRes = least_squares(getErr, np.zeros(2*mnLen), args=(np.append(thetaArr,zetaArr), sArr), bounds=(-np.inf, np.inf), **kwargs)

    if debug:
        return optimizeRes
    else:
        assert optimizeRes.success
        return xm, xn, optimizeRes.x[0:mnLen], optimizeRes.x[mnLen:2*mnLen]


def getMN(mpol: int, ntor: int) -> Tuple[np.ndarray, np.ndarray]: 
    def getM(index: int) -> int:
            if index < ntor+1:
                return 0
            else:
                return (index+ntor) // (2*ntor+1)
    def getN(index: int) -> int:
        index %= (2*ntor+1)
        if index > ntor:
            index -= (2*ntor+1)
        return index
    xm = [getM(index) for index in range((ntor+1)+mpol*(2*ntor+1))]
    xn = [getN(index) for index in range((ntor+1)+mpol*(2*ntor+1))]
    return np.array(xm), np.array(xn)


if __name__ == "__main__":
    pass
