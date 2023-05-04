#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# surface.py


import numpy as np
from .specField import SPECField
from .fieldLine import FieldLine
from scipy.optimize import leastsq
from typing import Tuple


class SPECSurface:
    
    def __init__(self, bField: SPECField, line: FieldLine, mpol: int=8, ntor: int=8) -> None: 
        """
        Use the least squares method to get the radial coordinates of the magnetic surface in the SPEC coordinates! 
        """
        self.nfp = bField.nfp 
        self.stellsym = bField.specData.input.physics.Istellsym
        if not self.stellsym:
            raise ValueError(
                "There is no codes without stellarator symmetry! "
            )
        self.mpol, self.ntor = mpol, ntor
        self.xm, self.xn = self.initMN(mpol, ntor)
        coeffArr = self.initCoeff(line)
        mnLen = len(self.xm)
        self.sCoeffcos, self.sCoeffsin = coeffArr[0: mnLen], coeffArr[mnLen: 2*mnLen]

    def initMN(self, mpol: int, ntor: int) -> Tuple[np.ndarray]:
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
    
    def initCoeff(self, line: FieldLine): 
        mnLen = len(self.xm)
        angleLen = len(line.thetaArr)
        def getS(coeffArr, angleArr):
            sCoeffcos = coeffArr[0: mnLen]
            sCoeffsin = coeffArr[mnLen: 2*mnLen]
            thetaArr = angleArr[0: angleLen]
            zetaArr = angleArr[angleLen: 2*angleLen]
            angleMat = np.dot(self.xm.reshape(-1,1), thetaArr.reshape(1,-1)) - self.nfp * np.dot(self.xn.reshape(-1,1), zetaArr.reshape(1,-1))
            return np.dot(sCoeffcos.reshape(1,-1), np.cos(angleMat)) + np.dot(sCoeffsin.reshape(1,-1), np.sin(angleMat))
        def getErr(coeffArr, angleArr, s):
            return (s.reshape(1,-1) - getS(coeffArr, angleArr)).flatten()
        coeffArr = leastsq(getErr, np.zeros(2*mnLen), args=(np.append(line.thetaArr,line.zetaArr), line.sArr))
        return coeffArr[0]
        
    def getS(self, theta: float or np.ndarray, zeta: float or np.ndarray):
        if isinstance(theta, float):
            angle = self.xm * theta - self.xn * self.nfp * zeta
            return np.sum(self.sCoeffcos*np.cos(angle)) + np.sum(self.sCoeffsin*np.sin(angle))
        else:
            pass


if __name__ == "__main__": 
    pass
