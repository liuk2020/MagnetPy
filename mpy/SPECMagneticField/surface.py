#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# surface.py


import numpy as np
from .specField import SPECField
from .fieldLine import FieldLine
from ..fitting import fitSurface
from typing import Tuple


class SPECSurface:
    
    def __init__(self, bField: SPECField, line: FieldLine, mpol: int=8, ntor: int=8, **kwargs) -> None: 
        """
        Use the least squares method to get the radial coordinates of the magnetic surface in the SPEC coordinates! 
        """
        self.nfp = bField.nfp 
        self.stellsym = bField.specData.input.physics.Istellsym
        if not self.stellsym:
            raise ValueError(
                "There is no codes without stellarator symmetry! "
            )
        if kwargs.get("verbose") is None:
            kwargs.update({"verbose": 1}) 
        xm, xn, sCoeffSin, sCoeffCos = fitSurface(
            line.thetaArr, line.zetaArr, line.sArr,
            mpol = mpol, ntor = ntor, 
            nfp = self.nfp, **kwargs
        )
        xm, xn, rCoeffSin, rCoeffCos = fitSurface(
            line.thetaArr, line.zetaArr, line.rArr,
            mpol = mpol, ntor = ntor, 
            # nfp = self.nfp, **kwargs
            nfp = self.nfp, stellsym = "cos", **kwargs
        )
        xm, xn, zCoeffSin, zCoeffCos = fitSurface(
            line.thetaArr, line.zetaArr, line.zArr,
            mpol = mpol, ntor = ntor, 
            # nfp = self.nfp, **kwargs
            nfp = self.nfp,  stellsym = "sin", **kwargs
        )
        self.xm, self.xn = xm, xn
        self.sCoeffSin, self.sCoeffCos = sCoeffSin, sCoeffCos
        self.rCoeffSin, self.rCoeffCos = rCoeffSin, rCoeffCos
        self.zCoeffSin, self.zCoeffCos = zCoeffSin, zCoeffCos

    def getValue(self, theta: np.ndarray, zeta: np.ndarray, value: str='s') -> np.ndarray:
        if value == 's':
            coeffSin, coeffCos = self.sCoeffSin, self.sCoeffCos
        if value == 'r':
            coeffSin, coeffCos = self.rCoeffSin, self.rCoeffCos
        if value == 'z':
            coeffSin, coeffCos = self.zCoeffSin, self.zCoeffCos
        angleMat = (
            np.dot(self.xm.reshape(-1,1), theta.reshape(1,-1)) - 
            self.nfp * np.dot(self.xn.reshape(-1,1), zeta.reshape(1,-1))
        )
        datas =  (
            np.dot(coeffSin.reshape(1,-1), np.sin(angleMat)) + 
            np.dot(coeffCos.reshape(1,-1), np.cos(angleMat))
        )
        try:
            m, n = theta.shape
            return datas.reshpe(m, n) 
        except:
            return datas.flatten()
        
        


if __name__ == "__main__": 
    pass
