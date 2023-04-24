#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# fieldLine.py


import h5py
import numpy as np
from .torMagneticField import TorMagneticField
from .poincare import Poincare


class FieldLine:

    def __init__(self, bField: TorMagneticField, sArr: np.ndarray, thetaArr: np.ndarray, zetaArr: np.ndarray) -> None:
        """
        The field line in SPEC corrdinates! 
        """
        self.bField = bField
        assert sArr.size == thetaArr.size == zetaArr.size
        self.sArr = sArr
        self.thetaArr = thetaArr
        self.zetaArr = zetaArr
        rArr, zArr, _bSupS, _bSupTheta, _bSupZeta, _jacobian, _metric = self.bField.interpValue(
            sValue = sArr, 
            thetaValue = thetaArr, 
            zetaValue = zetaArr
        )
        self.rArr = rArr
        self.zArr = zArr

    @classmethod
    def readH5(cls, lineFile: str, fieldFile: str) -> None:
        _content = h5py.File(lineFile + "_fieldLine.h5", 'r')
        return cls(
            bField = TorMagneticField.readH5(fieldFile), 
            sArr = _content["sArr"][:], 
            thetaArr = _content["thetaArr"][:], 
            zetaArr = _content["zetaArr"][:],
        )

    def writeH5(self, fileName: str) -> None:
        with h5py.File(fileName + "_fieldLine.h5", 'w') as f:
            f.create_dataset("sArr", data=self.sArr)
            f.create_dataset("thetaArr", data=self.thetaArr)
            f.create_dataset("zetaArr", data=self.zetaArr)

    def getPoincare(self, phi: float=0) -> Poincare:
        rArr = list()
        zArr = list()
        for i in range(len(self.zetaArr)-1):
            nPhi = (self.zetaArr[i]-phi) // (2*np.pi)
            nextnPhi = (self.zetaArr[i+1]-phi) // (2*np.pi)
            zeta = (self.zetaArr[i]-phi) % (2*np.pi)
            nextZeta = (self.zetaArr[i+1]-phi) % (2*np.pi)
            if nextnPhi - 1 == nPhi or nextnPhi + 1 == nPhi:
                if nextnPhi - 1 == nPhi:
                    zeta = zeta - (2*np.pi)
                elif nextnPhi + 1 == nPhi:
                    nextZeta = nextZeta - (2*np.pi)
                _r = (
                    (nextZeta * self.rArr[i] - zeta * self.rArr[i+1])
                    / (nextZeta - zeta)
                )
                _z = (
                    (nextZeta * self.zArr[i] - zeta * self.zArr[i+1])
                    / (nextZeta - zeta)
                )
                rArr.append(_r)
                zArr.append(_z)
        return Poincare(phi, np.array(rArr).reshape(-1), np.array(zArr).reshape(-1))


if __name__ == "__main__":
    pass
