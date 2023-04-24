#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# _FieldLine.py


import numpy as np
from ._SPECField import SPECField
from typing import List


class FieldLine:
    """
    Field line in SPEC coordinates and cylindrical coordinates! 
    """

    def __init__(self, nfp: int, nZeta: int, sArr: np.ndarray, thetaArr: np.ndarray, zetaArr: np.ndarray, rArr: np.ndarray, zArr: np.ndarray) -> None:
        self.nfp = nfp
        self.nZeta = nZeta
        self.sArr = sArr
        self.thetaArr = thetaArr
        self.zetaArr = zetaArr
        self.rArr = rArr
        self.zArr = zArr

    @classmethod
    def getLine_tracing(cls, bField: SPECField, nZeta: int, sArr: np.ndarray, thetaArr: np.ndarray, zetaArr: np.ndarray):
        rGrid, r_s, r_theta, r_zeta, zGrid, z_s, z_theta, z_zeta = bField.getGrid()
        rArr = bField.interpValue(baseData=rGrid, sValue=sArr, thetaValue=thetaArr, zetaValue=zetaArr)
        zArr = bField.interpValue(baseData=zGrid, sValue=sArr, thetaValue=thetaArr, zetaValue=zetaArr)
        return cls(
            nfp = bField.nfp, 
            nZeta = nZeta,
            sArr = sArr,
            thetaArr = thetaArr,
            zetaArr = zetaArr,
            rArr = rArr,
            zArr = zArr
        )


if __name__ == "__main__":
    pass
