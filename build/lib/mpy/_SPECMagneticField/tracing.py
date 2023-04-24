#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tracing.py


import numpy as np
from scipy.integrate import solve_ivp
from .torMagneticField import TorMagneticField
from .fieldLine import FieldLine
from typing import List
from ..misc import print_progress


def traceLine(bField: TorMagneticField, s0: np.ndarray, theta0: np.ndarray, zeta0: np.ndarray, 
niter: int = 100, radius: float=5.0, **kwargs) -> List[FieldLine]:
    r"""
    Working in SPEC coordintes (s, \theta, \zeta), compute magnetic field lines by solving
        $$ \frac{ds}{dl} = \frac{B^s}{B} $$     
        $$ \frac{d\theta}{dl} = \frac{B^\theta}{B} $$ 
        $$ \frac{d\zeta}{dl} = \frac{B^\zeta}{B} $$ 
    Args:
        bField: the toroidal magnetic field. 
        s0: list of s components of initial points. 
        theta0: list of theta components of initial points. 
        zeta0: list of zeta components of initial points.
        niter, radius: the length of the field line is 2*pi*radius*niter 
    """

    assert s0.shape == theta0.shape == zeta0.shape

    def getB(dLength, point):
        s = point[0]
        thtea = point[1]
        zeta = point[2]
        r, z, bSupS, bSupTheta, bSupZeta, jacobian, metric = bField.interpValue(sValue=s, thetaValue=thtea, zetaValue=zeta)
        bArr = [bSupS, bSupTheta, bSupZeta]
        bPow = 0
        for i in range(3):
            for j in range(3):
                bPow += (bArr[i]*bArr[j]*metric[0,i,j])
        b = np.power((bPow), 0.5)
        return [bSupS/b, bSupTheta/b, bSupZeta/b]

    if kwargs.get("method") is None:
        kwargs.update({"method": "LSODA"}) 
    if kwargs.get("rtol") is None:
        kwargs.update({"rtol": 1e-6}) 
    print("Begin field-line tracing: ")
    lines = list()
    nLine = len(s0)
    for i in range(nLine):
        # lengthArr = np.array([])
        sArr = np.array([])
        thetaArr = np.array([])
        zetaArr = np.array([])
        point = [s0[i], theta0[i], zeta0[i]]
        for j in range(niter):
            print_progress(i*niter+j+1, nLine*niter)
            sol = solve_ivp(getB, (j*2*np.pi*radius, (j+1)*2*np.pi*radius), point, **kwargs)
            # lengthArr = np.concatenate((lengthArr, sol.t), axis=0)
            sArr = np.concatenate((sArr, sol.y[0, :]), axis=0)
            thetaArr = np.concatenate((thetaArr, sol.y[1, :]), axis=0)
            zetaArr = np.concatenate((zetaArr, sol.y[2, :]), axis=0)
            point = [sArr[-1], thetaArr[-1], zetaArr[-1]]
        lines.append(FieldLine(bField, sArr, thetaArr, zetaArr))
    return lines


if __name__ == "__main__":
    pass
