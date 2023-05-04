#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# axis.py


import numpy as np
from scipy.integrate import solve_ivp 
from .specField import SPECField
from .fieldLine import FieldLine
from .readData import readB, readJacobian
from typing import Tuple


def findAxis(
    bField: SPECField, sInit: float, thetaInit: float, zetaInit: float, 
    nstep: int=32, jacobianData: str=None, maxInter: int=50, err: float=1e-5, alpha: float=0.08, **kwargs
) -> FieldLine:
    """
    Find magnetic axis by tracing field line!
    """
    if jacobianData is None:
        base_Jacobian = bField.getB()
        base_sArr = bField.sArr
        base_thetaArr = bField.thetaArr
        base_zetaArr = bField.zetaArr
    else:
        base_sArr, base_thetaArr, base_zetaArr, base_Jacobian = readJacobian(jacobianData)
    index, deltaR, deltaZ = 0, 10.0, 10.0
    sValue, thetaValue, zetaValue = sInit, thetaInit, zetaInit
    while index < maxInter and abs(deltaR)+abs(deltaZ) > err:
        line, deltaR, deltaZ, gradR, gradZ = getValue_Grad(
            bField, sValue, thetaValue, zetaValue, nstep,
            base_sArr, base_thetaArr, base_zetaArr, base_Jacobian, **kwargs
        )
        index += 1
        print("niter = " + str(index) +": " 
        + "(s,theta,zeta) = (" + "{:.1e}".format(sValue) + ", " + "{:.1e}".format(thetaValue) + ", " + "{:.1e}".format(zetaValue) +"), "
        + " (deltaR, deltaZ) = (" + "{:.1e}".format(deltaR) + ", " + "{:.1e}".format(deltaZ) + ")")
        # sValue = sValue - alpha*(deltaR+deltaZ)/(gradR[0]+gradZ[0])
        # thetaValue = thetaValue - alpha*(deltaR+deltaZ)/(gradR[1]+gradZ[1])
        # sValue = sValue - alpha*deltaR/gradR[0]/2 - alpha*deltaZ/gradZ[0]/2
        # thetaValue = thetaValue - alpha*deltaR/gradR[1]/2 - alpha*deltaZ/gradZ[1]/2
        if abs(deltaR) > abs(deltaZ):
            sValue = sValue - alpha*deltaR/gradR[0]
            thetaValue = thetaValue - alpha*deltaR/gradR[1] 
        else:
            sValue = sValue - alpha*deltaZ/gradZ[0]
            thetaValue = thetaValue - alpha*deltaZ/gradZ[1] 
    return line, sValue, thetaValue, zetaValue


def getValue_Grad(
    bField: SPECField, sValue: float, thetaValue: float, zetaValue: float, nstep: int, 
    base_sArr: np.ndarray, base_thetaArr: np.ndarray, base_zetaArr: np.ndarray, 
    base_Jacobian: np.ndarray, **kwargs
) -> Tuple[FieldLine, float, float, float]:
    """
    return line, deltaR, deltaZ, [deltaR_s,deltaR_theta], [deltaZ_s,deltaZ_theta]
    """
    deltaS, deltaTheta = 1e-10, 1e-10,
    deltaR, deltaZ, line = traceAxis(bField, sValue, thetaValue, zetaValue, nstep, 
    base_sArr, base_thetaArr, base_zetaArr, base_Jacobian, **kwargs)
    deltaR_s, deltaZ_s, line_s = traceAxis(bField, sValue+deltaS, thetaValue, zetaValue, nstep, 
    base_sArr, base_thetaArr, base_zetaArr, base_Jacobian, **kwargs)
    deltaR_theta, deltaZ_theta, line_theta = traceAxis(bField, sValue, thetaValue+deltaTheta, zetaValue, nstep, 
    base_sArr, base_thetaArr, base_zetaArr, base_Jacobian, **kwargs)
    gradR = [(deltaR_s-deltaR)/deltaS, (deltaR_theta-deltaR)/deltaTheta]
    gradZ = [(deltaZ_s-deltaZ)/deltaS, (deltaZ_theta-deltaZ)/deltaTheta]
    return line, deltaR, deltaZ, gradR, gradZ


def traceAxis(
    bField: SPECField, sValue: float, thetaValue: float, zetaValue: float, nstep: int, 
    base_sArr: np.ndarray, base_thetaArr: np.ndarray, base_zetaArr: np.ndarray, 
    base_Jacobian: np.ndarray, **kwargs
) -> Tuple[float, float, FieldLine]:
    """
    return: 
        deltaR, deltaZ, axisLine 
    """

    import pyoculus
    pyoculusField = pyoculus.problems.SPECBfield(bField.specData, bField.lvol+1)
    def getB(zeta, s_theta):
        field = pyoculusField.B_many(s_theta[0], s_theta[1], zeta) / bField.interpValue(base_Jacobian, s_theta[0], s_theta[1], zeta, sArr=base_sArr, thetaArr=base_thetaArr, zetaArr=base_zetaArr)
        bSupS = field[0, 0]
        bSupTheta = field[0, 1]
        bSupZeta = field[0, 2]
        return [bSupS/bSupZeta, bSupTheta/bSupZeta]

    if kwargs.get("method") is None:
        kwargs.update({"method": "LSODA"}) 
    if kwargs.get("rtol") is None:
        kwargs.update({"rtol": 1e-9}) 

    s_theta = [sValue, thetaValue]
    zetaStart = zetaValue
    dZeta = 2 * np.pi / bField.nfp / nstep
    sArr = [sValue]
    thetaArr = [thetaValue]
    zetaArr = [zetaValue]
    for k in range(nstep):
        sol = solve_ivp(getB, (zetaStart,zetaStart+dZeta), s_theta, **kwargs)
        sArr.append(sol.y[0,-1])
        thetaArr.append(sol.y[1,-1])
        zetaArr.append(zetaStart+dZeta)
        s_theta = [sArr[-1], thetaArr[-1]]
        zetaStart = zetaArr[-1]
    line = FieldLine.getLine_tracing(bField, nstep, np.array(sArr), np.array(thetaArr), np.array(zetaArr))
    deltaR = line.rArr[-1] - line.rArr[0]
    deltaZ = line.zArr[-1] - line.zArr[0]
    return deltaR, deltaZ, line


if __name__ == "__main__":
    pass
