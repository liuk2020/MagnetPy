#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# trace.py


import numpy as np 
from scipy.integrate import solve_ivp 
from ..geometry import Line 
from ..misc import print_progress


def traceCylindrical(fun, initPosition: np.ndarray, niter: int=128, nstep: int=128, **kwargs) -> Line:
    r"""
    Working in cylindrical coordintes (R, \phi, Z), trace the field line by solving the ODEs
        $$ \frac{dR}{d\phi} = \frac{RB_R}{B_\phi} $$
        $$ \frac{dZ}{d\phi} = \frac{RB_Z}{B_\phi} $$
    Args:
        fun: callable, thhe function to get the magnetic field in cylindrical coordintes
            `fun(R, phi, z) -> B_R, B_phi, B_Z`
        initPosition: list of the components of initial position
            `R, phi, Z`
        niter: number of toroidal periods. 
        nstep: number of intermediate step for one period        
    """

    if kwargs.get("method") is None:
        kwargs.update({"method": "LSODA"}) 
    if kwargs.get("rtol") is None:
        kwargs.update({"rtol": 1e-10})

    def ODEs(phi, rz):
        r, z = rz[0], rz[1]
        bR, bPhi, bZ = fun(r, phi, z)
        return [r*bR/bPhi, r*bZ/bPhi]

    rInit, phiInit, zInit = initPosition[0], initPosition[1], initPosition[2]
    dPhi = 2*np.pi / nstep
    rArr, zArr = [rInit], [zInit]
    print("Begin field-line tracing: ")
    for i in range(niter):                  # loop over each toroidal iteration
        for j in range(nstep):              # loop inside one iteration
            sol = solve_ivp(
                ODEs, 
                (phiInit, phiInit+dPhi), 
                [rInit, zInit],
                **kwargs
            )
            phiInit += dPhi
            rArr.append(sol.y[0,-1])
            zArr.append(sol.y[1,-1])
            rInit, zInit = rArr[-1], zArr[-1]
            print_progress(i*nstep+j+1, nstep*niter)
    
    return Line(
        rArr=np.array(rArr), zArr=np.array(zArr), phiNums=nstep
    )


if __name__ == "__main__": 
    pass
