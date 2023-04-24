#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# _SPECField.py


import numpy as np
from scipy.interpolate import interpn
from ..specOut import SPECOut


deltaS = 1e-4


class SPECField:
    """
    Magnetic field in SPEC coordinates! 
    """

    def __init__(self, specData: SPECOut, lvol: int,
    sResolution: int=2, thetaResolution: int=2, zetaResolution: int=2) -> None:
        """
        Args:
            specData: the `mpy.SPECOut` or `py_spec.SPECout` class 
            lvol: the number of the volume 
            sResolution: the resolution in the s direction 
            thetaResolution: the resolution in the poloidal direction
            zetaResolution: the resolution in the toroidal direction 
        """
        self.specData = specData
        self.lvol = lvol
        self.nfp = specData.input.physics.Nfp
        self.sArr = np.linspace(-1+deltaS, 1-deltaS, sResolution)
        self.thetaArr = np.linspace(0, 2*np.pi, thetaResolution)
        self.zetaArr = np.linspace(0, 2*np.pi/self.nfp, zetaResolution)

    def interpValue(self, baseData: np.ndarray, sValue: float or np.ndarray, thetaValue: float or np.ndarray, zetaValue: float or np.ndarray):
        thetaValue = thetaValue %  (2*np.pi)
        zetaValue = zetaValue % (2*np.pi/self.nfp)
        grid = (self.sArr, self.thetaArr, self.zetaArr)
        point = (sValue, thetaValue, zetaValue)
        return interpn(grid, baseData, point)

    def changeResolution(self, sResolution: int=2, thetaResolution: int=2,zetaResolution: int=2) -> None:
        self.sArr = np.linspace(-1+deltaS, 1-deltaS, sResolution)
        self.thetaArr = np.linspace(0, 2*np.pi, thetaResolution)
        self.zetaArr = np.linspace(0, 2*np.pi/self.nfp, zetaResolution)

    def getGrid(self):
        """
        return:
            rGrid, r_s, r_theta, r_zeta, zGrid, z_s, z_theta, z_zeta
        """
        rGrid, r_s, r_theta, r_zeta, zGrid, z_s, z_theta, z_zeta = self.specData.get_RZ_derivatives(
            lvol = self.lvol, 
            sarr = self.sArr,
            tarr = self.thetaArr,
            zarr = self.zetaArr
        )
        return rGrid, r_s, r_theta, r_zeta, zGrid, z_s, z_theta, z_zeta

    def getB(self):
        """
        return:
            bSupS, bSupTheta, bSupZeta
        """
        field = self.specData.get_B(
            lvol = self.lvol, 
            sarr = self.sArr,
            tarr = self.thetaArr,
            zarr = self.zetaArr
        )
        bSupS = field[:,:,:,0]
        bSupTheta = field[:,:,:,1]
        bSupZeta = field[:,:,:,2]
        return bSupS, bSupTheta, bSupZeta
    
    def getJacobian(self):
        jacobian = self.specData.jacobian(
            lvol = self.lvol, 
            sarr = self.sArr,
            tarr = self.thetaArr,
            zarr = self.zetaArr
        )
        return jacobian

    def getMetric(self):
        metric = self.specData.metric(
            lvol = self.lvol, 
            sarr = self.sArr,
            tarr = self.thetaArr,
            zarr = self.zetaArr
        )
        return metric


if __name__ == "__main__":
    pass
