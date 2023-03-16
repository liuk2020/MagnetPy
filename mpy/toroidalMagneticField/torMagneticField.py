#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# torMagneticField.py


import h5py
import numpy as np
from scipy.interpolate import interpn
from ..specOut import SPECOut
from typing import Tuple


class TorMagneticField():

    def __init__(
        self, rGrid: np.ndarray, zGrid: np.ndarray, nfp: int, 
        r_s: np.ndarray, r_theta: np.ndarray, r_zeta: np.ndarray, 
        z_s: np.ndarray, z_theta: np.ndarray, z_zeta: np.ndarray, 
        bSupS: np.ndarray, bSupTheta: np.ndarray, bSupZeta: np.ndarray,
        jacobian: np.ndarray, metric: np.ndarray
    ) -> None:
        """
        Magnetic field in toroidal coordinates!
        """
        self.sResolution, self.thetaResolution, self.zetaResolution = rGrid.shape
        self.sArr = np.linspace(-1, 1, self.sResolution)
        self.thetaArr = np.linspace(0, 2*np.pi, self.thetaResolution)
        self.zetaArr = np.linspace(0, 2*np.pi/nfp, self.zetaResolution)
        self.rGrid = rGrid
        self.zGrid = zGrid
        self.nfp = nfp
        self.r_s = r_s
        self.r_theta = r_theta
        self.r_zeta = r_zeta 
        self.z_s = z_s
        self.z_theta = z_theta
        self.z_zeta = z_zeta
        self.bSupS = bSupS
        self.bSupTheta = bSupTheta
        self.bSupZeta = bSupZeta 
        self.jacobian = jacobian
        self.metric = metric

    @classmethod
    def readSPECout(cls, specData: SPECOut, lvol: int=0, sResolution: int=2, thetaResolution: int=2, zetaResolution: int=2):
        nfp = specData.input.physics.Nfp
        sArr = np.linspace(-1, 1, sResolution)
        thetaArr = np.linspace(0, 2*np.pi, thetaResolution)
        zetaArr = np.linspace(0, 2*np.pi/nfp, zetaResolution)
        rGrid, r_s, r_theta, r_zeta, zGrid, z_s, z_theta, z_zeta = specData.get_RZ_derivatives(
            lvol = lvol,
            sarr = sArr,
            tarr = thetaArr,
            zarr = zetaArr
        )
        bSupField = specData.get_B(
            lvol = lvol,
            sarr = sArr,
            tarr = thetaArr,
            zarr = zetaArr
        )
        bSupS = bSupField[:,:,:,0]
        bSupTheta = bSupField[:,:,:,1]
        bSupZeta = bSupField[:,:,:,2]
        jacobian = specData.jacobian(
            lvol = lvol,
            sarr = sArr,
            tarr = thetaArr,
            zarr = zetaArr
        )
        metric = specData.metric(
            lvol = lvol,
            sarr = sArr,
            tarr = thetaArr,
            zarr = zetaArr
        )
        return cls(
            rGrid=rGrid, zGrid=zGrid, nfp=nfp,
            r_s=r_s, r_theta=r_theta, r_zeta=r_zeta,
            z_s=z_s, z_theta=z_theta, z_zeta=z_zeta,
            bSupS=bSupS, bSupTheta=bSupTheta, bSupZeta=bSupZeta, 
            jacobian = jacobian, metric = metric
        )

    @classmethod
    def readH5(cls, fileName: str) -> None:
        _content = h5py.File(fileName + "_torMField.h5", 'r')
        return cls(
            rGrid=_content["rGrid"][:], zGrid=_content["zGrid"][:], nfp=int(_content["nfp"][0]), 
            r_s=_content["r_s"][:], r_theta=_content["r_theta"][:], r_zeta=_content["r_zeta"][:], 
            z_s=_content["z_s"][:], z_theta=_content["z_theta"][:], z_zeta=_content["z_zeta"][:],
            bSupS=_content["bSupS"][:], bSupTheta=_content["bSupTheta"][:], bSupZeta=_content["bSupZeta"][:], 
            jacobian=_content["jacobian"][:], metric=_content["metric"][:]
        )

    def interpValue(self, sValue: float or np.ndarray, thetaValue: float or np.ndarray, zetaValue: float or np.ndarray) -> Tuple:
        r"""
        return: r, z, B^\s, B^\theta, B^\zeta, jacobian, metric
        """
        if isinstance(sValue, float):
            assert -1 <= sValue <= 1
        else:
            assert -1 <= sValue.all() <= 1
        thetaValue = thetaValue %  (2 * np.pi)
        zetaValue = zetaValue % (2 * np.pi / self.nfp)
        grid = (self.sArr, self.thetaArr, self.zetaArr)
        point = (sValue, thetaValue, zetaValue)
        return (
            interpn(grid, self.rGrid, point), 
            interpn(grid, self.zGrid, point), 
            interpn(grid, self.bSupS, point), 
            interpn(grid, self.bSupTheta, point), 
            interpn(grid, self.bSupZeta, point), 
            interpn(grid, self.jacobian, point),
            interpn(grid, self.metric, point)
        )

    def writeH5(self, fileName: str) -> None:
        with h5py.File(fileName + "_torMField.h5", 'w') as f:
            f.create_dataset("rGrid", data=self.rGrid)
            f.create_dataset("zGrid", data=self.zGrid)
            f.create_dataset("nfp", data=np.array([self.nfp]))
            f.create_dataset("r_s", data=self.r_s)
            f.create_dataset("r_theta", data=self.r_theta)
            f.create_dataset("r_zeta", data=self.r_zeta)
            f.create_dataset("z_s", data=self.z_s)
            f.create_dataset("z_theta", data=self.z_theta)
            f.create_dataset("z_zeta", data=self.z_zeta)
            f.create_dataset("bSupS", data=self.bSupS)
            f.create_dataset("bSupTheta", data=self.bSupTheta)
            f.create_dataset("bSupZeta", data=self.bSupZeta)
            f.create_dataset("jacobian", data=self.jacobian)
            f.create_dataset("metric", data=self.metric)


if __name__ == "__main__":
    pass
