#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# surface.py


from ._SPECField import SPECField
from ._FieldLine import FieldLine
from scipy.optimize import leastsq


class Surface:
    
    def __init__(self, bField: SPECField, line: FieldLine, mpol: int=4, ntor: int=4) -> None: 
        self.nfp = bField.nfp 


    def getCoeff(self, bField: SPECField, line: FieldLine, mpol: int, ntor: int): 
        pass


if __name__ == "__main__": 
    pass
