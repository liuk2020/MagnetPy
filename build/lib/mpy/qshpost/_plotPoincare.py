import matplotlib.pyplot as plt
from ..specOut import SPECOut


def plotCase(case: SPECOut, ax=None, **kwargs):
    toroidalIdx = 0
    if ax is None:
        fig, ax = plt.subplots()
    plt.sca(ax)
    rr = case.poincare.R[:, :, toroidalIdx]
    zz = case.poincare.Z[:, :, toroidalIdx]
    if kwargs.get("marker") == None:
        kwargs.update({"marker": "."})
    if kwargs.get("c") == None:
        pass
    if kwargs.get("s") == None:
        kwargs.update({"s": 0.3})
    nptrj = rr.shape[0]
    for ii in range(nptrj-1):
        dots = ax.scatter(rr[ii, :], zz[ii, :], **kwargs)
    plt.xlabel("R [m]", fontsize=14)
    plt.ylabel("Z [m]", fontsize=14)
    plt.axis("equal")


if __name__ == "__main__":
    pass
