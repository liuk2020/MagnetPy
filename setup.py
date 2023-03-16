import setuptools
from numpy.distutils.core import setup


__version__ = "0.0.0"

with open("README.md", "r") as fh:
    long_description = fh.read()

REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]

setup(
    name="mpy",
    version=__version__,
    description="Data processing tools for plasma and magnetic field",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Programming Language :: Python :: 3",
    ],
    url="https://github.com/liuk2020/MagnetPy.git",
    author="Ke Liu",
    author_email="lk2020@mail.ustc.edu.cn",
    license="GNU 3",
    packages=setuptools.find_packages(),
    install_requires=REQUIREMENTS,
)