#!/usr/bin/env python3

import os
import io
from setuptools import setup, find_packages


def read(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    return io.open(filepath, encoding="utf-8").read()

install_requires = [
    "numpy==2.0.1",
    "scipy==1.13.1",
    "xarray==2024.7.0",
    "pandas==2.2.2",
    "datetime==5.5",
    "netcdf4==1.7.2",
    "matplotlib==3.9.2",
    "cartopy==0.23.0",
    "cmocean==4.0.3",
]

test_requires = ["pytest", "pytest-cov"]

extras_require = {"test": test_requires, "all": install_requires + test_requires}

setup(
    name="energy_onshore",
    use_scm_version=True,
    setup_requires=["setuptools-scm>=8.1.0"],
    description="Library to compute wind energy indicators.",
    author="Aleksander Lacima, Francesc Roura-Adserias",
    author_email="aleksander.lacima@bsc.es, francesc.roura@bsc.es",
    url="https://earth.bsc.es/gitlab/digital-twins/de_340-2/energy_onshore",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
)
