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
    "pystac==1.10.1",
    "geopandas==1.0.1",
    "shapely==2.0.7",
    "one-pass @ git+https://gitlab.earth.bsc.es/digital-twins/de_340-2/one_pass.git@v0.10.0#egg=one-pass",
]

test_requires = ["pytest", "pytest-cov"]
docs_requires = ["sphinx", "setuptools-scm>=8.1.0"]
notebooks_requires = ["earthkit-data", "healpy", "conflator", "lxml"]

extras_require = {
    "test": test_requires,
    "docs": docs_requires,
    "notebooks": notebooks_requires,
    "all": install_requires + test_requires + docs_requires + notebooks_requires,
}

setup(
    name="energy_indicators",
    use_scm_version=True,
    setup_requires=["setuptools-scm>=8.1.0"],
    description="Library to compute wind energy indicators.",
    author="Aleksander Lacima, Francesc Roura-Adserias",
    author_email="aleksander.lacima@bsc.es, francesc.roura@bsc.es",
    url="https://earth.bsc.es/gitlab/digital-twins/de_340-2/energy_indicators",
    python_requires=">3.9",
    packages=find_packages(),
    package_data={
        "energy_indicators": ["power_curves"],
    },
    include_package_data=True,
    install_requires=install_requires,
    extras_require=extras_require,
)
