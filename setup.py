#!/usr/bin/env python

import os
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# from https://packaging.python.org/guides/single-sourcing-package-version/
def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), "rt") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]

    raise RuntimeError("Unable to find version string.")


setuptools.setup(
    name="diffir",
    version=get_version("diffir/__init__.py"),
    author="TODO",
    author_email="TODO",
    description="TODO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="TODO",
    packages=setuptools.find_packages(),
    install_requires=[
        "profane>=0.2.3,<0.3",
        "mako~=1.1",
        "ir_datasets>=0.3.1",
        "pytrec_eval>=0.5",
        "intervaltree>=3.1.0",
        "rich>=9.13.0",
    ],
    classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent"],
    python_requires=">=3.6",
    include_package_data=True,
    entry_points={"console_scripts": ["diffir=diffir.run:main", "diffir-batch=diffir.batchrun:main"]},
)
