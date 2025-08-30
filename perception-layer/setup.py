"""
Setup script for the Perception and Environment Layer package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
with open(this_directory / "requirements.txt", "r", encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            requirements.append(line)

setup(
    name="perception-layer",
    version="1.0.0",
    author="eVTOL Perception Team",
    author_email="perception@evtol-defense.com",
    description="Perception and Environment Layer for eVTOL Trajectory Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/evtol-defense/perception-layer",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=1.0.0",
        ],
        "gpu": [
            "cupy>=12.0.0",
            "cudf>=23.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "perception-layer=perception_layer.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "perception_layer": [
            "config/*.yaml",
            "data/**/*",
        ],
    },
    zip_safe=False,
    keywords=[
        "evtol",
        "perception",
        "geospatial",
        "atmospheric",
        "trajectory",
        "optimization",
        "defense",
        "gis",
        "remote-sensing",
    ],
    project_urls={
        "Bug Reports": "https://github.com/evtol-defense/perception-layer/issues",
        "Source": "https://github.com/evtol-defense/perception-layer",
        "Documentation": "https://perception-layer.readthedocs.io/",
    },
)

