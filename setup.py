import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kinetics_lbmc",
    version="0.0.3",
    author="Alexandre Naaim",
    author_email="alexandre.naaim@univ-lyon1.fr",
    description="A tool box for to calculate kinematics and kinetics based on natural coordinates and homogeneous formulation. ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ANaaim/Kinetics_LBMC",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
