from setuptools import setup, find_packages

setup(
    name="refoldingcontrol",
    description="",
    version="0.1",
    author="michaelkandler",
    maintainer="michaelkandler",
    maintainer_email="michael.kandler@protonmail.com",
    license="GNU GPLv3 ",
    python_requires='>=3.11',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "filterpy",
        "pandas",
        "xlsxwriter",
        "openpyxl",
        "ipywidgets",
        "jupyter"
    ],
)


