# BLOOD ON SNOW

[![DOI](https://zenodo.org/badge/708816615.svg)](https://zenodo.org/doi/10.5281/zenodo.10034079)

## About the Project

This project aims to detect intense red blooms in the European Alps using satellite imagery (Sentinel-2).

## Getting Started

This repository is not a module; it is used to store code. However, the module requirements are stored in `./requirements.txt`.

The algorithm applies to Sentinel-2 images from Theia (https://theia.cnes.fr/atdistrib/rocket/#/home). Download both L2A reflectances and L2B snow products before launching the computation and save them in `./data`. Warning: L2A product sizes are 2.2 GB.

Explanations about the method will be available soon.

## Usage

There are two ways to launch the `./main.py` script:

1. Fill in the values of `L2A_FOLDER`, `L2B_FOLDER`, and `OUTPUT` in `main.py`, and then launch the script using Python: `python3 ./main.py`.

2. Pass the folder path as arguments and launch the Python script: `python3 main.py <path_l2a> <path_l2b> <output_folder>`.

## License

Distributed under the MIT License. See `./LICENSE` for more information.

## Contact

Feel free to contact for further information.

Léon Roussel (leon.roussel@meteo.fr)

