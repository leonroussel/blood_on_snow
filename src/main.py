#################################################################################
# Copyright (c) 2023 Léon ROUSSEL
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#################################################################################
"""
Script containing main pipeline for red algae detection on Sentinel-2 images
(also detect dust, but only above RGND threshold, thus dust detection is not accurate)
"""

import os
import numpy as np
from osgeo import gdal
import scipy.ndimage as ndi  # utils for image operations
from skimage.measure import label  # connected components
import time
import sys

# Values on masks for algae and dust detection
ALGAE_VALUE = 205
DUST_VALUE = 100

# Global variable, setup bellow the directory
L2A_FOLDER = None # directory
L2B_FOLDER = None # directory
OUTPUT = None # directory

# ---- PARAMETERS : ----
THRESHOLD_ROCK = 5000  # [Casey B. Engstrom and al., 2022]
THRESHOLD_RGND = 0.035  # 0.025 in [Casey B. Engstrom and al., 2022]
A = 1.005671414469857  # linear separation y = A.x + B
B = 0.016540470809237772
MIN_SIZE_CONNECTED_COMPONENTS = 5
# -----------------------


def get_unique_dir(directory, id_contained):
    """Check that there exists a unique file or directory containing 'id_contained' and
    returns it

    :param directory: directory to search in
    :type directory: string
    :param id_contained:
    :type id_contained: string

    :return: file containing id_contained
    :rtype: string
    """
    fn = [filename for filename in os.listdir(directory) if id_contained in filename]
    assert len(fn) == 1
    fn = fn[0]
    return fn


def read_tif(filename):

    """Read a .tif file and extract the first band as a numpy array with geo parameters

    :param filename:
    :type filename: string

    :return: first band and geo informations
    :rtype: (numpy array, gdal transform, gdal projection)
    """
    dataset = gdal.Open(filename)
    geo_transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    extracted_array = dataset.GetRasterBand(1).ReadAsArray()
    return extracted_array.astype(np.float64), geo_transform, projection


def extract_band(path_l2a, band_id):
    """Extract band from GeoTiff format in 'path_l2a' directory containing 'band_id'

    :param path_l2a: folder
    :type path_l2a: string
    :param band_id: B02, B03, ...
    :type band_id: string
    :return: first band and geo informations
    :rtype: (numpy array, gdal transform, gdal projection)
    """
    fn = get_unique_dir(path_l2a, band_id)
    extracted_array, geo_transform, projection = read_tif(path_l2a + "/" + fn)
    return extracted_array, geo_transform, projection


def get_bands(path_l2a, band_ids=["FRE_B2.", "FRE_B3.", "FRE_B4."]):
    """Get bands from given path

    :param path_l2a: path
    :type path_l2a: string
    :param band_ids: id of the bands, defaults to ["FRE_B2.", "FRE_B3.", "FRE_B4."]
    :type band_ids: list of string, optional
    :return: bands and geo information
    :rtype: (list of numpy array, gdal transform, gdal projection)
    """
    bands = []

    # Get bands :
    geo_transform, projection = None, None  # save geo param for output mask
    for band_id in band_ids:
        array, geo_transform, projection = extract_band(path_l2a, band_id)
        bands.append(array)

    bands_len = len(bands)
    assert bands_len == 3
    for i in range(1, len(bands)):
        assert bands[i].shape == bands[0].shape

    return bands, geo_transform, projection


def extract_mask(path_l2b, end_filename, value):
    """Extract & upsampled snow mask at 10 meters resolution

    :param path_l2b: folder
    :type path_l2b: string
    :param end_filename: type of the mask
    :type end_filename: string
    :param value: value to extract in the mask
    :type value: int
    :return: mask upsampled
    :rtype: boolean numpy array
    """
    fn = get_unique_dir(path_l2b, end_filename)
    extracted_array, _, _ = read_tif(path_l2b + "/" + fn)
    snow_mask = extracted_array == np.ones_like(extracted_array) * value

    # snow mask --> 20m resolution (upsampling of 2)
    upsampled_snow_mask = np.array(ndi.zoom(snow_mask, 2, order=0), dtype=np.float64)

    return upsampled_snow_mask


def save_arrays_as_tiff(
    arrays, filename, geo_transform, projection, gdal_type=gdal.GDT_Byte
):
    """Save arrays with given transform and projection

    :param arrays: array to save
    :type arrays: numpy array
    :param filename: name output
    :type filename: string
    :param geo_transform: geotransform
    :type geo_transform: gdal geotransform
    :param projection: projection
    :type projection: gdal projection
    :param gdal_type: gdal type, defaults to gdal.GDT_Byte
    :type gdal_type: gdal type, optional
    """
    assert type(arrays) == list

    [rows, cols] = arrays[0].shape
    driver = gdal.GetDriverByName("GTiff")
    output = driver.Create(
        filename, cols, rows, len(arrays), gdal_type, options=["COMPRESS=DEFLATE"]
    )  #  type could be gdal.GDT_UInt16 also, compression to have light masks
    output.SetGeoTransform(geo_transform)
    output.SetProjection(projection)
    for i in range(len(arrays)):
        output.GetRasterBand(i + 1).WriteArray(arrays[i])

    output.FlushCache()  # save to disk


def normalized_difference(a, b):
    """Normalized difference

    :param a: first array
    :type a: numpy array
    :param b: second vector
    :type b: numpy array
    :return: normalized difference between the 2 arrays
    :rtype: numpy array
    """
    return np.divide(a - b, a + b, where=(a + b) != 0)


def delete_small_connected_components(
    algae_intensities, min_component_size, connectivity=1
):
    """With algae intensities, compute the connected components
    and delete the small components
    (i.e componenents of size < 'min_component_size')

    :param algae_intensities: 0 if no algae, >0 if algae detected
    :type algae_intensities: numpy array
    :param min_component_size: minimum threshold
    :type min_component_size: int
    :param connectivity: type of neighborhood, defaults to 1 (i.e von Neumann)
    :type connectivity: int, optional
    :return: connected components numbered with size > min_component_size
    :rtype: numpy array
    """
    algae_mask = np.where(algae_intensities > 0, 1, 0)

    # label each connected component with a different integer
    connected_components, nums = label(
        algae_mask, background=0, return_num=True, connectivity=connectivity
    )

    # get size of each connected components
    bin_count = np.bincount(connected_components.flatten())
    # set to 0 the small components
    connected_components[bin_count[connected_components] < min_component_size] = 0

    # to have connected components numbered in {1, ...,  nums}
    connected_components, nums = label(
        connected_components, background=0, return_num=True, connectivity=connectivity
    )

    return connected_components, nums


def main_pipeline(path_l2a, path_l2b, output_filename="", bands_filename=[]):
    """Main pipeline with detection chain

    :param path_l2a: path l2a product (image bands)
    :type path_l2a: string
    :param path_l2b: path l2b products (cloud and snow masks)
    :type path_l2b: string
    :param output_filename: filename, defaults to ""
    :type output_filename: str, optional
    :param bands_filename: filename, defaults to []
    :type bands_filename: list, optional
    :return: final algae mask
    :rtype: numpy array
    """
    start_t = time.time()
    print("Entering pipeline :")
    print("with params", path_l2a, path_l2b, output_filename)

    # Get bands
    print("  Get bands...")
    bands, geo_transform, projection = get_bands(path_l2a)
    bands = np.array(bands)

    # Get snow mask
    snow_mask = extract_mask(path_l2b, "SNW_R2.tif", value=100)  # upsampled snow mask

    # Init bloom mask to True (progressively filtering pixels toward False)
    bloom_mask = np.full(snow_mask.shape, True, dtype=bool)
    print("    Nb pixels:", np.count_nonzero(bloom_mask))
    print("    t=", time.time() - start_t)

    # -------------------
    # --- SNOW MASK ----- [S. Gascoin and al., 2019]
    print("  Snow mask...")
    bloom_mask[np.logical_not(snow_mask)] = False
    print("    Nb pixels:", np.count_nonzero(bloom_mask))
    print("    t=", time.time() - start_t)
    # -------------------

    # -------------------
    # --- ROCK MASK ----- [Casey B. Engstrom and al., 2022]
    print("  Rock mask...")
    band_b3 = bands[1, :, :]
    bloom_mask[band_b3 < THRESHOLD_ROCK] = False
    print("    Nb pixels:", np.count_nonzero(bloom_mask))
    print("    t=", time.time() - start_t)
    # -------------------

    # -------------------
    # --- RGND MASK ----- [Casey B. Engstrom and al., 2022]
    print("  RGND mask...")
    RGND = normalized_difference(
        bands[2, :, :], bands[1, :, :]
    )  # (B4 - B3) / (B4 + B3)
    bloom_mask[RGND < THRESHOLD_RGND] = False
    print("    Nb pixels:", np.count_nonzero(bloom_mask))
    print("    t=", time.time() - start_t)
    # -------------------

    # -------------------
    # --- LINEAR MASK ---
    # (bloom_mask contains either algae or dust)
    print("  Linear mask...")
    GBND = normalized_difference(
        bands[1, :, :], bands[0, :, :]
    )  # (B3 - B2) / (B3 + B2)
    dust_mask = bloom_mask.copy()  # init dust mask
    algae_condition = np.greater(RGND, A * GBND + B)
    dust_mask[algae_condition] = False
    bloom_mask[np.logical_not(algae_condition)] = False
    print("    Nb pixels:", np.count_nonzero(bloom_mask))
    print("    t=", time.time() - start_t)
    # -------------------

    # ------------------------
    # - CONNECTED COMPONENTS -
    print("  Connected components...")
    (connected_components, _,) = delete_small_connected_components(
        bloom_mask,
        min_component_size=MIN_SIZE_CONNECTED_COMPONENTS,
        connectivity=2,
    )
    bloom_mask = connected_components > 0
    print("    Nb pixels:", np.count_nonzero(bloom_mask))
    print("    t=", time.time() - start_t)
    # ------------------------

    # Init finak mask
    # which is a merge between algae and dust detection
    final_mask = np.zeros(bloom_mask.shape, dtype=np.uint8)
    final_mask[dust_mask] = DUST_VALUE
    final_mask[bloom_mask] = ALGAE_VALUE

    # SAVE final mask
    if not output_filename == "":
        print("  Save...")
        save_arrays_as_tiff([final_mask], output_filename, geo_transform, projection)

    print("Total time :", time.time() - start_t)

    return final_mask


# Usage is :
# python3 main.py <path_l2a> <path_l2b> <output_filename>
if __name__ == "__main__":
    print("Argv :", sys.argv)
    
    # if folder given in argument
    if len(sys.argv) > 2:
        L2A_FOLDER = sys.argv[1]
        L2B_FOLDER = sys.argv[2]
        OUTPUT = sys.argv[3]

    # if folder given in this script or in args
    if L2A_FOLDER:
        path_l2a = L2A_FOLDER
        path_l2b = L2B_FOLDER
        output_dir = OUTPUT

    else:
        raise ValueError("L2A and L2B folder of the image should be given in args or in main.py script")

    # each dates contains array for band B2, B3, B4 of S2 images
    # (we use here 'FRE' bands of theia abnds)
    band_ids = ["FRE_B2.", "FRE_B3.", "FRE_B4."]

    output_filename = output_dir + "/" + path_l2a.split("/")[-1] + "_ALG_DUS.tif"
    bands_filename = [
        output_dir + "/" + path_l2a.split("/")[-1] + "_" + band_id + "tif"
        for band_id in band_ids
    ]

    # compute and save mask
    final_mask = main_pipeline(path_l2a, path_l2b, output_filename, bands_filename)
