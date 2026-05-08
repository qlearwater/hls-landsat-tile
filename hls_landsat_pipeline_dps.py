#!/usr/bin/env python3

"""
DPS-ready Landsat -> HLS processing pipeline

Key DPS features:
- CLI arguments
- Uses /tmp for intermediate work
- Uses OUTPUT_DIR for DPS-collected outputs
- Uploads final outputs to S3
- Safe logging
- Proper failure handling
- Requester-pays compatible
"""

import os
import sys
import glob
import shutil
import logging
import argparse
from pathlib import Path

import boto3
import earthaccess
import rasterio as rio
from osgeo import gdal

from maap.maap import MAAP
from rasterio.session import AWSSession

import time
import fsspec
import s3fs
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime
from urllib.parse import urlparse
from collections import defaultdict

# =============================================================================
# CONFIG
# =============================================================================
LOG_DIR = "/tmp/logs"
os.makedirs(LOG_DIR, exist_ok=True)

TMP_DIR = "/tmp/hls_work"
os.makedirs(TMP_DIR, exist_ok=True)

# DPS harvest directory
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/dps_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Your MAAP S3 bucket
S3_BUCKET = "maap-ops-workspace"
S3_PREFIX = "clearwater/shared"

# GDAL tuning
gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
gdal.SetConfigOption("CPL_TMPDIR", "/tmp")
gdal.SetConfigOption("AWS_REQUEST_PAYER", "requester")
gdal.SetConfigOption("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".TIF,.tif,.vrt")

# =============================================================================
# LOGGING
# =============================================================================

logger = logging.getLogger("hls_pipeline")

logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s %(levelname)s %(message)s"
)

#
# Console handler
#
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

# #
# # File handler
# #
# file_handler = logging.FileHandler(LOG_FILE)
# file_handler.setFormatter(formatter)

#
# Attach handlers
#
logger.addHandler(console_handler)
# logger.addHandler(file_handler)

# =============================================================================
# AWS / MAAP SETUP
# =============================================================================

def configure_requester_pays():

    logger.info("Configuring requester-pays credentials")

    maap = MAAP(maap_host="api.maap-project.org")

    credentials = maap.aws.requester_pays_credentials()

    boto3_session = boto3.Session(
        aws_access_key_id=credentials["aws_access_key_id"],
        aws_secret_access_key=credentials["aws_secret_access_key"],
        aws_session_token=credentials["aws_session_token"],
    )

    aws_session = AWSSession(
        boto3_session,
        requester_pays=True
    )

    return maap, aws_session


# =============================================================================
# DOWNLOAD HLS REFERENCE GRANULE
# =============================================================================

def download_hls_granule(mgrs_tile, date):

    logger.info(f"Downloading HLS reference for {mgrs_tile} {date}")


    earthaccess.login()

    results = earthaccess.search_data(
        short_name="HLSL30",
        temporal=(date, date),
        granule_name=f"*T{mgrs_tile}*"
    )

    if len(results) == 0:
        raise RuntimeError(
            f"No HLS granule found for {mgrs_tile} {date}"
        )

    downloaded_files = earthaccess.download(
        results,
        local_path=OUTPUT_DIR
    )

    logger.info(f"Downloaded {len(downloaded_files)} files")

    return downloaded_files


# =============================================================================
# PROCESS LANDSAT
# =============================================================================

def landsat_toa_granule(MGRS_TILE, DATE, REF_PATH):
    logger.info(f"Processing Landsat L1TP scene {MGRS_TILE} {DATE}")
    out_dir = os.path.join(TMP_DIR, f"{MGRS_TILE}_{DATE}")
    os.makedirs(out_dir, exist_ok=True)
    
    mgrs_tile = MGRS_TILE
    date = datetime.strptime(DATE,"%Y-%m-%d").strftime("%Y%j")
    path = REF_PATH
    try:
        # Only open the last matched file to get metadata to find corresponding Landsat granules
        with rio.open(path) as src:
            landsat_tags = src.tags()['LANDSAT_PRODUCT_ID']
            logger.info(landsat_tags)
            landsat_list = [scene.strip() for scene in landsat_tags.split(';') if scene.strip()]
            # dictionary of band paths list by keys of scene_id
            landsat_scene_paths = {}
            toa_scene_paths = {}
            mgrs_scene_paths = {}
            aux_scene_paths = {}
            input_paths = defaultdict(list) # dictionary of compositing scenes list by keys of band
            bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "SZA"]
            new_landsat_list = []
            for scene in landsat_list:
                parts = scene.split("_")
                year = parts[3][:4]
                ymd = parts[3]
                landsatpath = parts[2][:3]
                landsatrow = parts[2][3:]
                tier = parts[6]
                # Define the base directory for Collection 2 Level-1
                base_vsis3 = "/vsis3/usgs-landsat/collection02/level-1/standard/oli-tirs"
                base_s3 = "s3://usgs-landsat/collection02/level-1/standard/oli-tirs"
                base_out = out_dir
                if tier == 'T1':
                    scene_id = scene
                else:
                    # construct scene_id
                    scene_s3 = find_landsat_scene(base_s3,year,landsatpath,landsatrow,ymd)
                    scene_id = scene_s3[0].split("/")[-1]
                new_landsat_list.append(scene_id)            
                # Construct full L1TP path, including SZA
                landsat_paths = [f"{base_vsis3}/{year}/{landsatpath}/{landsatrow}/{scene_id}/{scene_id}_{band}.TIF" for band in bands]            
                landsat_scene_paths[scene_id] = landsat_paths
                # auxiliary MTL file path
                aux_scene_paths[scene_id] = f"{base_s3}/{year}/{landsatpath}/{landsatrow}/{scene_id}/{scene_id}_MTL.xml"
                # temporal output paths               
                toa_paths = [f"{base_out}/{scene_id}_TOA_{band}.TIF" for band in bands[:7]]
                toa_scene_paths[scene_id] = toa_paths
                mgrs_paths = [f"{base_out}/{scene_id}_T{mgrs_tile}_TOA_{band}.TIF" for band in bands[:7]]
                mgrs_scene_paths[scene_id] = mgrs_paths            
            
            logger.info(new_landsat_list)
            for scene in new_landsat_list:
                with fsspec.open(aux_scene_paths[scene], mode='rb', s3={'requester_pays': True}) as f:
                    tree = ET.parse(f)
                    root = tree.getroot()
                    group = root.find('.//LEVEL1_RADIOMETRIC_RESCALING')
                    ref_mult = np.array([float(group.find(f'REFLECTANCE_MULT_BAND_{i}').text) for i in range(1,8)])
                    ref_add = np.array([float(group.find(f'REFLECTANCE_ADD_BAND_{i}').text) for i in range(1,8)])
                # print(ref_mult, ref_add)
                for i in range(7):
                    landsat_sza_scalor(landsat_scene_paths[scene][i], landsat_scene_paths[scene][7], ref_mult[i], ref_add[i], toa_scene_paths[scene][i]) # Landsat L1TP to TOA
                    warp_to_mgrs(toa_scene_paths[scene][i], mgrs_scene_paths[scene][i], path, 'cubic') # reproject, and clip Landsat tiles to MGRS grids                
                    # reconstruct to a MGRS tile-centric grouping of all scenes for the same band together
                    input_paths[i].append(mgrs_scene_paths[scene][i])
    
            # Construct VRT and TIF output paths
            vrt_paths = [f"{base_out}/Landsat_TOA_T{mgrs_tile}_{date}_mosaic_{band}.vrt" for band in bands[:7]]
            tiff_paths = [f"{OUTPUT_DIR}/Landsat.L1TP.TOA.T{mgrs_tile}.{date}.{band}.tif" for band in bands[:7]]        
            nodata = get_nodata(mgrs_scene_paths[new_landsat_list[0]][0])
            for i in range(7):            
                build_vrt(input_paths[i], vrt_paths[i], nodata, 'cubic', 'mean')
                translate_to_geotiff(vrt_paths[i], tiff_paths[i], nodata)
    except Exception as e:
        logger.info(f"Error occurred:{e}")
        return []
    else:
        return tiff_paths

def landsat_sr_granule(MGRS_TILE, DATE, REF_PATH):
    logger.info(f"Processing Landsat SR scene {MGRS_TILE} {DATE}")
    out_dir = os.path.join(TMP_DIR, f"{MGRS_TILE}_{DATE}")
    os.makedirs(out_dir, exist_ok=True)
    
    mgrs_tile = MGRS_TILE
    date = datetime.strptime(DATE,"%Y-%m-%d").strftime("%Y%j")
    path = REF_PATH
    try:
        # Only open the last matched file to get metadata to find corresponding Landsat granules
        with rio.open(path) as src:
            landsat_tags = src.tags()['LANDSAT_PRODUCT_ID']
            logger.info(landsat_tags)
            landsat_list = [scene.strip() for scene in landsat_tags.split(';') if scene.strip()]
            # dictionary of band paths list by keys of scene_id
            landsat_scene_paths = {}
            sr_scene_paths = {}
            mgrs_scene_paths = {}
            # aux_scene_paths = {}
            input_paths = defaultdict(list) # dictionary of compositing scenes list by keys of band
            bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]
            new_landsat_list = []
            for scene in landsat_list:
                parts = scene.split("_")
                year = parts[3][:4]
                ymd = parts[3]
                landsatpath = parts[2][:3]
                landsatrow = parts[2][3:]
                # tier = parts[6]
                # Define the base directory for Collection 2 Level-2
                base_vsis3 = "/vsis3/usgs-landsat/collection02/level-2/standard/oli-tirs"
                base_s3 = "s3://usgs-landsat/collection02/level-2/standard/oli-tirs"
                base_out = out_dir
                # Construct full L2SP path
                #LC08_L2SP_230087_20210328_20210402_02_T1/LC08_L2SP_230087_20210328_20210402_02_T1_SR_B5.TIF
                scene_s3 = find_landsat_scene(base_s3,year,landsatpath,landsatrow,ymd)
                scene_id = scene_s3[0].split("/")[-1]
                new_landsat_list.append(scene_id)
                landsat_paths = [f"{base_vsis3}/{year}/{landsatpath}/{landsatrow}/{scene_id}/{scene_id}_SR_{band}.TIF" for band in bands]     
                landsat_scene_paths[scene_id] = landsat_paths
                # # auxiliary MTL file path
                # aux_scene_paths[scene_id] = f"{base_s3}/{year}/{landsatpath}/{landsatrow}/{scene_id}/{scene_id}_MTL.xml"
                # temporal output paths            
                sr_paths = [f"{base_out}/{scene_id}_SR_{band}.TIF" for band in bands[:7]]
                sr_scene_paths[scene_id] = sr_paths
                mgrs_paths = [f"{base_out}/{scene_id}_T{mgrs_tile}_SR_{band}.TIF" for band in bands[:7]]
                mgrs_scene_paths[scene_id] = mgrs_paths
    
            logger.info(new_landsat_list)
            for scene in new_landsat_list:
                for i in range(7):
                    landsat_scalor(landsat_scene_paths[scene][i], sr_scene_paths[scene][i]) # Landsat L2SP to SR
                    warp_to_mgrs(sr_scene_paths[scene][i], mgrs_scene_paths[scene][i], path, 'cubic') # reproject, and clip Landsat tiles to MGRS grids                
                    # reconstruct to a MGRS tile-centric grouping of all scenes for the same band together
                    input_paths[i].append(mgrs_scene_paths[scene][i])
    
            # Construct VRT and TIF output paths
            vrt_paths = [f"{base_out}/Landsat_SR_T{mgrs_tile}_{date}_mosaic_{band}.vrt" for band in bands[:7]]
            tiff_paths = [f"{OUTPUT_DIR}/Landsat.L2SP.SR.T{mgrs_tile}.{date}.{band}.tif" for band in bands[:7]]        
            nodata = get_nodata(mgrs_scene_paths[new_landsat_list[0]][0])
            for i in range(7):            
                build_vrt(input_paths[i], vrt_paths[i], nodata, 'cubic', 'mean')
                translate_to_geotiff(vrt_paths[i], tiff_paths[i], nodata)
        
    except Exception as e:
        logger.info(f"Error occurred:{e}")
        return []
    else:
        return tiff_paths


def find_landsat_scene(base, year, path, row, date):
    fs = s3fs.S3FileSystem(anon=False, requester_pays=True)
    prefix = f"{base}/{year}/{path}/{row}/"
    dirs = fs.ls(prefix)
    return [d for d in dirs if date in d and "_T1" in d]

###### Landsat Collection 2 L1TP TOA reflectance
###### https://www.usgs.gov/landsat-missions/using-usgs-landsat-level-1-data-product
def landsat_sza_scalor(in_band_path, sza_path, scalor, offset, out_band_path):
    ds = gdal.Open(in_band_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Could not open input raster:{in_band_path}")
    ds_sza = gdal.Open(sza_path, gdal.GA_ReadOnly)
    if ds_sza is None:
        raise RuntimeError(f"Could not open input raster:{sza_path}")
    driver = gdal.GetDriverByName("GTiff")
    rows, cols = ds.RasterYSize, ds.RasterXSize
    out = driver.Create(out_band_path, cols, rows, 1, gdal.GDT_Float32)
    ## Copy Metadata
    out.SetGeoTransform(ds.GetGeoTransform())
    out.SetProjection(ds.GetProjection())
    in_band = ds.GetRasterBand(1)
    in_array = in_band.ReadAsArray()
    old_nodata = in_band.GetNoDataValue()
    # 3. Create a mask of the NoData pixels
    # If the raster has no metadata NoData, you can check for NaNs or a custom value
    if old_nodata is not None:
        mask = (in_array == old_nodata)
    else:
        mask = np.isnan(in_array)
    in_array = in_array*scalor+offset
    sza_band = ds_sza.GetRasterBand(1)
    sza_array = sza_band.ReadAsArray()*0.01
    out_array = in_array/np.cos(np.radians(sza_array))
    new_nodata = -9999.0
    out_array[mask] = new_nodata
    band = out.GetRasterBand(1)
    band.WriteArray(out_array)
    band.SetNoDataValue(new_nodata)
    band.FlushCache()
    out = None
    ds = None

###### Landsat Collection 2 Surface Reflectance
###### https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-reflectance
def landsat_scalor(in_band_path, out_band_path):
    ds = gdal.Open(in_band_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Could not open input raster:{in_band_path}")
    driver = gdal.GetDriverByName("GTiff")
    rows, cols = ds.RasterYSize, ds.RasterXSize
    out = driver.Create(out_band_path, cols, rows, 1, gdal.GDT_Float32)
    ## Copy Metadata
    out.SetGeoTransform(ds.GetGeoTransform())
    out.SetProjection(ds.GetProjection())
    in_band = ds.GetRasterBand(1)
    in_array = in_band.ReadAsArray()
    old_nodata = in_band.GetNoDataValue()
    # 3. Create a mask of the NoData pixels
    # If the raster has no metadata NoData, you can check for NaNs or a custom value
    if old_nodata is not None:
        mask = (in_array == old_nodata)
    else:
        mask = np.isnan(in_array)
    in_array = in_array*0.0000275-0.2
    out_array = in_array
    new_nodata = -9999.0
    out_array[mask] = new_nodata
    band = out.GetRasterBand(1)
    band.WriteArray(out_array)
    band.SetNoDataValue(new_nodata)
    band.FlushCache()
    out = None
    ds = None


def get_nodata(path):
    ds = gdal.Open(path)
    if ds is None:
        raise RuntimeError(f"Could not open input raster: {path}")

    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    ds = None
    return nodata

# build vrt to mosaic first
# potential issue: input Landsat granules can be in different UTM zones and need to be reprojected before mosaic
def build_vrt(input_tiles, vrt_path, nodata, resample_alg, pixel_func):
    vrt_options = gdal.BuildVRTOptions(
        resampleAlg=resample_alg,
        pixelFunction=pixel_func,
        addAlpha=False,
        srcNodata=nodata,
        VRTNodata=nodata,
    )
    vrt_ds = gdal.BuildVRT(vrt_path, input_tiles, options=vrt_options)
    if vrt_ds is None:
        raise RuntimeError(f"Failed to build VRT: {vrt_path}")
    vrt_ds = None

# A safer way is to reproject a Landsat path/row tile to grids of an MGRS tile first
# then do the mosaic with overlapping grids function
def warp_to_mgrs(input_path, warped_path, mgrs_path, resample_alg): #dst_srs, extent, nodata,
    """
    Warp an input raster to EXACTLY match an HLS reference tile grid.
    """
    # Get input nodata
    input_ds = gdal.Open(input_path)
    nodata = input_ds.GetRasterBand(1).GetNoDataValue()
    # Open the dataset
    ds = gdal.Open(mgrs_path)
    if ds is None:
        raise RuntimeError(f"Could not open input raster: {path}")
    gt = ds.GetGeoTransform()
    dst_srs = ds.GetProjection()
    # Get raster dimensions
    width = ds.RasterXSize
    height = ds.RasterYSize 
    xres = gt[1]
    yres = gt[5]
    # Calculate the bounds
    # gt[0] = Top-left X, gt[1] = Pixel width, gt[2] = X-skew (usually 0)
    # gt[3] = Top-left Y, gt[4] = Y-skew (usually 0), gt[5] = Pixel height (negative)
    minx = gt[0]
    maxy = gt[3]
    maxx = minx + gt[1] * width
    miny = maxy + gt[5] * height    
    # Format for -te / outputBounds: [minX, minY, maxX, maxY]
    te_extent = [minx, miny, maxx, maxy]
    
    warp_options = gdal.WarpOptions(
        dstSRS=dst_srs,
        outputBounds=te_extent,
        xRes=xres,
        yRes=yres,
        targetAlignedPixels=True,
        resampleAlg=resample_alg,
        srcNodata=nodata,
        dstNodata=nodata,
        multithread=True,
        format="GTiff",
        # creationOptions=[
        #     "TILED=YES",
        #     "COMPRESS=DEFLATE",
        #     "PREDICTOR=2",
        #     "BIGTIFF=YES",
        # ],
    )
    warped_ds = gdal.Warp(warped_path, input_path, options=warp_options)
    if warped_ds is None:
        raise RuntimeError(f"Failed to warp VRT to {dst_srs}")
    input_ds = None
    warped_ds = None
    ds = None

def translate_to_geotiff(input_path, output_path, nodata):
    translate_options = gdal.TranslateOptions(
        format="GTiff",
        noData=nodata,
        creationOptions=[
            "TILED=YES",
            "COMPRESS=DEFLATE",
            "PREDICTOR=2",
            "BIGTIFF=YES",
        ],
    )
    out_ds = gdal.Translate(output_path, input_path, options=translate_options)
    if out_ds is None:
        raise RuntimeError(f"Failed to translate output GeoTIFF: {output_path}")
    out_ds = None


# =============================================================================
# per granule logger
# =============================================================================
def setup_logger(mgrs_tile, date):

    logger = logging.getLogger("hls_pipeline")

    logger.setLevel(logging.INFO)

    #
    # Remove old handlers
    #
    logger.handlers.clear()

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")

    log_file = os.path.join(
        LOG_DIR,
        f"{mgrs_tile}_{date}_{timestamp}.log"
    )

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s"
    )

    #
    # Console
    #
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    #
    # File
    #
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger, log_file
    
# =============================================================================
# COPY OUTPUTS TO DPS OUTPUT_DIR
# =============================================================================

def stage_outputs_for_dps(output_files):

    staged = []

    for fp in output_files:

        dst = os.path.join(
            OUTPUT_DIR,
            os.path.basename(fp)
        )

        shutil.copyfile(fp, dst)

        logger.info(f"Staged for DPS: {dst}")

        staged.append(dst)

    return staged


# =============================================================================
# UPLOAD TO S3
# =============================================================================

def upload_outputs_to_s3(output_files, mgrs_tile, date):

    logger.info("Uploading outputs to S3")

    s3 = boto3.client("s3")

    uploaded = []

    for fp in output_files:

        key = (
            f"{S3_PREFIX}/"
            f"{mgrs_tile}_{date}/"
            f"{os.path.basename(fp)}"
        )

        logger.info(f"Uploading s3://{S3_BUCKET}/{key}")

        s3.upload_file(
            fp,
            S3_BUCKET,
            key
        )

        uploaded.append(f"s3://{S3_BUCKET}/{key}")

    return uploaded

def upload_all_outputs_to_s3(mgrs_tile, date):

    logger.info("Uploading outputs to S3")

    s3 = boto3.client("s3")

    uploaded = []

    # Create search pattern for files like: *18TWN*20231005*.tif
    julian_date = datetime.strptime(date,"%Y-%m-%d").strftime("%Y%j")
    search_pattern = os.path.join(OUTPUT_DIR, f"*{mgrs_tile}*{julian_date}*.tif")
    # Find files
    tif_files = glob.glob(search_pattern)
                
    for fp in tif_files:
        key = (
            f"{S3_PREFIX}/"
            f"{mgrs_tile}_{date}/"
            f"{os.path.basename(fp)}"
        )

        logger.info(f"Uploading s3://{S3_BUCKET}/{key}")

        s3.upload_file(
            fp,
            S3_BUCKET,
            key
        )

        uploaded.append(f"s3://{S3_BUCKET}/{key}")

    return uploaded


def upload_to_s3(local_file, mgrs_tile, date):
    s3 = boto3.client("s3")
    key = (
        f"{S3_PREFIX}/"
        f"{mgrs_tile}_{date}/"
        f"{os.path.basename(local_file)}"
    )
    s3.upload_file(
        local_file,
        S3_BUCKET,
        key,
    )

    logger.info(f"Uploaded s3://{S3_BUCKET}/{key}")
# =============================================================================
# CLEANUP
# =============================================================================

def cleanup():

    logger.info("Cleaning temporary files")

    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR, ignore_errors=True)


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mgrs_tile",
        required=True,
        help="MGRS tile"
    )

    parser.add_argument(
        "--date",
        required=True,
        help="Acquisition date YYYY-MM-DD"
    )

    args = parser.parse_args()

    mgrs_tile = args.mgrs_tile
    date = args.date
    
    logger, log_file = setup_logger(
        mgrs_tile,
        date
    )

    logger.info("=" * 60)
    logger.info(f"START PROCESSING {mgrs_tile} {date}")
    logger.info("=" * 60)

    try:

        #
        # Configure MAAP requester-pays
        #
        maap, aws_session = configure_requester_pays()

        #
        # Download HLS reference granule
        #
        hls_granules = download_hls_granule(
            mgrs_tile,
            date
        )
        ref_path = [path for path in hls_granules if 'B05' in path.name.split('.')]
        ref_path = ref_path[0]
        logger.info(ref_path)
        #
        # Run your Landsat -> HLS workflow
        #
        output_files = landsat_toa_granule(
            mgrs_tile,
            date,
            ref_path
        )
        # #
        # # Stage outputs for DPS collection
        # #
        # stage_outputs_for_dps(output_files)
        
        output_files = landsat_sr_granule(
            mgrs_tile,
            date,
            ref_path
        )
        # stage_outputs_for_dps(output_files)
        #
        # Upload to S3
        #

        uploaded = upload_all_outputs_to_s3(
            mgrs_tile,
            date
        )
        logger.info("Uploaded outputs:")
        for u in uploaded:
            logger.info(u)
        logger.info("PROCESSING COMPLETE")
        
        upload_to_s3(log_file, mgrs_tile, date)
        
    except Exception as e:

        logger.exception("PROCESSING FAILED")
        try:
            upload_to_s3(log_file, mgrs_tile, date)
        except Exception:
            pass
        
        cleanup()

        sys.exit(1)

    cleanup()


# =============================================================================
# ENTRY
# =============================================================================

if __name__ == "__main__":
    main()
    

