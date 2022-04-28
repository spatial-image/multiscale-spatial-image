import pytest

from ipfsspec import IPFSFileSystem  # type: ignore
from fsspec.implementations.http import HTTPFileSystem
import xarray as xr
from zarr.storage import DirectoryStore
from datatree import open_datatree
from pathlib import Path

from spatial_image_multiscale import Methods, to_multiscale

IPFS_FS = IPFSFileSystem()
IPFS_CID = "bafybeict5avtpfe5iyxmpny5qs4mfyuw3xqm6jyuhjzw4sjvydqf5c3w7q"
DATA_PATH = Path(__file__).absolute().parent / 'data'


@pytest.fixture
def input_images():
    result = {}

    # store = IPFS_FS.get_mapper(f"ipfs://{IPFS_CID}/input/cthead1.zarr")
    store = DirectoryStore(
        DATA_PATH / 'input' / 'cthead1.zarr',
        dimension_separator='/'
    )
    image_ds = xr.open_zarr(store)
    image_da = image_ds.cthead1
    result["cthead1"] = image_da

    store = DirectoryStore(
        DATA_PATH / 'input' / 'small_head.zarr',
        dimension_separator='/'
    )
    image_ds = xr.open_zarr(store)
    image_da = image_ds.small_head
    result["small_head"] = image_da

    return result


def verify_against_baseline(dataset_name, baseline_name, multiscale):
    store = DirectoryStore(
        DATA_PATH / f"baseline/{dataset_name}/{baseline_name}",
        dimension_separator='/'
    )
    # store = IPFS_FS.get_mapper(
    #     f"ipfs://{IPFS_CID}/baseline/{dataset_name}/{baseline_name}"
    # )
    dt = open_datatree(store, engine="zarr", mode="r")
    xr.testing.assert_equal(dt.ds, multiscale.ds)
    for scale in multiscale.children:
        xr.testing.assert_equal(dt[scale.name].ds, multiscale[scale.name].ds)


def test_base_scale(input_images):
    image = input_images["cthead1"]

    multiscale = to_multiscale(image, [])
    xr.testing.assert_equal(image, multiscale.children[0].ds["cthead1"])

    image = input_images["small_head"]
    multiscale = to_multiscale(image, [])
    xr.testing.assert_equal(image, multiscale.children[0].ds["small_head"])


def test_isotropic_scale_factors(input_images):
    dataset_name = "cthead1"
    image = input_images[dataset_name]
    multiscale = to_multiscale(image, [4, 2], method=Methods.XARRAY_COARSEN)
    verify_against_baseline(dataset_name, "4_2/XARRAY_COARSEN", multiscale)

    dataset_name = "cthead1"
    image = input_images[dataset_name]
    multiscale = to_multiscale(image, [3, 2], method=Methods.XARRAY_COARSEN)
    verify_against_baseline(dataset_name, "3_2/XARRAY_COARSEN", multiscale)

    dataset_name = "small_head"
    image = input_images[dataset_name]
    multiscale = to_multiscale(image, [3, 2, 2], method=Methods.XARRAY_COARSEN)
    verify_against_baseline(dataset_name, "3_2_2/XARRAY_COARSEN", multiscale)

    dataset_name = "cthead1"
    image = input_images[dataset_name]
    multiscale = to_multiscale(image, [4, 2], method=Methods.ITK_BIN_SHRINK)
    verify_against_baseline(dataset_name, "4_2/ITK_BIN_SHRINK", multiscale)

    dataset_name = "cthead1"
    image = input_images[dataset_name]
    multiscale = to_multiscale(image, [3, 2], method=Methods.ITK_BIN_SHRINK)
    verify_against_baseline(dataset_name, "3_2/ITK_BIN_SHRINK", multiscale)

    dataset_name = "small_head"
    image = input_images[dataset_name]
    multiscale = to_multiscale(image, [3, 2, 2], method=Methods.ITK_BIN_SHRINK)
    verify_against_baseline(dataset_name, "3_2_2/ITK_BIN_SHRINK", multiscale)


def test_anisotropic_scale_factors(input_images):
    dataset_name = "cthead1"
    image = input_images[dataset_name]
    scale_factors = [{"x": 2, "y": 4}, {"x": 1, "y": 2}]
    multiscale = to_multiscale(image, scale_factors, method=Methods.XARRAY_COARSEN)
    verify_against_baseline(dataset_name, "x2y4_x1y2/XARRAY_COARSEN", multiscale)

    dataset_name = "small_head"
    image = input_images[dataset_name]
    scale_factors = [
        {"x": 3, "y": 2, "z": 4},
        {"x": 2, "y": 2, "z": 2},
        {"x": 1, "y": 2, "z": 1},
    ]
    multiscale = to_multiscale(image, scale_factors)
    verify_against_baseline(
        dataset_name, "x3y2z4_x2y2z2_x1y2z1/XARRAY_COARSEN", multiscale
    )

    dataset_name = "cthead1"
    image = input_images[dataset_name]
    scale_factors = [{"x": 2, "y": 4}, {"x": 1, "y": 2}]
    multiscale = to_multiscale(image, scale_factors, method=Methods.ITK_BIN_SHRINK)
    verify_against_baseline(dataset_name, "x2y4_x1y2/ITK_BIN_SHRINK", multiscale)

    dataset_name = "small_head"
    image = input_images[dataset_name]
    scale_factors = [
        {"x": 3, "y": 2, "z": 4},
        {"x": 2, "y": 2, "z": 2},
        {"x": 1, "y": 2, "z": 1},
    ]
    multiscale = to_multiscale(image, scale_factors)
    # print(multiscale)
    # store = DirectoryStore(
    #     f"/home/matt/data/spatial-image-multiscale/baseline/{dataset_name}/x3y2z4_x2y2z2_x1y2z1/ITK_BIN_SHRINK",
    #     dimension_separator="/",
    # )
    # multiscale.to_zarr(store)
    verify_against_baseline(
        dataset_name, "x3y2z4_x2y2z2_x1y2z1/ITK_BIN_SHRINK", multiscale
    )
