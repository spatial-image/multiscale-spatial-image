import pytest

from ipfsspec import IPFSFileSystem
import xarray as xr

from spatial_image_multiscale import Method, to_multiscale

IPFS_FS = IPFSFileSystem()
IPFS_CID = "bafybeibpqky6d335duxtkmwowcc6igt2q5qorqd7e5xqfoxlfxm4pozg74"


@pytest.fixture
def input_images():
    result = {}

    store = IPFS_FS.get_mapper(f"ipfs://{IPFS_CID}/input/cthead1.zarr")
    image_ds = xr.open_zarr(store)
    image_da = image_ds.cthead1
    result["cthead1"] = image_da

    store = IPFS_FS.get_mapper(f"ipfs://{IPFS_CID}/input/small_head.zarr")
    image_ds = xr.open_zarr(store)
    image_da = image_ds.small_head
    result["small_head"] = image_da

    return result


def verify_against_baseline(dataset_name, baseline_name, multiscale):
    for idx, scale in enumerate(multiscale):
        store = IPFS_FS.get_mapper(
            f"ipfs://{IPFS_CID}/baseline/{dataset_name}/{baseline_name}/{idx}"
        )
        image_ds = xr.open_zarr(store)
        baseline = image_ds[dataset_name]
        xr.testing.assert_equal(baseline, scale)


def test_base_scale(input_images):
    image = input_images["cthead1"]

    multiscale = to_multiscale(image, [])
    xr.testing.assert_equal(image, multiscale[0])

    image = input_images["small_head"]
    multiscale = to_multiscale(image, [])
    xr.testing.assert_equal(image, multiscale[0])


def test_isotropic_scale_factors(input_images):
    dataset_name = "cthead1"
    image = input_images[dataset_name]
    multiscale = to_multiscale(image, [4, 2])
    verify_against_baseline(dataset_name, "4_2", multiscale)

    dataset_name = "small_head"
    image = input_images[dataset_name]
    multiscale = to_multiscale(image, [3, 2, 2])
    verify_against_baseline(dataset_name, "3_2_2", multiscale)


def test_anisotropic_scale_factors(input_images):
    dataset_name = "cthead1"
    image = input_images[dataset_name]
    scale_factors = [{"x": 2, "y": 4}, {"x": 1, "y": 2}]
    multiscale = to_multiscale(image, scale_factors)
    verify_against_baseline(dataset_name, "x2y4_x1y2", multiscale)

    dataset_name = "small_head"
    image = input_images[dataset_name]
    scale_factors = [
        {"x": 3, "y": 2, "z": 4},
        {"x": 2, "y": 2, "z": 2},
        {"x": 1, "y": 2, "z": 1},
    ]
    multiscale = to_multiscale(image, scale_factors)
    verify_against_baseline(dataset_name, "x3y2z4_x2y2z2_x1y2z1", multiscale)
