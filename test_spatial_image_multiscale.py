import pytest

from ipfsspec import IPFSFileSystem
import xarray as xr

from spatial_image_multiscale import Method, to_multiscale

IPFS_FS = IPFSFileSystem()
IPFS_CID = 'bafybeigtpnf3w2iymkm4ilcgbin6btbsyrchknzs2d4dh3szmjl3ey4jue'

@pytest.fixture
def input_images():
    result = {}

    store = IPFS_FS.get_mapper(f'ipfs://{IPFS_CID}/input/cthead1.zarr')
    image_ds = xr.open_zarr(store)
    image_da = image_ds.cthead1
    result['cthead1']  = image_da

    store = IPFS_FS.get_mapper(f'ipfs://{IPFS_CID}/input/small_head.zarr')
    image_ds = xr.open_zarr(store)
    image_da = image_ds.small_head
    result['small_head']  = image_da

    return result

def test_base_scale(input_images):
    image = input_images['cthead1']

    multiscale = to_multiscale(image, [])
    xr.testing.assert_equal(image, multiscale[0])

    image = input_images['small_head']
    multiscale = to_multiscale(image, [])
    xr.testing.assert_equal(image, multiscale[0])

def test_uniform_scale_factors(input_images):
    dataset_name = 'cthead1'
    image = input_images[dataset_name]
    multiscale = to_multiscale(image, [4,2])
    baseline_name = '4_2'
    for idx, scale in enumerate(multiscale):
        store = IPFS_FS.get_mapper(f'ipfs://{IPFS_CID}/baseline/{dataset_name}/{baseline_name}/{idx}')
        image_ds = xr.open_zarr(store)
        baseline = image_ds[dataset_name]
        xr.testing.assert_equal(baseline, scale)

    dataset_name = 'small_head'
    image = input_images[dataset_name]
    multiscale = to_multiscale(image, [3,2,2])
    baseline_name = '3_2_2'
    for idx, scale in enumerate(multiscale):
        store = IPFS_FS.get_mapper(f'ipfs://{IPFS_CID}/baseline/{dataset_name}/{baseline_name}/{idx}')
        image_ds = xr.open_zarr(store)
        baseline = image_ds[dataset_name]
        xr.testing.assert_equal(baseline, scale)
