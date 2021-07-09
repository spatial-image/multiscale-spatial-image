import pytest

import fsspec
from ipfsspec import IPFSFileSystem
import xarray as xr

from spatial_image_multiscale import Method, to_multiscale

IPFS_FS = IPFSFileSystem()
IPFS_CID = 'bafybeibkmjaucsb4kzkyyprxnwezidayz76kbybpujkhywsjak4esdqyai'

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
