from pathlib import Path

import pytest
import pooch
from zarr.storage import DirectoryStore
import xarray as xr
from datatree import open_datatree

test_data_ipfs_cid = 'bafybeidr5be65a67njdaiw4cm27gjqpcmxlnhor7wak5hgm3jbhcnikt4y'
test_data_sha256 = '95c5836b49c0f2a29b48a3865b3e5e23858d555c8dceebcd43f129052ee4525d'


test_dir = Path(__file__).resolve().parent
extract_dir = "data"
test_data_dir = test_dir / extract_dir
test_data = pooch.create(path=test_dir,
     base_url=f"https://{test_data_ipfs_cid}.ipfs.w3s.link/ipfs/{test_data_ipfs_cid}/",
    registry= {
        "data.tar.gz": f"sha256:{test_data_sha256}",
    },
    retry_if_failed=5
    )

@pytest.fixture
def input_images():
    untar = pooch.Untar(extract_dir=extract_dir)
    test_data.fetch("data.tar.gz", processor=untar)
    result = {}

    store = DirectoryStore(
        test_data_dir / "input" / "cthead1.zarr", dimension_separator="/"
    )
    image_ds = xr.open_zarr(store)
    image_da = image_ds.cthead1
    result["cthead1"] = image_da

    store = DirectoryStore(
        test_data_dir / "input" / "small_head.zarr", dimension_separator="/"
    )
    image_ds = xr.open_zarr(store)
    image_da = image_ds.small_head
    result["small_head"] = image_da

    store = DirectoryStore(
        test_data_dir / "input" / "2th_cthead1.zarr",
    )
    image_ds = xr.open_zarr(store)
    image_da = image_ds['2th_cthead1']
    result["2th_cthead1"] = image_da

    return result

def verify_against_baseline(dataset_name, baseline_name, multiscale):
    store = DirectoryStore(
        test_data_dir / f"baseline/{dataset_name}/{baseline_name}", dimension_separator="/"
    )
    dt = open_datatree(store, engine="zarr", mode="r")
    xr.testing.assert_equal(dt.ds, multiscale.ds)
    for scale in multiscale.children:
        xr.testing.assert_equal(dt[scale].ds, multiscale[scale].ds)

def store_new_image(dataset_name, baseline_name, multiscale_image):
    '''Helper method for writing output results to disk
       for later upload as test baseline'''
    store = DirectoryStore(
        test_data_dir / f"baseline/{dataset_name}/{baseline_name}", dimension_separator="/",
    )
    multiscale_image.to_zarr(store)
