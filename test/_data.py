from pathlib import Path

import pytest
import pooch
import xarray as xr

test_data_ipfs_cid = "bafybeiaskr5fxg6rbcwlxl6ibzqhubdleacenrpbnymc6oblwoi7ceqzta"
test_data_sha256 = "507dd779cba007c46ea68a5fe8865cabd5d8a7e00816470faae9195d1f1c3cd1"


test_dir = Path(__file__).resolve().parent
extract_dir = "data"
test_data_dir = test_dir / extract_dir
test_data = pooch.create(
    path=test_dir,
    # base_url=f"https://{test_data_ipfs_cid}.ipfs.w3s.link/ipfs/{test_data_ipfs_cid}/",
    base_url="https://github.com/spatial-image/multiscale-spatial-image/releases/download/v2.0.0/",
    registry={
        "data.tar.gz": f"sha256:{test_data_sha256}",
    },
    retry_if_failed=5,
)


@pytest.fixture
def input_images():
    untar = pooch.Untar(extract_dir=extract_dir)
    test_data.fetch("data.tar.gz", processor=untar)
    result = {}

    store_path = test_data_dir / "input" / "cthead1.zarr"
    try:
        from zarr.storage import DirectoryStore

        store = DirectoryStore(store_path, dimension_separator="/")
    except ImportError:
        from zarr.storage import LocalStore

        store = LocalStore(store_path)
    image_ds = xr.open_zarr(store)
    image_da = image_ds.cthead1
    result["cthead1"] = image_da

    store_path = test_data_dir / "input" / "small_head.zarr"
    try:
        from zarr.storage import DirectoryStore

        store = DirectoryStore(store_path, dimension_separator="/")
    except ImportError:
        from zarr.storage import LocalStore

        store = LocalStore(store_path)
    image_ds = xr.open_zarr(store)
    image_da = image_ds.small_head
    result["small_head"] = image_da

    store_path = test_data_dir / "input" / "2th_cthead1.zarr"
    try:
        from zarr.storage import DirectoryStore

        store = DirectoryStore(store_path, dimension_separator="/")
    except ImportError:
        from zarr.storage import LocalStore

        store = LocalStore(store_path)
    image_ds = xr.open_zarr(store)
    image_da = image_ds["2th_cthead1"]
    result["2th_cthead1"] = image_da

    return result


def verify_against_baseline(dataset_name, baseline_name, multiscale):
    store_path = test_data_dir / f"baseline/{dataset_name}/{baseline_name}"
    try:
        from zarr.storage import DirectoryStore

        store = DirectoryStore(store_path, dimension_separator="/")
    except ImportError:
        from zarr.storage import LocalStore

        store = LocalStore(store_path)
    dt = xr.open_datatree(store, engine="zarr", mode="r")
    xr.testing.assert_equal(dt.ds, multiscale.ds)
    for scale in multiscale.children:
        xr.testing.assert_equal(dt[scale].ds, multiscale[scale].ds)


def store_new_image(dataset_name, baseline_name, multiscale_image):
    """Helper method for writing output results to disk
    for later upload as test baseline"""
    path = test_data_dir / f"baseline/{dataset_name}/{baseline_name}"
    try:
        from zarr.storage import DirectoryStore

        store = DirectoryStore(
            str(path),
            dimension_separator="/",
        )
    except ImportError:
        from zarr.storage import LocalStore

        store = LocalStore(str(path))
    multiscale_image.to_zarr(store, mode="w")
