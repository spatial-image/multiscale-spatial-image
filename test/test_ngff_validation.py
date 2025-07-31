import json
from typing import Dict
import urllib3

from referencing import Registry, Resource
from jsonschema import Draft202012Validator

from xarray import DataTree

from multiscale_spatial_image import to_multiscale, MultiscaleSpatialImage
from spatial_image import to_spatial_image
import numpy as np
import zarr

http = urllib3.PoolManager()

ngff_uri = "https://ngff.openmicroscopy.org"


def load_schema(version: str = "0.4", strict: bool = False) -> Dict:
    strict_str = ""
    if strict:
        strict_str = "strict_"
    response = http.request(
        "GET", f"{ngff_uri}/{version}/schemas/{strict_str}image.schema"
    )
    schema = json.loads(response.data.decode())
    return schema


def check_valid_ngff(multiscale: DataTree):
    store = zarr.storage.MemoryStore()
    assert isinstance(multiscale.msi, MultiscaleSpatialImage)
    multiscale.msi.to_zarr(store, compute=True)

    # Handle consolidate_metadata for both zarr v2 and v3
    try:
        # Zarr v3
        zarr.consolidate_metadata(store)
    except AttributeError:
        # Zarr v2
        zarr.convenience.consolidate_metadata(store)

    # Get metadata - handle both v2 and v3 formats
    try:
        # Try zarr v2 format first
        metadata = json.loads(store.get(".zmetadata"))["metadata"]
        ngff = metadata[".zattrs"]
    except (KeyError, TypeError):
        # Try zarr v3 format
        # In v3, consolidated metadata might be stored differently
        zarr_json = store.get("zarr.json")
        if zarr_json:
            root_attrs = json.loads(zarr_json).get("attributes", {})
            ngff = root_attrs
        else:
            # Fallback: try to get attributes directly
            attrs_bytes = store.get(".zattrs")
            if attrs_bytes:
                ngff = json.loads(attrs_bytes)
            else:
                raise ValueError("Could not find NGFF metadata in store")

    image_schema = load_schema(version="0.4", strict=False)
    # strict_image_schema = load_schema(version="0.4", strict=True)
    registry = Registry().with_resource(
        ngff_uri, resource=Resource.from_contents(image_schema)
    )
    validator = Draft202012Validator(image_schema, registry=registry)
    # registry_strict = Registry().with_resource(ngff_uri, resource=Resource.from_contents(strict_image_schema))
    # strict_validator = Draft202012Validator(strict_schema, registry=registry_strict)

    validator.validate(ngff)
    # Need to add NGFF metadata property
    # strict_validator.validate(ngff)


def test_y_x_valid_ngff():
    array = np.random.random((32, 16))
    image = to_spatial_image(array)
    multiscale = to_multiscale(image, [2, 4])

    check_valid_ngff(multiscale)


def test_z_y_x_valid_ngff():
    array = np.random.random((32, 32, 16))
    image = to_spatial_image(array)
    multiscale = to_multiscale(image, [2, 4])

    check_valid_ngff(multiscale)


def test_z_y_x_c_valid_ngff():
    array = np.random.random((32, 32, 16, 3))
    image = to_spatial_image(array)
    multiscale = to_multiscale(image, [2, 4])

    check_valid_ngff(multiscale)


def test_t_z_y_x_c_valid_ngff():
    array = np.random.random((2, 32, 32, 16, 3))
    image = to_spatial_image(array)
    multiscale = to_multiscale(image, [2, 4])

    check_valid_ngff(multiscale)
