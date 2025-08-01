import json
from typing import Dict
import urllib3
import asyncio

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

    # Get NGFF metadata
    ngff = None

    # Method 1: Try to access via zarr group attrs (works for both v2 and v3)
    try:
        group = zarr.open_group(store=store, mode="r")
        ngff = dict(group.attrs)
    except Exception:
        pass

    # Method 2: Try zarr v2 dict-like store access
    if ngff is None and hasattr(store, "__getitem__"):
        try:
            # Try consolidated metadata
            if ".zmetadata" in store:
                metadata = json.loads(store[".zmetadata"])["metadata"]
                ngff = metadata[".zattrs"]
            # Try direct attributes
            elif ".zattrs" in store:
                ngff = json.loads(store[".zattrs"])
        except Exception:
            pass

    # Method 3: Try zarr v3 async store API
    if ngff is None and hasattr(store, "get"):
        try:

            async def get_attrs_v3():
                # Get default buffer prototype for v3
                if hasattr(zarr, "buffer"):
                    prototype = zarr.buffer.default_buffer_prototype()
                else:
                    # Fallback for older versions
                    prototype = None

                # Try zarr.json first (v3 format)
                try:
                    if prototype is not None:
                        zarr_json_bytes = await store.get(
                            "zarr.json", prototype=prototype
                        )
                    else:
                        zarr_json_bytes = await store.get("zarr.json")

                    if zarr_json_bytes:
                        if hasattr(zarr_json_bytes, "to_bytes"):
                            data = zarr_json_bytes.to_bytes()
                        else:
                            data = zarr_json_bytes
                        zarr_json = json.loads(data)
                        return zarr_json.get("attributes", {})
                except Exception:
                    pass

                # Try .zattrs (legacy/compatibility)
                try:
                    if prototype is not None:
                        attrs_bytes = await store.get(".zattrs", prototype=prototype)
                    else:
                        attrs_bytes = await store.get(".zattrs")

                    if attrs_bytes:
                        if hasattr(attrs_bytes, "to_bytes"):
                            data = attrs_bytes.to_bytes()
                        else:
                            data = attrs_bytes
                        return json.loads(data)
                except Exception:
                    pass

                return None

            # Run async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                ngff = loop.run_until_complete(get_attrs_v3())
            finally:
                loop.close()
        except Exception:
            pass

    if ngff is None:
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
