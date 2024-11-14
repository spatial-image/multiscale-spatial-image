from multiscale_spatial_image.utils import skip_non_dimension_nodes
from xarray import Dataset
from typing import Any


@skip_non_dimension_nodes
def assign_coords(ds: Dataset, *args: Any, **kwargs: Any) -> Dataset:
    return ds.assign_coords(*args, **kwargs)


@skip_non_dimension_nodes
def transpose(ds: Dataset, *args: Any, **kwargs: Any) -> Dataset:
    return ds.transpose(*args, **kwargs)


@skip_non_dimension_nodes
def reindex(ds: Dataset, *args: Any, **kwargs: Any) -> Dataset:
    # A copy is required as a dataset view as used in map_over_datasets is not mutable
    # TODO: Check whether setting item on wrapping datatree node would be better than copy or this can be dropped.
    ds_copy = ds.copy()
    ds_copy["image"] = ds_copy["image"].reindex(*args, **kwargs)
    return ds_copy
