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
def reindex_data_arrays(ds: Dataset, *args: Any, **kwargs: Any) -> Dataset:
    return ds["image"].reindex(*args, **kwargs).to_dataset()
