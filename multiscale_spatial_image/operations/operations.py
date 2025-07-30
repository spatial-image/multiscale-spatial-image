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
    # Extract the first argument as indexers, and pass the rest as keyword arguments
    if args:
        indexers = args[0]
        # Map positional arguments to their parameter names
        reindex_kwargs = {}
        if len(args) > 1:
            reindex_kwargs["method"] = args[1]
        if len(args) > 2:
            reindex_kwargs["tolerance"] = args[2]
        if len(args) > 3:
            reindex_kwargs["copy"] = args[3]
        if len(args) > 4:
            reindex_kwargs["fill_value"] = args[4]
        # Add any additional keyword arguments
        reindex_kwargs.update(kwargs)
        return ds["image"].reindex(indexers, **reindex_kwargs).to_dataset()
    else:
        # Fall back to original behavior if no arguments
        return ds["image"].reindex(**kwargs).to_dataset()
