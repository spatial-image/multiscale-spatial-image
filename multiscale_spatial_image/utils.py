from typing import Callable, Any
from xarray import Dataset
import functools


def skip_non_dimension_nodes(
    func: Callable[[Dataset], Dataset],
) -> Callable[[Dataset], Dataset]:
    """Skip nodes in Datatree that do not contain dimensions.

    This function implements the workaround of https://github.com/pydata/xarray/issues/9693. In particular,
    we need this because of our DataTree representing multiscale image having a root node that does not have
    dimensions. Several functions need to be mapped over the datasets in the datatree that depend on having
    dimensions, e.g. a transpose.
    """

    @functools.wraps(func)
    def _func(ds: Dataset, *args: Any, **kwargs: Any) -> Dataset:
        # check if dimensions are present otherwise return verbatim
        if len(ds.dims) == 0:
            return ds
        return func(ds, *args, **kwargs)

    return _func
