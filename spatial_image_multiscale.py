"""spatial-image-multiscale

Generate a multiscale spatial image."""

__version__ = "0.1.0"

from typing import Union, Sequence, Mapping, Hashable, List, Optional
from enum import Enum

import xarray as xr
import numpy as np

_spatial_dims = {"x", "y", "z"}


class Method(Enum):
    XARRAY_COARSEN = "xarray.coarsen"


def to_multiscale(
    image: xr.DataArray,
    scale_factors: Sequence[Union[Mapping[Hashable, int], int]],
    method: Optional[Method] = None,
) -> List[xr.DataArray]:
    """Generate a multiscale representation of a spatial image.

    Parameters
    ----------

    image : xarray.DataArray
        The spatial image from which we generate a multi-scale representation.

    scale_factors : int per scale or spatial dimension int's per scale
        Sequence of integer scale factors to apply along each spatial dimension.

    method : spatial_image_multiscale.Method, optional
        Method to reduce the input image.

    Returns
    -------

    result : list of xr.DataArray's
        Multiscale representation. The input image, is returned as in the first
        element. Subsequent elements are downsampled following the provided
        scale_factors.
    """

    result = [image]
    current_input = image
    for scale_factor in scale_factors:
        if isinstance(scale_factor, int):
            dim = {dim: scale_factor for dim in _spatial_dims.intersection(image.dims)}
        else:
            dim = scale_factor
        downscaled = current_input.coarsen(
            dim=dim, boundary="trim", side="right"
        ).mean()
        result.append(downscaled)
        current_input = downscaled

    return result
