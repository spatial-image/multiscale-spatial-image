"""spatial-image-multiscale

Generate a multiscale spatial image."""

__version__ = "0.0.2"

from typing import Union, Sequence, List, Optional
from enum import Enum

import xarray as xr
import numpy as np

class Method(Enum):
    XARRAY_COARSEN = "xarray.coarsen"

def to_multiscale(image: xr.DataArray,
        scale_factors: Sequence[Union[Sequence[int], int]],
        method: Optional[Method] = None,
        ranges: bool = False) -> List[xr.DataArray]:
    """Generate a multiscale representation of a spatial image.

    Parameters
    ----------

    image : xarray.DataArray
        The spatial image from which we generate a multi-scale representation.

    scale_factors : int per scale or spatial dimension int's per scale
        Sequence of integer scale factors to apply along each spatial dimension.

    method : spatial_image_multiscale.Method, optional
        Method to reduce the input image.

    ranges : bool
        Compute ranges of every image component, output on the 'ranges' attr.


    Returns
    -------

    result : list of xr.DataArray's
        Multiscale representation. The input image, is returned as in the first
        element. Subsequent elements are downsampled following the provided
        scale_factors.
    """

    result = [image]

    return result
