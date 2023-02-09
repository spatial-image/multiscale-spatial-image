from typing import Union, Sequence, List, Optional, Dict, Mapping, Any, Tuple
from enum import Enum

from spatial_image import to_spatial_image, SpatialImage  # type: ignore

from dask.array import map_blocks, map_overlap
import numpy as np

from ..multiscale_spatial_image import MultiscaleSpatialImage 

from ._xarray import _downsample_xarray_coarsen
from ._itk import _downsample_itk_bin_shrink, _downsample_itk_gaussian, _downsample_itk_label
from ._dask_image import _downsample_dask_image
from .._docs import inject_docs

class Methods(Enum):
    XARRAY_COARSEN = "xarray_coarsen"
    ITK_BIN_SHRINK = "itk_bin_shrink"
    ITK_GAUSSIAN = "itk_gaussian"
    ITK_LABEL_GAUSSIAN = "itk_label_gaussian"
    DASK_IMAGE_GAUSSIAN = "dask_image_gaussian"
    DASK_IMAGE_MODE = "dask_image_mode"
    DASK_IMAGE_NEAREST = "dask_image_nearest"

@inject_docs(m=Methods)
def to_multiscale(
    image: SpatialImage,
    scale_factors: Sequence[Union[Dict[str, int], int]],
    method: Optional[Methods] = None,
    chunks: Optional[
        Union[
            int,
            Tuple[int, ...],
            Tuple[Tuple[int, ...], ...],
            Mapping[Any, Union[None, int, Tuple[int, ...]]],
        ]
    ] = None,
) -> MultiscaleSpatialImage:
    """\
    Generate a multiscale representation of a spatial image.

    Parameters
    ----------

    image : SpatialImage
        The spatial image from which we generate a multi-scale representation.

    scale_factors : int per scale or dict of spatial dimension int's per scale
        Integer scale factors to apply uniformly across all spatial dimension or
        along individual spatial dimensions.
        Examples: [2, 2] or [{{'x': 2, 'y': 4 }}, {{'x': 5, 'y': 10}}]

    method : multiscale_spatial_image.Methods, optional
        Method to reduce the input image. Available methods are the following:

        - `{m.XARRAY_COARSEN.value!r}` - Use xarray coarsen to downsample the image.
        - `{m.ITK_BIN_SHRINK.value!r}` - Use ITK BinShrinkImageFilter to downsample the image.
        - `{m.ITK_GAUSSIAN.value!r}` - Use ITK GaussianImageFilter to downsample the image.
        - `{m.ITK_LABEL_GAUSSIAN.value!r}` - Use ITK LabelGaussianImageFilter to downsample the image.
        - `{m.DASK_IMAGE_GAUSSIAN.value!r}` - Use dask-image gaussian_filter to downsample the image.
        - `{m.DASK_IMAGE_MODE.value!r}` - Use dask-image mode_filter to downsample the image.
        - `{m.DASK_IMAGE_NEAREST.value!r}` - Use dask-image zoom to downsample the image.

    chunks : xarray Dask array chunking specification, optional
        Specify the chunking used in each output scale.

    Returns
    -------

    result : MultiscaleSpatialImage
        Multiscale representation. An xarray DataTree where each node is a SpatialImage Dataset
        named by the integer scale.  Increasing scales are downscaled versions of the input image.
    """

    # IPFS and visualization friendly default chunks
    if "z" in image.dims:
        default_chunks = 64
    else:
        default_chunks = 256
    default_chunks = {d: default_chunks for d in image.dims}
    if "t" in image.dims:
        default_chunks["t"] = 1
    out_chunks = chunks
    if out_chunks is None:
        out_chunks = default_chunks
    
    # check for valid scale factors
    current_shape = {d:s for (d, s) in zip(image.dims, image.shape) if d not in {"t", "c"}}

    for scale_factor in scale_factors:
        if isinstance(scale_factor, dict):
            current_shape = {k: (current_shape[k] / s) for (k, s) in scale_factor.items()}
        elif isinstance(scale_factor, int):
            current_shape = {k: (s / scale_factor) for (k, s) in current_shape.items()}
        for k,v in current_shape.items():
            if v < 1:
                raise ValueError(f"Scale factor {scale_factor} is incompatible with image shape {image.shape} along dimension `{k}`.")

    current_input = image.chunk(out_chunks)
    # https://github.com/pydata/xarray/issues/5219
    if "chunks" in current_input.encoding:
        del current_input.encoding["chunks"]
    data_objects = {f"scale0": current_input.to_dataset(name=image.name, promote_attrs=True)}

    if method is None:
        method = Methods.XARRAY_COARSEN

    if method is Methods.XARRAY_COARSEN:
        data_objects = _downsample_xarray_coarsen(current_input, default_chunks, out_chunks, scale_factors, data_objects, image.name)
    elif method is Methods.ITK_BIN_SHRINK:
        data_objects = _downsample_itk_bin_shrink(current_input, default_chunks, out_chunks, scale_factors, data_objects, image)
    elif method is Methods.ITK_GAUSSIAN:
        data_objects = _downsample_itk_gaussian(current_input, default_chunks, out_chunks, scale_factors, data_objects, image)
    elif method is Methods.ITK_LABEL_GAUSSIAN:
        data_objects = _downsample_itk_label(current_input, default_chunks, out_chunks, scale_factors, data_objects, image)
    elif method is Methods.DASK_IMAGE_GAUSSIAN:
        data_objects = _downsample_dask_image(current_input, default_chunks, out_chunks, scale_factors, data_objects, image, label=False)
    elif method is Methods.DASK_IMAGE_NEAREST:
        data_objects = _downsample_dask_image(current_input, default_chunks, out_chunks, scale_factors, data_objects, image, label='nearest')
    elif method is Methods.DASK_IMAGE_MODE:
        data_objects = _downsample_dask_image(current_input, default_chunks, out_chunks, scale_factors, data_objects, image, label='mode')

    multiscale = MultiscaleSpatialImage.from_dict(
        d=data_objects
    )

    return multiscale
