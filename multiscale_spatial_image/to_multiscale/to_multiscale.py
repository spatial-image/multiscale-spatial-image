from typing import Union, Sequence, Optional, Dict, Mapping, Any, Tuple
from enum import Enum
import warnings

from spatial_image import SpatialImage, to_spatial_image  # type: ignore
import ngff_zarr as nz
from xarray import DataTree
from .._docs import inject_docs

from ngff_zarr import Methods as NZMethods

class Methods(Enum):
    XARRAY_COARSEN = "xarray_coarsen"
    ITK_BIN_SHRINK = "itk_bin_shrink"
    ITK_GAUSSIAN = "itk_gaussian"
    ITK_LABEL_GAUSSIAN = "itk_label_gaussian"
    DASK_IMAGE_GAUSSIAN = "dask_image_gaussian"
    DASK_IMAGE_MODE = "dask_image_mode"
    DASK_IMAGE_NEAREST = "dask_image_nearest"

MSI_METHOD_MAPPING = {
    'xarray_coarsen': NZMethods.ITK_BIN_SHRINK,
    'itk_bin_shrink': NZMethods.ITK_BIN_SHRINK,
    'itk_gaussian': NZMethods.ITK_GAUSSIAN,
    'itk_label_gaussian': NZMethods.ITKWASM_LABEL_IMAGE,
    'dask_image_gaussian': NZMethods.DASK_IMAGE_GAUSSIAN,
    'dask_image_mode': NZMethods.DASK_IMAGE_MODE,
    'dask_image_nearest': NZMethods.DASK_IMAGE_NEAREST,
}

@inject_docs(m=NZMethods)
def to_multiscale(
    image: SpatialImage,
    scale_factors: Sequence[Union[Dict[str, int], int]],
    method: Optional[NZMethods] = None,
    chunks: Optional[
        Union[
            int,
            Tuple[int, ...],
            Tuple[Tuple[int, ...], ...],
            Mapping[Any, Union[None, int, Tuple[int, ...]]],
        ]
    ] = None,
) -> DataTree:
    """\
    Generate a multiscale representation of a spatial image.

    Parameters
    ----------

    image : SpatialImage
        The spatial image from which we generate a multi-scale representation.

    scale_factors : int per scale or dict of spatial dimension int's per scale
        Integer scale factors to apply uniformly across all spatial dimension or
        along individual spatial dimensions. The scale factors need to be passed
        in **ascending order**.
        This will work: [2, 4] or [{{'x': 2, 'y': 4}}, {{'x': 5, 'y': 10}}].
        This will not work: [4, 2] or [{{'x': 5, 'y': 10}}, {{'x': 2, 'y': 4}}].

    method : multiscale_spatial_image.Methods, optional
        Method to reduce the input image. Available methods are the following:

        - `{m.ITK_BIN_SHRINK.value!r}` - Use ITK ShrinkImageFilter to downsample the image (DEFAULT).
        - `{m.ITKWASM_BIN_SHRINK.value!r}` - Use ITKWASM BinShrinkImageFilter to downsample the image.
        - `{m.ITK_GAUSSIAN.value!r}` - Use ITK GaussianImageFilter to downsample the image.
        - `{m.ITKWASM_GAUSSIAN.value!r}` - Use ITK ShrinkImageFilter to downsample the image.
        - `{m.ITKWASM_LABEL_IMAGE.value!r}` - Use ITK LabelGaussianImageFilter to downsample the image.
        - `{m.DASK_IMAGE_GAUSSIAN.value!r}` - Use dask-image gaussian_filter to downsample the image.
        - `{m.DASK_IMAGE_MODE.value!r}` - Use dask-image mode_filter to downsample the image.
        - `{m.DASK_IMAGE_NEAREST.value!r}` - Use dask-image zoom to downsample the image.

    chunks : xarray Dask array chunking specification, optional
        Specify the chunking used in each output scale.

    Returns
    -------

    result : DataTree
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
    if "c" in image.dims:
        default_chunks["c"] = 1
    out_chunks = chunks
    if out_chunks is None:
        out_chunks = default_chunks

    # check for valid scale factors
    current_shape = {
        d: s for (d, s) in zip(image.dims, image.shape) if d not in {"t", "c"}
    }

    for scale_factor in scale_factors:
        if isinstance(scale_factor, dict):
            current_shape = {
                k: (current_shape[k] / s) for (k, s) in scale_factor.items()
            }
        elif isinstance(scale_factor, int):
            current_shape = {k: (s / scale_factor) for (k, s) in current_shape.items()}
        for k, v in current_shape.items():
            if v < 1:
                raise ValueError(
                    f"Scale factor {scale_factor} is incompatible with image shape {image.shape} along dimension `{k}`."
                )

    current_input = image.chunk(out_chunks)
    # https://github.com/pydata/xarray/issues/5219
    if "chunks" in current_input.encoding:
        del current_input.encoding["chunks"]

    # get metadata from the image
    axes_names = {d: image[d].attrs.get('long_name', None) for d in image.dims}
    axes_names = None if all(v is None for v in axes_names.values()) else axes_names

    axes_units = {d: image[d].attrs.get('units', None) for d in image.dims}
    axes_units = None if all(v == '' or v is None for v in axes_units.values()) else axes_units

    spatial_dims = [d for d in image.dims if d in {"z", "y", "x"}]
    scale = {
        d: float(image[d][1] - image[d][0]) if isinstance(image[d], Sequence) and len(image[d]) > 1 else 1.0 for d in spatial_dims
    }
    translation = {
        d: float(image[d][0]) if len(image[d]) > 0 else 0.0 for d in spatial_dims
    }

    ngff_image = nz.to_ngff_image(
        current_input,
        dims=image.dims,
        name=image.name,
        scale=scale,
        translation=translation,
    )

    if method is None:
        method = nz.Methods.ITK_BIN_SHRINK
    else:
        if isinstance(method, Enum) and method.value in MSI_METHOD_MAPPING:
            warnings.warn(
                f"Method {method.name} is deprecated, use {MSI_METHOD_MAPPING[method.value]} instead.",
                DeprecationWarning,
            )
            method = MSI_METHOD_MAPPING[method.value]

    multiscales = nz.to_multiscales(
        ngff_image,
        scale_factors=scale_factors,
        method=method,
        chunks=out_chunks,
    )

    data_objects = {}
    for factor, img in enumerate(multiscales.images):
        si = to_spatial_image(
            img.data,
            dims=img.dims,
            scale=img.scale,
            axis_names=axes_names,
            axis_units=axes_units,
            translation=img.translation,
            )
        data_objects[f"scale{factor}"] = si.to_dataset(name=image.name, promote_attrs=True)

    multiscale = DataTree.from_dict(data_objects)

    return multiscale
