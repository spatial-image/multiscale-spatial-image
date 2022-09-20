from typing import Union, Sequence, List, Optional, Dict, Mapping, Any, Tuple
from enum import Enum

from spatial_image import to_spatial_image, SpatialImage  # type: ignore

from dask.array import map_blocks, map_overlap
import numpy as np

from ..multiscale_spatial_image import MultiscaleSpatialImage 

from ._xarray import _downsample_xarray_coarsen
from ._itk import _downsample_itk_bin_shrink, _downsample_itk_gaussian, _downsample_itk_label
from ._support import _align_chunks, _dim_scale_factors, _compute_sigma

class Methods(Enum):
    XARRAY_COARSEN = "xarray.DataArray.coarsen"
    ITK_BIN_SHRINK = "itk.bin_shrink_image_filter"
    ITK_GAUSSIAN = "itk.discrete_gaussian_image_filter"
    ITK_LABEL_GAUSSIAN = "itk.discrete_gaussian_image_filter_label_interpolator"
    DASK_IMAGE_GAUSSIAN = "dask_image.ndfilters.gaussian_filter"


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
    """Generate a multiscale representation of a spatial image.

    Parameters
    ----------

    image : SpatialImage
        The spatial image from which we generate a multi-scale representation.

    scale_factors : int per scale or dict of spatial dimension int's per scale
        Integer scale factors to apply uniformly across all spatial dimension or
        along individual spatial dimensions.
        Examples: [2, 2] or [{'x': 2, 'y': 4 }, {'x': 5, 'y': 10}]

    method : multiscale_spatial_image.Methods, optional
        Method to reduce the input image.

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

    current_input = image.chunk(out_chunks)
    # https://github.com/pydata/xarray/issues/5219
    if "chunks" in current_input.encoding:
        del current_input.encoding["chunks"]
    data_objects = {f"scale0": current_input.to_dataset(name=image.name, promote_attrs=True)}

    if method is None:
        method = Methods.XARRAY_COARSEN

    def compute_input_spacing(input_image):
        '''Helper method to manually compute image spacing. Assumes even spacing along any axis.
           
        input_image: xarray.core.dataarray.DataArray
            The image for which voxel spacings are computed

        result: Dict
            Spacing along each enumerated image axis
            Example {'x': 1.0, 'y': 0.5}
        '''
        return {dim: float(input_image.coords[dim][1]) - float(input_image.coords[dim][0])
                for dim in input_image.dims}

    def compute_output_spacing(input_image, dim_factors):
        '''Helper method to manually compute output image spacing.
           
        input_image: xarray.core.dataarray.DataArray
            The image for which voxel spacings are computed

        dim_factors: Dict
            Shrink ratio along each enumerated axis

        result: Dict
            Spacing along each enumerated image axis
            Example {'x': 2.0, 'y': 1.0}
        '''
        input_spacing = compute_input_spacing(input_image)
        return {dim: input_spacing[dim] * dim_factors[dim] for dim in input_image.dims}

    def compute_output_origin(input_image, dim_factors):    
        '''Helper method to manually compute output image physical offset.
           Note that this method does not account for an image direction matrix.
           
        input_image: xarray.core.dataarray.DataArray
            The image for which voxel spacings are computed

        dim_factors: Dict
            Shrink ratio along each enumerated axis

        result: Dict
            Offset in physical space of first voxel in output image
            Example {'x': 0.5, 'y': 1.0}
        '''
        import math
        image_dims: Tuple[str, str, str, str] = ("x", "y", "z", "t")
            
        input_spacing = compute_input_spacing(input_image)
        input_origin = {dim: float(input_image.coords[dim][0])
                          for dim in image_dims if dim in dim_factors}

        # Index in input image space corresponding to offset after shrink
        input_index = {dim: 0.5 * (dim_factors[dim] - 1)
                                  for dim in image_dims if dim in dim_factors}
        # Translate input index coordinate to offset in physical space
        # NOTE: This method fails to account for direction matrix
        return {dim: input_index[dim] * input_spacing[dim] + input_origin[dim]
                          for dim in image_dims if dim in dim_factors}

    if method is Methods.XARRAY_COARSEN:
        data_objects = _downsample_xarray_coarsen(current_input, default_chunks, out_chunks, scale_factors, data_objects, image.name)
    elif method is Methods.ITK_BIN_SHRINK:
        data_objects = _downsample_itk_bin_shrink(current_input, default_chunks, out_chunks, scale_factors, data_objects, image)
    elif method is Methods.ITK_GAUSSIAN:
        data_objects = _downsample_itk_gaussian(current_input, default_chunks, out_chunks, scale_factors, data_objects, image)
    elif method is Methods.ITK_LABEL_GAUSSIAN:
        data_objects = _downsample_itk_label(current_input, default_chunks, out_chunks, scale_factors, data_objects, image)
    elif method is Methods.DASK_IMAGE_GAUSSIAN:
        import dask_image.ndfilters
        import dask_image.ndinterp

        def get_truncate(xarray_image, sigma_values, truncate_start=4.0) -> float:
            '''Discover truncate parameter yielding a viable kernel width
               for dask_image.ndfilters.gaussian_filter processing. Block overlap
               cannot be greater than image size, so kernel radius is more limited
               for small images. A lower stddev truncation ceiling for kernel
               generation can result in a less precise kernel.

            xarray_image: xarray.core.dataarray.DataArray
               Chunked image to be smoothed

            sigma:values: List
               Gaussian kernel standard deviations in tzyx order

            truncate_start: float
               First truncation value to try.
               "dask_image.ndfilters.gaussian_filter" defaults to 4.0.

            result: float
               Truncation value found to yield largest possible kernel width without
               extending beyond one chunk such that chunked smoothing would fail.
            '''

            from dask_image.ndfilters._gaussian import _get_border

            truncate = truncate_start
            stddev_step = 0.5  # search by stepping down by 0.5 stddev in each iteration

            border = _get_border(xarray_image.data, sigma_values, truncate)
            while any([border_len > image_len for border_len, image_len in zip(border, xarray_image.shape)]):
                truncate = truncate - stddev_step
                if(truncate <= 0.0):
                    break                    
                border = _get_border(xarray_image.data, sigma_values, truncate)

            return truncate

        for factor_index, scale_factor in enumerate(scale_factors):
            dim_factors = _dim_scale_factors(image.dims, scale_factor)
            current_input = _align_chunks(current_input, default_chunks, dim_factors)

            image_dims: Tuple[str, str, str, str] = ("x", "y", "z", "t")
            shrink_factors = [dim_factors[sf] for sf in image_dims if sf in dim_factors]

            # Compute/discover region splitting parameters
            input_spacing = compute_input_spacing(current_input)
            input_spacing = [input_spacing[dim] for dim in image_dims if dim in dim_factors]
            sigma_values = _compute_sigma(input_spacing, shrink_factors)
            truncate = get_truncate(current_input, np.flip(sigma_values))

            # Compute output shape and metadata
            output_shape = [int(image_len / shrink_factor)
                for image_len, shrink_factor in zip(current_input.shape, np.flip(shrink_factors))]
            output_spacing = compute_output_spacing(current_input, dim_factors)
            output_origin = compute_output_origin(current_input, dim_factors)

            blurred_array = dask_image.ndfilters.gaussian_filter(
                image=current_input.data,
                sigma=np.flip(sigma_values), # tzyx order
                mode='nearest',
                truncate=truncate
            )

            # Construct downsample parameters
            image_dimension = len(dim_factors)
            transform = np.eye(image_dimension)
            for dim, shrink_factor in enumerate(np.flip(shrink_factors)):
                transform[dim,dim] = shrink_factor

            downscaled_array = dask_image.ndinterp.affine_transform(
                blurred_array,
                matrix=transform,
                output_shape=output_shape # tzyx order
            ).compute()
            
            downscaled = to_spatial_image(
                downscaled_array,
                dims=image.dims,
                scale=output_spacing,
                translation=output_origin,
                name=current_input.name,
                axis_names={
                    d: image.coords[d].attrs.get("long_name", d) for d in image.dims
                },
                axis_units={
                    d: image.coords[d].attrs.get("units", "") for d in image.dims
                },
                t_coords=image.coords.get("t", None),
                c_coords=image.coords.get("c", None),
            )
            downscaled = downscaled.chunk(out_chunks)
            data_objects[f"scale{factor_index+1}"] = downscaled.to_dataset(
                name=image.name, promote_attrs=True
            )
            current_input = downscaled


    multiscale = MultiscaleSpatialImage.from_dict(
        d=data_objects
    )

    return multiscale
