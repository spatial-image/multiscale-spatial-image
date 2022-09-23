from spatial_image import to_spatial_image
from dask.array import map_blocks, map_overlap
import numpy as np

from ._support import _align_chunks, _dim_scale_factors, _compute_sigma

def _compute_input_spacing(input_image):
    '''Helper method to manually compute image spacing. Assumes even spacing along any axis.
        
    input_image: xarray.core.dataarray.DataArray
        The image for which voxel spacings are computed

    result: Dict
        Spacing along each enumerated image axis
        Example {'x': 1.0, 'y': 0.5}
    '''
    return {dim: float(input_image.coords[dim][1]) - float(input_image.coords[dim][0])
            for dim in input_image.dims}

def _compute_output_spacing(input_image, dim_factors):
    '''Helper method to manually compute output image spacing.
        
    input_image: xarray.core.dataarray.DataArray
        The image for which voxel spacings are computed

    dim_factors: Dict
        Shrink ratio along each enumerated axis

    result: Dict
        Spacing along each enumerated image axis
        Example {'x': 2.0, 'y': 1.0}
    '''
    input_spacing = _compute_input_spacing(input_image)
    return {dim: input_spacing[dim] * dim_factors[dim] for dim in input_image.dims}

def _compute_output_origin(input_image, dim_factors):    
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
        
    input_spacing = _compute_input_spacing(input_image)
    input_origin = {dim: float(input_image.coords[dim][0])
                      for dim in input_image.dims if dim in dim_factors}

    # Index in input image space corresponding to offset after shrink
    input_index = {dim: 0.5 * (dim_factors[dim] - 1)
                              for dim in input_image.dims if dim in dim_factors}
    # Translate input index coordinate to offset in physical space
    # NOTE: This method fails to account for direction matrix
    return {dim: input_index[dim] * input_spacing[dim] + input_origin[dim]
                      for dim in input_image.dims if dim in dim_factors}

def _get_truncate(xarray_image, sigma_values, truncate_start=4.0) -> float:
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

def _downsample_dask_image(current_input, default_chunks, out_chunks, scale_factors, data_objects, image, label=False):
    import dask_image.ndfilters
    import dask_image.ndinterp

    for factor_index, scale_factor in enumerate(scale_factors):
        dim_factors = _dim_scale_factors(image.dims, scale_factor)
        current_input = _align_chunks(current_input, default_chunks, dim_factors)

        shrink_factors = []
        for dim in image.dims:
            if dim in dim_factors:
                shrink_factors.append(dim_factors[dim])
            else:
                shrink_factors.append(1)

        # Compute/discover region splitting parameters
        input_spacing = _compute_input_spacing(current_input)

        # Compute output shape and metadata
        output_shape = [int(image_len / shrink_factor)
            for image_len, shrink_factor in zip(current_input.shape, shrink_factors)]
        output_spacing = _compute_output_spacing(current_input, dim_factors)
        output_origin = _compute_output_origin(current_input, dim_factors)

        if label == 'mode':
            def largest_mode(arr):
                values, counts = np.unique(arr, return_counts=True)
                m = counts.argmax()
                return values[m]
            size = tuple(shrink_factors)
            blurred_array = dask_image.ndfilters.generic_filter(
                image=current_input.data,
                function=largest_mode,
                size=size,
                mode='nearest',
            )
        elif label == 'nearest':
            blurred_array = current_input.data
        else:
            input_spacing_list = [input_spacing[dim] for dim in image.dims]
            sigma_values = _compute_sigma(input_spacing_list, shrink_factors)
            truncate = _get_truncate(current_input, sigma_values)

            blurred_array = dask_image.ndfilters.gaussian_filter(
                image=current_input.data,
                sigma=sigma_values, # tzyx order
                mode='nearest',
                truncate=truncate
            )

        # Construct downsample parameters
        image_dimension = len(dim_factors)
        transform = np.eye(image_dimension)
        for dim, shrink_factor in enumerate(shrink_factors):
            transform[dim,dim] = shrink_factor
        if label:
            order = 0
        else:
            order = 1

        downscaled_array = dask_image.ndinterp.affine_transform(
            blurred_array,
            matrix=transform,
            order=order,
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

    return data_objects
