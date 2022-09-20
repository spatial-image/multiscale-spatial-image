from spatial_image import to_spatial_image
from dask.array import map_blocks, map_overlap

from ._support import _align_chunks, _dim_scale_factors

def _downsample_itk_bin_shrink(current_input, default_chunks, out_chunks, scale_factors, data_objects, image):
    import itk

    for factor_index, scale_factor in enumerate(scale_factors):
        dim_factors = _dim_scale_factors(image.dims, scale_factor)
        current_input = _align_chunks(current_input, default_chunks, dim_factors)

        image_dims: Tuple[str, str, str, str] = ("x", "y", "z", "t")
        shrink_factors = [dim_factors[sf] for sf in image_dims if sf in dim_factors]

        block_0_shape = [c[0] for c in current_input.chunks]
        block_0 = current_input[tuple([slice(0, s) for s in block_0_shape])]
        # For consistency for now, do not utilize direction until there is standardized support for
        # direction cosines / orientation in OME-NGFF
        block_0.attrs.pop("direction", None)
        block_input = itk.image_from_xarray(block_0)
        filt = itk.BinShrinkImageFilter.New(
            block_input, shrink_factors=shrink_factors
        )
        filt.UpdateOutputInformation()
        block_output = filt.GetOutput()
        scale = {
            image_dims[i]: s for (i, s) in enumerate(block_output.GetSpacing())
        }
        translation = {
            image_dims[i]: s for (i, s) in enumerate(block_output.GetOrigin())
        }
        dtype = block_output.dtype
        output_chunks = list(current_input.chunks)
        for i, c in enumerate(output_chunks):
            output_chunks[i] = [
                block_output.shape[i],
            ] * len(c)

        block_neg1_shape = [c[-1] for c in current_input.chunks]
        block_neg1 = current_input[tuple([slice(0, s) for s in block_neg1_shape])]
        block_neg1.attrs.pop("direction", None)
        block_input = itk.image_from_xarray(block_neg1)
        filt = itk.BinShrinkImageFilter.New(
            block_input, shrink_factors=shrink_factors
        )
        filt.UpdateOutputInformation()
        block_output = filt.GetOutput()
        for i, c in enumerate(output_chunks):
            output_chunks[i][-1] = block_output.shape[i]
            output_chunks[i] = tuple(output_chunks[i])
        output_chunks = tuple(output_chunks)

        downscaled_array = map_blocks(
            itk.bin_shrink_image_filter,
            current_input.data,
            shrink_factors=shrink_factors,
            dtype=dtype,
            chunks=output_chunks,
        )
        downscaled = to_spatial_image(
            downscaled_array,
            dims=image.dims,
            scale=scale,
            translation=translation,
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