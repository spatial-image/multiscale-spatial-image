from ._support import _align_chunks, _dim_scale_factors

def _downsample_xarray_coarsen(current_input, default_chunks, out_chunks, scale_factors, data_objects, name):
    for factor_index, scale_factor in enumerate(scale_factors):
        dim_factors = _dim_scale_factors(current_input.dims, scale_factor)
        current_input = _align_chunks(current_input, default_chunks, dim_factors)

        downscaled = (
            current_input.coarsen(dim=dim_factors, boundary="trim", side="right")
            .mean()
            .astype(current_input.dtype)
        )

        downscaled = downscaled.chunk(out_chunks)

        data_objects[f"scale{factor_index+1}"] = downscaled.to_dataset(
            name=name, promote_attrs=True
        )
        current_input = downscaled

    return data_objects