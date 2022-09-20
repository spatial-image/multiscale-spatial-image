_spatial_dims = {"x", "y", "z"}

def _dim_scale_factors(dims, scale_factor):
    if isinstance(scale_factor, int):
        dim = {dim: scale_factor for dim in _spatial_dims.intersection(dims)}
    else:
        dim = scale_factor
    return dim


def _align_chunks(current_input, default_chunks, dim_factors):
    block_0_shape = [c[0] for c in current_input.chunks]

    rechunk = False
    aligned_chunks = {}
    for dim, factor in dim_factors.items():
        dim_index = current_input.dims.index(dim)
        if block_0_shape[dim_index] % factor:
            aligned_chunks[dim] = block_0_shape[dim_index] * factor
            rechunk = True
        else:
            aligned_chunks[dim] = default_chunks[dim]
    if rechunk:
        current_input = current_input.chunk(aligned_chunks)

    return current_input