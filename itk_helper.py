def xarray_from_image(l_image: "itkt.ImageOrImageSource", view:bool=False) -> "xr.DataArray":
    """Convert an itk.Image to an xarray.DataArray.

    TODO remove itk_helper when itk>5.3rc04 is available.
    itk_helper is a workaround for memory reference fix in itk>5.3rc04.
    xarray_from_image is copied from itk.

    Origin and spacing metadata is preserved in the xarray's coords. The
    Direction is set in the `direction` attribute.
    Dims are labeled as `x`, `y`, `z`, `t`, and `c`.

    view may be set to True to get an xarray referencing the ITK image data container
    rather than an entirely new copy. This is best used in the narrow case where
      1. poor copy performance impacts larger operations, such as for a large image, and
      2. the underlying ITK image will not release its data during the xarray lifetime.
    In most cases view should be set to False so that the lifetime of the xarray data
    is independent of the ITK pipeline.

    This interface is and behavior is experimental and is subject to possible
    future changes."""
    import xarray as xr
    import itk
    import numpy as np

    # Fixed from itk<=v5.3rc04
    if view:
        array = itk.array_view_from_image(l_image)
    else:
        array = itk.array_from_image(l_image)

    l_spacing = itk.spacing(l_image)
    l_origin = itk.origin(l_image)
    l_size = itk.size(l_image)
    direction = np.flip(itk.array_from_matrix(l_image.GetDirection()))
    image_dimension = l_image.GetImageDimension()

    image_dims: Tuple[str, str, str, str] = ("x", "y", "z", "t")
    coords = {}
    for l_index, dim in enumerate(image_dims[:image_dimension]):
        coords[dim] = np.linspace(
            l_origin[l_index],
            l_origin[l_index] + (l_size[l_index] - 1) * l_spacing[l_index],
            l_size[l_index],
            dtype=np.float64,
        )

    dims = list(reversed(image_dims[:image_dimension]))
    components = l_image.GetNumberOfComponentsPerPixel()
    if components > 1:
        dims.append("c")
        coords["c"] = np.arange(components, dtype=np.uint32)

    direction = np.flip(itk.array_from_matrix(l_image.GetDirection()))
    attrs = {"direction": direction}
    metadata = dict(l_image)
    ignore_keys = {"direction", "origin", "spacing"}
    for key in metadata:
        if not key in ignore_keys:
            attrs[key] = metadata[key]
    name = "image"
    if l_image.GetObjectName():
        name = l_image.GetObjectName()
    data_array = xr.DataArray(
        array, name=name, dims=dims, coords=coords, attrs=attrs
    )
    return data_array