from typing import Union, Sequence, List, Optional, Dict, Mapping, Any, Tuple

from spatial_image import to_spatial_image

from .to_multiscale import to_multiscale, Methods
from ..multiscale_spatial_image import MultiscaleSpatialImage 

def itk_image_to_multiscale(
    image,
    scale_factors: Sequence[Union[Dict[str, int], int]],
    anatomical_axes: bool = False,
    axis_names: List[str] = None,
    axis_units: List[str] = None,
    name: List[str] = None,
    method: Optional[Methods] = None,
    chunks: Optional[
        Union[
            int,
            Tuple[int, ...],
            Tuple[Tuple[int, ...], ...],
            Mapping[Any, Union[None, int, Tuple[int, ...]]],
        ]
    ] = None) -> MultiscaleSpatialImage:

    import itk
    import numpy as np

    if not name:
        object_name = image.GetObjectName().strip()
        if object_name and not object_name.isspace():
            name = object_name
        else:
            name = 'image'

    # Handle anatomical axes
    if anatomical_axes and (axis_names is None):
        axis_names = {"x": "right-left", "y": "anterior-posterior", "z": "inferior-superior"}
    
    # Orient 3D image so that direction is identity wrt RAI coordinates
    image_dimension = image.GetImageDimension()
    input_direction = np.array(image.GetDirection())
    if anatomical_axes and image_dimension == 3 and not (np.eye(image_dimension) == input_direction).all():
        desired_orientation = itk.SpatialOrientationEnums.ValidCoordinateOrientations_ITK_COORDINATE_ORIENTATION_RAI
        oriented_image = itk.orient_image_filter(image, use_image_direction=True, desired_coordinate_orientation=desired_orientation)

    elif anatomical_axes and image_dimension != 3:
        raise ValueError(f'Cannot use anatomical axes for input image of size {image_dimension}')
        
    image_da = itk.xarray_from_image(image)
    image_da.name = name

    image_dims: Tuple[str, str, str, str] = ("x", "y", "z", "t") # ITK dims are in xyzt order
    scale = {image_dims[i]: s for (i, s) in enumerate(image.GetSpacing())}
    translation = {image_dims[i]: s for (i, s) in enumerate(image.GetOrigin())}

    spatial_image = to_spatial_image(image_da.data,
                                     dims=image_da.dims,
                                     scale=scale,
                                     translation=translation,
                                     name=name,
                                     axis_names=axis_names,
                                     axis_units=axis_units)

    return to_multiscale(spatial_image, 
                         scale_factors, 
                         method=method, 
                         chunks=chunks)
