import pytest
import numpy as np
from spatial_image import to_spatial_image
from multiscale_spatial_image import to_multiscale


@pytest.fixture()
def multiscale_data():
    data = np.zeros((3, 200, 200))
    dims = ("c", "y", "x")
    scale_factors = [2, 2]
    image = to_spatial_image(array_like=data, dims=dims)
    return to_multiscale(image, scale_factors=scale_factors)
