from multiscale_spatial_image import skip_non_dimension_nodes
import numpy as np
from spatial_image import to_spatial_image
from multiscale_spatial_image import to_multiscale


def test_skip_nodes():
    data = np.zeros((2, 200, 200))
    dims = ("c", "y", "x")
    scale_factors = [2, 2]
    image = to_spatial_image(array_like=data, dims=dims)
    multiscale_img = to_multiscale(image, scale_factors=scale_factors)

    @skip_non_dimension_nodes
    def transpose(ds, *args, **kwargs):
        return ds.transpose(*args, **kwargs)

    for scale in list(multiscale_img.keys()):
        assert multiscale_img[scale]["image"].dims == ("c", "y", "x")

    # applying this function without skipping the root node would fail as the root node does not have dimensions.
    result = multiscale_img.map_over_datasets(transpose, "y", "x", "c")
    for scale in list(result.keys()):
        assert result[scale]["image"].dims == ("y", "x", "c")
