import xarray as xr

from multiscale_spatial_image import Methods, to_multiscale 

from ._data import input_images

def test_base_scale(input_images):
    image = input_images["cthead1"]

    multiscale = to_multiscale(image, [])
    xr.testing.assert_equal(image, multiscale['scale0'].ds["cthead1"])

    image = input_images["small_head"]
    multiscale = to_multiscale(image, [])
    xr.testing.assert_equal(image, multiscale['scale0'].ds["small_head"])