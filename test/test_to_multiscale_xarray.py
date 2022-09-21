from multiscale_spatial_image import Methods, to_multiscale

from ._data import input_images, verify_against_baseline, store_new_image

def test_isotropic_scale_factors(input_images):
    dataset_name = "cthead1"
    image = input_images[dataset_name]
    baseline_name = "2_4/XARRAY_COARSEN"
    multiscale = to_multiscale(image, [2, 4], method=Methods.XARRAY_COARSEN)
    verify_against_baseline(dataset_name, baseline_name, multiscale)

    dataset_name = "cthead1"
    image = input_images[dataset_name]
    baseline_name = "2_3/XARRAY_COARSEN"
    multiscale = to_multiscale(image, [2, 3], method=Methods.XARRAY_COARSEN)
    verify_against_baseline(dataset_name, baseline_name, multiscale)

    dataset_name = "small_head"
    image = input_images[dataset_name]
    baseline_name = "2_3_4/XARRAY_COARSEN"
    multiscale = to_multiscale(image, [2, 3, 4], method=Methods.XARRAY_COARSEN)
    verify_against_baseline(dataset_name, baseline_name, multiscale)

def test_anisotropic_scale_factors(input_images):
    dataset_name = "cthead1"
    image = input_images[dataset_name]
    scale_factors = [{"x": 2, "y": 4}, {"x": 1, "y": 2}]
    multiscale = to_multiscale(image, scale_factors, method=Methods.XARRAY_COARSEN)
    baseline_name = "x2y4_x1y2/XARRAY_COARSEN"
    verify_against_baseline(dataset_name, baseline_name, multiscale)
    # Test default method: Methods.XARRAY_COARSEN
    multiscale = to_multiscale(image, scale_factors)
    verify_against_baseline(dataset_name, baseline_name, multiscale)

    dataset_name = "small_head"
    image = input_images[dataset_name]
    scale_factors = [
        {"x": 3, "y": 2, "z": 4},
        {"x": 2, "y": 2, "z": 2},
        {"x": 1, "y": 2, "z": 1},
    ]
    multiscale = to_multiscale(image, scale_factors, method=Methods.XARRAY_COARSEN)
    baseline_name = "x3y2z4_x2y2z2_x1y2z1/XARRAY_COARSEN"
    verify_against_baseline(dataset_name, baseline_name, multiscale)