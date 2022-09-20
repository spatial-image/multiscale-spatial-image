from multiscale_spatial_image import Methods, to_multiscale, itk_image_to_multiscale

from ._data import input_images, verify_against_baseline, store_new_image

def test_isotropic_scale_factors(input_images):
    dataset_name = "cthead1"
    image = input_images[dataset_name]
    multiscale = to_multiscale(image, [2, 4], method=Methods.ITK_BIN_SHRINK)
    baseline_name = "2_4/ITK_BIN_SHRINK"
    verify_against_baseline(dataset_name, baseline_name, multiscale)

    dataset_name = "cthead1"
    image = input_images[dataset_name]
    multiscale = to_multiscale(image, [2, 3], method=Methods.ITK_BIN_SHRINK)
    baseline_name = "2_3/ITK_BIN_SHRINK"
    verify_against_baseline(dataset_name, baseline_name, multiscale)

    dataset_name = "small_head"
    image = input_images[dataset_name]
    multiscale = to_multiscale(image, [2, 3, 4], method=Methods.ITK_BIN_SHRINK)
    baseline_name = "2_3_4/ITK_BIN_SHRINK"
    verify_against_baseline(dataset_name, baseline_name, multiscale)

    
def test_gaussian_isotropic_scale_factors(input_images):
    dataset_name = "cthead1"
    image = input_images[dataset_name]
    baseline_name = "2_4/ITK_GAUSSIAN"
    multiscale = to_multiscale(image, [2, 4], method=Methods.ITK_GAUSSIAN)
    verify_against_baseline(dataset_name, baseline_name, multiscale)

    dataset_name = "cthead1"
    image = input_images[dataset_name]
    baseline_name = "2_3/ITK_GAUSSIAN"
    multiscale = to_multiscale(image, [2, 3], method=Methods.ITK_GAUSSIAN)
    verify_against_baseline(dataset_name, baseline_name, multiscale)

    dataset_name = "small_head"
    image = input_images[dataset_name]
    baseline_name = "2_3_4/ITK_GAUSSIAN"
    multiscale = to_multiscale(image, [2, 3, 4], method=Methods.ITK_GAUSSIAN)
    verify_against_baseline(dataset_name, baseline_name, multiscale)


def test_label_gaussian_isotropic_scale_factors(input_images):
    dataset_name = "2th_cthead1"
    image = input_images[dataset_name]
    baseline_name = "2_4/ITK_LABEL_GAUSSIAN"
    multiscale = to_multiscale(image, [2, 4], method=Methods.ITK_LABEL_GAUSSIAN)
    verify_against_baseline(dataset_name, baseline_name, multiscale)

    dataset_name = "2th_cthead1"
    image = input_images[dataset_name]
    baseline_name = "2_3/ITK_LABEL_GAUSSIAN"
    multiscale = to_multiscale(image, [2, 3], method=Methods.ITK_LABEL_GAUSSIAN)
    verify_against_baseline(dataset_name, baseline_name, multiscale)


def test_anisotropic_scale_factors(input_images):
    dataset_name = "cthead1"
    image = input_images[dataset_name]
    scale_factors = [{"x": 2, "y": 4}, {"x": 1, "y": 2}]
    multiscale = to_multiscale(image, scale_factors, method=Methods.ITK_BIN_SHRINK)
    baseline_name = ("x2y4_x1y2/ITK_BIN_SHRINK",)
    verify_against_baseline(dataset_name, baseline_name, multiscale)

    dataset_name = "small_head"
    image = input_images[dataset_name]
    scale_factors = [
        {"x": 3, "y": 2, "z": 4},
        {"x": 2, "y": 2, "z": 2},
        {"x": 1, "y": 2, "z": 1},
    ]
    multiscale = to_multiscale(image, scale_factors, method=Methods.ITK_BIN_SHRINK)
    baseline_name = "x3y2z4_x2y2z2_x1y2z1/ITK_BIN_SHRINK"
    verify_against_baseline(dataset_name, baseline_name, multiscale)
    

def test_gaussian_anisotropic_scale_factors(input_images):
    dataset_name = "cthead1"
    image = input_images[dataset_name]
    scale_factors = [{"x": 2, "y": 4}, {"x": 1, "y": 2}]
    multiscale = to_multiscale(image, scale_factors, method=Methods.ITK_GAUSSIAN)
    baseline_name = "x2y4_x1y2/ITK_GAUSSIAN"
    verify_against_baseline(dataset_name, baseline_name, multiscale)

    dataset_name = "small_head"
    image = input_images[dataset_name]
    scale_factors = [
        {"x": 3, "y": 2, "z": 4},
        {"x": 2, "y": 2, "z": 2},
        {"x": 1, "y": 2, "z": 1},
    ]
    multiscale = to_multiscale(image, scale_factors, method=Methods.ITK_GAUSSIAN)
    baseline_name = "x3y2z4_x2y2z2_x1y2z1/ITK_GAUSSIAN"
    verify_against_baseline(dataset_name, baseline_name, multiscale)


def test_label_gaussian_anisotropic_scale_factors(input_images):
    dataset_name = "2th_cthead1"
    image = input_images[dataset_name]
    scale_factors = [{"x": 2, "y": 4}, {"x": 1, "y": 2}]
    multiscale = to_multiscale(image, scale_factors, method=Methods.ITK_LABEL_GAUSSIAN)
    baseline_name = "x2y4_x1y2/ITK_LABEL_GAUSSIAN"
    verify_against_baseline(dataset_name, baseline_name, multiscale)


def test_from_itk(input_images):
    import itk
    import numpy as np

    # Test 2D with ITK default metadata
    dataset_name = "cthead1"
    image = itk.image_from_xarray(input_images[dataset_name])
    scale_factors=[4,2]
    multiscale = itk_image_to_multiscale(image, scale_factors)
    baseline_name = "4_2/from_itk"
    verify_against_baseline(dataset_name, baseline_name, multiscale)

    # Test 2D with nonunit metadata
    dataset_name = "cthead1"
    image = itk.image_from_xarray(input_images[dataset_name])
    image.SetDirection(np.array([[-1,0],[0,1]]))
    image.SetSpacing([0.5,2.0])
    image.SetOrigin([3.0,5.0])

    name='cthead1_nonunit_metadata'
    axis_units={dim: 'millimeters' for dim in ('x','y','z')}

    scale_factors=[4,2]
    multiscale = itk_image_to_multiscale(image, scale_factors=scale_factors, anatomical_axes=False, axis_units=axis_units, name=name)
    baseline_name = "4_2/from_itk_nonunit_metadata"
    verify_against_baseline(dataset_name, baseline_name, multiscale)

    # Expect error for 2D image with anatomical axes
    try:
        itk_image_to_multiscale(image, scale_factors=scale_factors, anatomical_axes=True)
        raise Exception('Failed to catch expected exception for 2D image requesting anatomical axes')
    except ValueError:
        pass # caught expected exception
    
    # Test 3D with ITK default metadata
    dataset_name = "small_head"
    image = itk.image_from_xarray(input_images[dataset_name])
    scale_factors=[4,2]
    multiscale = itk_image_to_multiscale(image, scale_factors)
    baseline_name = "4_2/from_itk"
    verify_against_baseline(dataset_name, baseline_name, multiscale)

    # Test 3D with additional metadata
    dataset_name = "small_head"
    image = itk.image_from_xarray(input_images[dataset_name])
    image.SetObjectName(str(input_images[dataset_name].name)) # implicit in image_from_xarray in itk>v5.3rc04

    name='small_head_anatomical'
    axis_units={dim: 'millimeters' for dim in input_images[dataset_name].dims}

    scale_factors=[4,2]
    multiscale = itk_image_to_multiscale(image, scale_factors=scale_factors, anatomical_axes=True, axis_units=axis_units, name=name)
    baseline_name = "4_2/from_itk_anatomical"
    verify_against_baseline(dataset_name, baseline_name, multiscale)
    