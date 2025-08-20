from multiscale_spatial_image import Methods, to_multiscale

from ._data import verify_against_baseline, input_images  # noqa: F401


def test_gaussian_isotropic_scale_factors(input_images):  # noqa: F811
    dataset_name = "cthead1"
    image = input_images[dataset_name]
    baseline_name = "2_4/DASK_IMAGE_GAUSSIAN"
    multiscale = to_multiscale(image, [2, 4], method=Methods.DASK_IMAGE_GAUSSIAN)
    verify_against_baseline(dataset_name, baseline_name, multiscale)

    dataset_name = "cthead1"
    image = input_images[dataset_name]
    baseline_name = "2_3/DASK_IMAGE_GAUSSIAN"
    multiscale = to_multiscale(image, [2, 3], method=Methods.DASK_IMAGE_GAUSSIAN)
    verify_against_baseline(dataset_name, baseline_name, multiscale)

    dataset_name = "small_head"
    image = input_images[dataset_name]
    baseline_name = "2_3_4/DASK_IMAGE_GAUSSIAN"
    multiscale = to_multiscale(image, [2, 3, 4], method=Methods.DASK_IMAGE_GAUSSIAN)
    verify_against_baseline(dataset_name, baseline_name, multiscale)


def test_gaussian_anisotropic_scale_factors(input_images):  # noqa: F811
    dataset_name = "cthead1"
    image = input_images[dataset_name]
    scale_factors = [{"x": 1, "y": 2}, {"x": 2, "y": 4}]
    multiscale = to_multiscale(image, scale_factors, method=Methods.DASK_IMAGE_GAUSSIAN)
    baseline_name = "x2y4_x1y2/DASK_IMAGE_GAUSSIAN"
    verify_against_baseline(dataset_name, baseline_name, multiscale)

    dataset_name = "small_head"
    image = input_images[dataset_name]
    scale_factors = [
        {"x": 1, "y": 2, "z": 1},
        {"x": 2, "y": 2, "z": 2},
        {"x": 3, "y": 2, "z": 4},
    ]
    multiscale = to_multiscale(image, scale_factors, method=Methods.DASK_IMAGE_GAUSSIAN)
    baseline_name = "x3y2z4_x2y2z2_x1y2z1/DASK_IMAGE_GAUSSIAN"
    verify_against_baseline(dataset_name, baseline_name, multiscale)


def test_label_nearest_isotropic_scale_factors(input_images):  # noqa: F811
    dataset_name = "2th_cthead1"
    image = input_images[dataset_name]
    baseline_name = "2_4/DASK_IMAGE_NEAREST"
    multiscale = to_multiscale(image, [2, 4], method=Methods.DASK_IMAGE_NEAREST)
    verify_against_baseline(dataset_name, baseline_name, multiscale)

    dataset_name = "2th_cthead1"
    image = input_images[dataset_name]
    baseline_name = "2_3/DASK_IMAGE_NEAREST"
    multiscale = to_multiscale(image, [2, 3], method=Methods.DASK_IMAGE_NEAREST)
    verify_against_baseline(dataset_name, baseline_name, multiscale)


def test_label_nearest_anisotropic_scale_factors(input_images):  # noqa: F811
    dataset_name = "2th_cthead1"
    image = input_images[dataset_name]
    scale_factors = [{"x": 1, "y": 2}, {"x": 2, "y": 4}]
    multiscale = to_multiscale(image, scale_factors, method=Methods.DASK_IMAGE_NEAREST)
    baseline_name = "x2y4_x1y2/DASK_IMAGE_NEAREST"
    verify_against_baseline(dataset_name, baseline_name, multiscale)


def test_label_mode_isotropic_scale_factors(input_images):  # noqa: F811
    dataset_name = "2th_cthead1"
    image = input_images[dataset_name]
    baseline_name = "2_4/DASK_IMAGE_MODE"
    multiscale = to_multiscale(image, [2, 4], method=Methods.DASK_IMAGE_MODE)
    verify_against_baseline(dataset_name, baseline_name, multiscale)

    dataset_name = "2th_cthead1"
    image = input_images[dataset_name]
    baseline_name = "2_3/DASK_IMAGE_MODE"
    multiscale = to_multiscale(image, [2, 3], method=Methods.DASK_IMAGE_MODE)
    verify_against_baseline(dataset_name, baseline_name, multiscale)


def test_label_mode_anisotropic_scale_factors(input_images):  # noqa: F811
    dataset_name = "2th_cthead1"
    image = input_images[dataset_name]
    scale_factors = [{"x": 1, "y": 2}, {"x": 2, "y": 4}]
    multiscale = to_multiscale(image, scale_factors, method=Methods.DASK_IMAGE_MODE)
    baseline_name = "x2y4_x1y2/DASK_IMAGE_MODE"
    verify_against_baseline(dataset_name, baseline_name, multiscale)
