from multiscale_spatial_image import skip_non_dimension_nodes


def test_skip_nodes(multiscale_data):
    @skip_non_dimension_nodes
    def transpose(ds, *args, **kwargs):
        return ds.transpose(*args, **kwargs)

    for scale in list(multiscale_data.keys()):
        assert multiscale_data[scale]["image"].dims == ("c", "y", "x")

    # applying this function without skipping the root node would fail as the root node does not have dimensions.
    result = multiscale_data.map_over_datasets(transpose, "y", "x", "c")
    for scale in list(result.keys()):
        assert result[scale]["image"].dims == ("y", "x", "c")
