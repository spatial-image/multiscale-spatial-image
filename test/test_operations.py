def test_transpose(multiscale_data):
    multiscale_data = multiscale_data.msi.transpose("y", "x", "c")

    for scale in list(multiscale_data.keys()):
        assert multiscale_data[scale]["image"].dims == ("y", "x", "c")


def test_reindex_arrays(multiscale_data):
    multiscale_data = multiscale_data.msi.reindex_data_arrays({"c": ["r", "g", "b"]})
    for scale in list(multiscale_data.keys()):
        assert multiscale_data[scale].c.data.tolist() == ["r", "g", "b"]


def test_assign_coords(multiscale_data):
    multiscale_data = multiscale_data.msi.assign_coords({"c": ["r", "g", "b"]})
    for scale in list(multiscale_data.keys()):
        assert multiscale_data[scale].c.data.tolist() == ["r", "g", "b"]
