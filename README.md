# spatial-image-multiscale

[![Test](https://github.com/spatial-image/spatial-image-multiscale/actions/workflows/test.yml/badge.svg)](https://github.com/spatial-image/spatial-image-multiscale/actions/workflows/test.yml)
[![Notebook tests](https://github.com/spatial-image/spatial-image-multiscale/actions/workflows/notebook-test.yml/badge.svg)](https://github.com/spatial-image/spatial-image-multiscale/actions/workflows/notebook-test.yml)
[![image](https://img.shields.io/pypi/v/spatial_image_multiscale.svg)](https://pypi.python.org/pypi/spatial_image_multiscale/)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

Generate a multiscale, chunked, multi-dimensional spatial image data structure serializable to [OME-NGFF].

Each scale is a scientific Python [Xarray] [spatial-image] [Dataset], organized into nodes of an Xarray [Datatree].


## Installation

```sh
pip install spatial_image_multiscale
```


## Development

Contributions are welcome and appreciated.

To run the test suite:

```
git clone https://github.com/spatial-image/spatial-image-multiscale
cd spatial-image-multiscale
pip install -e '.[test]'
cid=$(grep 'IPFS_CID =' test/test_spatial_image_multiscale.py | cut -d ' ' -f 3 | tr -d '"')
# Needs ipfs, e.g. https://docs.ipfs.io/install/ipfs-desktop/
ipfs get -o ./test/data -- $cid
pytest
# Notebook tests
pytest --nbmake --nbmake-timeout=3000 examples/*ipynb
```

[spatial-image]: https://github.com/spatial-image/spatial-image
[Xarray]: https://xarray.pydata.org/en/stable/
[OME-NGFF]: https://ngff.openmicroscopy.org/
[Dataset]: https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html
[Datatree]: https://xarray-datatree.readthedocs.io/en/latest/