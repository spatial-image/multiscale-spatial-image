# multiscale-spatial-image

[![Test](https://github.com/spatial-image/multiscale-spatial-image/actions/workflows/test.yml/badge.svg)](https://github.com/spatial-image/multiscale-spatial-image/actions/workflows/test.yml)
[![Notebook tests](https://github.com/spatial-image/multiscale-spatial-image/actions/workflows/notebook-test.yml/badge.svg)](https://github.com/spatial-image/multiscale-spatial-image/actions/workflows/notebook-test.yml)
[![image](https://img.shields.io/pypi/v/multiscale_spatial_image.svg)](https://pypi.python.org/pypi/multiscale_spatial_image/)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![DOI](https://zenodo.org/badge/379678181.svg)](https://zenodo.org/badge/latestdoi/379678181)

Generate a multiscale, chunked, multi-dimensional spatial image data structure that can serialized to [OME-NGFF].

Each scale is a scientific Python [Xarray] [spatial-image] [Dataset], organized into nodes of an Xarray [Datatree].


## Installation

```sh
pip install multiscale_spatial_image
```

## Usage

```python
import numpy as np
from spatial_image import to_spatial_image
from multiscale_spatial_image import to_multiscale
import zarr

# Image pixels
array = np.random.randint(0, 256, size=(128,128), dtype=np.uint8)

image = to_spatial_image(array)
print(image)
```

An [Xarray] [spatial-image] [DataArray].
Spatial metadata can also be passed during construction.

```
<xarray.SpatialImage 'image' (y: 128, x: 128)>
array([[114,  47, 215, ..., 245,  14, 175],
       [ 94, 186, 112, ...,  42,  96,  30],
       [133, 170, 193, ..., 176,  47,   8],
       ...,
       [202, 218, 237, ...,  19, 108, 135],
       [ 99,  94, 207, ..., 233,  83, 112],
       [157, 110, 186, ..., 142, 153,  42]], dtype=uint8)
Coordinates:
  * y        (y) float64 0.0 1.0 2.0 3.0 4.0 ... 123.0 124.0 125.0 126.0 127.0
  * x        (x) float64 0.0 1.0 2.0 3.0 4.0 ... 123.0 124.0 125.0 126.0 127.0
```

```python
# Create multiscale pyramid, downscaling by a factor of 2, then 4
multiscale = to_multiscale(image, [2, 4])
print(multiscale)
```

A chunked [Dask] Array MultiscaleSpatialImage [Xarray] [Datatree].

```
DataTree('multiscales', parent=None)
├── DataTree('scale0')
│   Dimensions:  (y: 128, x: 128)
│   Coordinates:
│     * y        (y) float64 0.0 1.0 2.0 3.0 4.0 ... 123.0 124.0 125.0 126.0 127.0
│     * x        (x) float64 0.0 1.0 2.0 3.0 4.0 ... 123.0 124.0 125.0 126.0 127.0
│   Data variables:
│       image    (y, x) uint8 dask.array<chunksize=(128, 128), meta=np.ndarray>
├── DataTree('scale1')
│   Dimensions:  (y: 64, x: 64)
│   Coordinates:
│     * y        (y) float64 0.5 2.5 4.5 6.5 8.5 ... 118.5 120.5 122.5 124.5 126.5
│     * x        (x) float64 0.5 2.5 4.5 6.5 8.5 ... 118.5 120.5 122.5 124.5 126.5
│   Data variables:
│       image    (y, x) uint8 dask.array<chunksize=(64, 64), meta=np.ndarray>
└── DataTree('scale2')
    Dimensions:  (y: 16, x: 16)
    Coordinates:
      * y        (y) float64 3.5 11.5 19.5 27.5 35.5 ... 91.5 99.5 107.5 115.5 123.5
      * x        (x) float64 3.5 11.5 19.5 27.5 35.5 ... 91.5 99.5 107.5 115.5 123.5
    Data variables:
        image    (y, x) uint8 dask.array<chunksize=(16, 16), meta=np.ndarray>
```

Store as an Open Microscopy Environment-Next Generation File Format ([OME-NGFF]) / [netCDF] [Zarr] store.

It is highly recommended to use `dimension_separator='/'` in the construction of the Zarr stores.

```python
store = zarr.storage.DirectoryStore('multiscale.zarr', dimension_separator='/')
multiscale.to_zarr(store)
```

**Note**: The API is under development, and it may change until 1.0.0 is
released. We mean it :-).

## Examples

- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/spatial-image/multiscale-spatial-image/main?urlpath=lab/tree/examples%2FHelloMultiscaleSpatialImageWorld.ipynb) [Hello MultiscaleSpatialImage World!](./examples/HelloMultiscaleSpatialImageWorld.ipynb) 
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/spatial-image/multiscale-spatial-image/main?urlpath=lab/tree/examples%2FConvertITKImage.ipynb) [Convert itk.Image](./examples/ConvertITKImage.ipynb) 
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/spatial-image/multiscale-spatial-image/main?urlpath=lab/tree/examples%2FConvertImageioImageResource.ipynb) [Convert imageio ImageResource](./examples/ConvertImageioImageResource.ipynb) 
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/spatial-image/multiscale-spatial-image/main?urlpath=lab/tree/examples%2FConvertPyImageJDataset.ipynb) [Convert pyimagej Dataset](./examples/ConvertPyImageJDataset.ipynb) 

## Development

Contributions are welcome and appreciated.

To run the test suite:

```
git clone https://github.com/spatial-image/multiscale-spatial-image
cd multiscale-spatial-image
pip install -e ".[test]"
pytest
# Notebook tests
pytest --nbmake --nbmake-timeout=3000 examples/*ipynb
```

To add new or update testing data, such as a new baseline for this block:

```py
dataset_name = "cthead1"
image = input_images[dataset_name]
baseline_name = "2_4/XARRAY_COARSEN"
multiscale = to_multiscale(image, [2, 4], method=Methods.XARRAY_COARSEN)
verify_against_baseline(test_data_dir, dataset_name, baseline_name, multiscale)
```

Add a `store_new_image` call in your test block:

```py
dataset_name = "cthead1"
image = input_images[dataset_name]
baseline_name = "2_4/XARRAY_COARSEN"
multiscale = to_multiscale(image, [2, 4], method=Methods.XARRAY_COARSEN)

store_new_image(dataset_name, baseline_name, multiscale)

verify_against_baseline(dataset_name, baseline_name, multiscale)
```

Run the tests to generate the output. Remove the `store_new_image` call.

Then, create a tarball of the current testing data

```console
cd test/data
tar cvf ../data.tar *
gzip -9 ../data.tar
python3 -c 'import pooch; print(pooch.file_hash("../data.tar.gz"))'
```

Update the `test_data_sha256` variable in the *test/_data.py* file.
Upload the data to [web3.storage](https://web3.storage).
And update the `test_data_ipfs_cid` [Content Identifier (CID)](https://proto.school/anatomy-of-a-cid/01) variable, which is available in the web3.storage web page interface.


[spatial-image]: https://github.com/spatial-image/spatial-image
[Xarray]: https://xarray.pydata.org/en/stable/
[OME-NGFF]: https://ngff.openmicroscopy.org/
[Dataset]: https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html
[Datatree]: https://xarray-datatree.readthedocs.io/en/latest/
[DataArray]: https://xarray.pydata.org/en/stable/generated/xarray.DataArray.html
[Zarr]: https://zarr.readthedocs.io/en/stable/
[Dask]: https://docs.dask.org/en/stable/array.html
[netCDF]: https://www.unidata.ucar.edu/software/netcdf/
