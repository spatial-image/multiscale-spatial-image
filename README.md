# multiscale-spatial-image

[![Test](https://github.com/spatial-image/multiscale-spatial-image/actions/workflows/test.yml/badge.svg)](https://github.com/spatial-image/multiscale-spatial-image/actions/workflows/test.yml)
[![Notebook tests](https://github.com/spatial-image/multiscale-spatial-image/actions/workflows/notebook-test.yml/badge.svg)](https://github.com/spatial-image/multiscale-spatial-image/actions/workflows/notebook-test.yml)
[![image](https://img.shields.io/pypi/v/multiscale_spatial_image.svg)](https://pypi.python.org/pypi/multiscale_spatial_image/)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![DOI](https://zenodo.org/badge/379678181.svg)](https://zenodo.org/badge/latestdoi/379678181)

Generate a multiscale, chunked, multi-dimensional spatial image data structure
that can serialized to [OME-NGFF].

Each scale is a scientific Python [Xarray] [spatial-image] [Dataset], organized
into nodes of an Xarray [Datatree].

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

An [Xarray] [spatial-image] [DataArray]. Spatial metadata can also be passed
during construction.

```
<xarray.DataArray 'image' (y: 128, x: 128)> Size: 16kB
array([[170,  79, 215, ...,  31, 151, 150],
       [ 77, 181,   1, ..., 217, 176, 228],
       [193,  91, 240, ..., 132, 152,  41],
       ...,
       [ 50, 140, 231, ...,  80, 236,  28],
       [ 89,  46, 180, ...,  84,  42, 140],
       [ 96, 148, 240, ...,  61,  43, 255]], dtype=uint8)
Coordinates:
  * y        (y) float64 1kB 0.0 1.0 2.0 3.0 4.0 ... 124.0 125.0 126.0 127.0
  * x        (x) float64 1kB 0.0 1.0 2.0 3.0 4.0 ... 124.0 125.0 126.0 127.0
```

```python
# Create multiscale pyramid, downscaling by a factor of 2, then 4
multiscale = to_multiscale(image, [2, 4])
print(multiscale)
```

A chunked [Dask] Array MultiscaleSpatialImage [Xarray] [Datatree].

```
<xarray.DataTree>
Group: /
├── Group: /scale0
│       Dimensions:  (y: 128, x: 128)
│       Coordinates:
│         * y        (y) float64 1kB 0.0 1.0 2.0 3.0 4.0 ... 124.0 125.0 126.0 127.0
│         * x        (x) float64 1kB 0.0 1.0 2.0 3.0 4.0 ... 124.0 125.0 126.0 127.0
│       Data variables:
│           image    (y, x) uint8 16kB dask.array<chunksize=(128, 128), meta=np.ndarray>
├── Group: /scale1
│       Dimensions:  (y: 64, x: 64)
│       Coordinates:
│         * y        (y) float64 512B 0.5 2.5 4.5 6.5 8.5 ... 120.5 122.5 124.5 126.5
│         * x        (x) float64 512B 0.5 2.5 4.5 6.5 8.5 ... 120.5 122.5 124.5 126.5
│       Data variables:
│           image    (y, x) uint8 4kB dask.array<chunksize=(64, 64), meta=np.ndarray>
└── Group: /scale2
        Dimensions:  (y: 16, x: 16)
        Coordinates:
          * y        (y) float64 128B 3.5 11.5 19.5 27.5 35.5 ... 99.5 107.5 115.5 123.5
          * x        (x) float64 128B 3.5 11.5 19.5 27.5 35.5 ... 99.5 107.5 115.5 123.5
        Data variables:
            image    (y, x) uint8 256B dask.array<chunksize=(16, 16), meta=np.ndarray>
```

Map a function over datasets while skipping nodes that do not contain dimensions

```python
import numpy as np
from spatial_image import to_spatial_image
from multiscale_spatial_image import skip_non_dimension_nodes, to_multiscale

data = np.zeros((2, 200, 200))
dims = ("c", "y", "x")
scale_factors = [2, 2]
image = to_spatial_image(array_like=data, dims=dims)
multiscale = to_multiscale(image, scale_factors=scale_factors)

@skip_non_dimension_nodes
def transpose(ds, *args, **kwargs):
    return ds.transpose(*args, **kwargs)

multiscale = multiscale.map_over_datasets(transpose, "y", "x", "c")
print(multiscale)
```

A transposed MultiscaleSpatialImage.

```
<xarray.DataTree>
Group: /
├── Group: /scale0
│       Dimensions:  (c: 2, y: 200, x: 200)
│       Coordinates:
│         * c        (c) int32 8B 0 1
│         * y        (y) float64 2kB 0.0 1.0 2.0 3.0 4.0 ... 196.0 197.0 198.0 199.0
│         * x        (x) float64 2kB 0.0 1.0 2.0 3.0 4.0 ... 196.0 197.0 198.0 199.0
│       Data variables:
│           image    (y, x, c) float64 640kB dask.array<chunksize=(200, 200, 2), meta=np.ndarray>
├── Group: /scale1
│       Dimensions:  (c: 2, y: 100, x: 100)
│       Coordinates:
│         * c        (c) int32 8B 0 1
│         * y        (y) float64 800B 0.5 2.5 4.5 6.5 8.5 ... 192.5 194.5 196.5 198.5
│         * x        (x) float64 800B 0.5 2.5 4.5 6.5 8.5 ... 192.5 194.5 196.5 198.5
│       Data variables:
│           image    (y, x, c) float64 160kB dask.array<chunksize=(100, 100, 2), meta=np.ndarray>
└── Group: /scale2
        Dimensions:  (c: 2, y: 50, x: 50)
        Coordinates:
          * c        (c) int32 8B 0 1
          * y        (y) float64 400B 1.5 5.5 9.5 13.5 17.5 ... 185.5 189.5 193.5 197.5
          * x        (x) float64 400B 1.5 5.5 9.5 13.5 17.5 ... 185.5 189.5 193.5 197.5
        Data variables:
            image    (y, x, c) float64 40kB dask.array<chunksize=(50, 50, 2), meta=np.ndarray>
```

While the decorator allows you to define your own methods to map over datasets
in the `DataTree` while ignoring those datasets not having dimensions, this
library also provides a few convenience methods. For example, the transpose
method we saw earlier can also be applied as follows:

```python
multiscale = multiscale.msi.transpose("y", "x", "c")
```

Other methods implemented this way are `reindex`, equivalent to the
`xr.DataArray`
[reindex](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.reindex.html)
method and `assign_coords`, equivalent to `xr.Dataset` `assign_coords` method.

Store as an Open Microscopy Environment-Next Generation File Format ([OME-NGFF])
/ [netCDF] [Zarr] store.

**Note**: The API is under development, and it may change until 1.0.0 is
released. We mean it :-).

## Examples

- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/spatial-image/multiscale-spatial-image/main?urlpath=lab/tree/examples%2FHelloMultiscaleSpatialImageWorld.ipynb)
  [Hello MultiscaleSpatialImage World!](./examples/HelloMultiscaleSpatialImageWorld.ipynb)
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/spatial-image/multiscale-spatial-image/main?urlpath=lab/tree/examples%2FConvertITKImage.ipynb)
  [Convert itk.Image](./examples/ConvertITKImage.ipynb)
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/spatial-image/multiscale-spatial-image/main?urlpath=lab/tree/examples%2FConvertImageioImageResource.ipynb)
  [Convert imageio ImageResource](./examples/ConvertImageioImageResource.ipynb)
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/spatial-image/multiscale-spatial-image/main?urlpath=lab/tree/examples%2FConvertPyImageJDataset.ipynb)
  [Convert pyimagej Dataset](./examples/ConvertPyImageJDataset.ipynb)

## Development

Contributions are welcome and appreciated.

### Get the source code

```shell
git clone https://github.com/spatial-image/multiscale-spatial-image
cd multiscale-spatial-image
```

### Install dependencies

First install [pixi]. Then, install project dependencies:

```shell
pixi install -a
pixi run pre-commit-install
```

### Run the test suite

The unit tests:

```shell
pixi run -e test test
```

The notebooks tests:

```shell
pixi run test-notebooks
```

### Update test data

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

Update the `test_data_sha256` variable in the _test/\_data.py_ file. Upload the
data to [web3.storage](https://web3.storage). And update the
`test_data_ipfs_cid`
[Content Identifier (CID)](https://proto.school/anatomy-of-a-cid/01) variable,
which is available in the web3.storage web page interface.

### Submit the patch

We use the standard [GitHub flow].

### Create a release

This section is relevant only for maintainers.

1. Pull `git`'s `main` branch.
2. `pixi install -a`
3. `pixi run pre-commit-install`
4. `pixi run -e test test`
5. `pixi shell`
6. `hatch version <new-version>`
7. `git add .`
8. `git commit -m "ENH: Bump version to <version>"`
9. `hatch build`
10. `hatch publish`
11. `git push upstream main`
12. Create a new tag and Release via the GitHub UI. Auto-generate release notes
    and add additional notes as needed.

[spatial-image]: https://github.com/spatial-image/spatial-image
[Xarray]: https://xarray.pydata.org/en/stable/
[OME-NGFF]: https://ngff.openmicroscopy.org/
[Dataset]: https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html
[Datatree]: https://xarray-datatree.readthedocs.io/en/latest/
[DataArray]: https://xarray.pydata.org/en/stable/generated/xarray.DataArray.html
[Zarr]: https://zarr.readthedocs.io/en/stable/
[Dask]: https://docs.dask.org/en/stable/array.html
[netCDF]: https://www.unidata.ucar.edu/software/netcdf/
[pixi]: https://pixi.sh
[GitHub flow]: https://docs.github.com/en/get-started/using-github/github-flow
