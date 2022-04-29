"""spatial-image-multiscale

Generate a multiscale spatial image."""

__version__ = "0.3.0"

import enum
from typing import Union, Sequence, List, Optional, Dict, Mapping, Any, Tuple
from enum import Enum

from spatial_image import to_spatial_image, SpatialImage  # type: ignore

import xarray as xr
from dask.array import map_blocks, map_overlap
from datatree import DataTree
from datatree.treenode import TreeNode
import numpy as np

_spatial_dims = {"x", "y", "z"}


class MultiscaleSpatialImage(DataTree):
    """A multi-scale representation of a spatial image.

    This is an xarray DataTree, where the root is named `multiscales` by default (to signal content that is
    compatible with the Open Microscopy Environment Next Generation File Format (OME-NGFF)
    instead of the default generic DataTree `root`.

    The tree contains nodes in the form: `multiscales/{scale}` where *scale* is the integer scale.
    Each node has a the same named `Dataset` that corresponds to to the NGFF dataset name.
     For example, a three-scale representation of a *cells* dataset would have `Dataset` nodes:

      multiscales/0
      multiscales/1
      multiscales/2
    """

    def __init__(
        self,
        name: str = "multiscales",
        data: Union[xr.Dataset, xr.DataArray] = None,
        parent: TreeNode = None,
        children: List[TreeNode] = None,
    ):
        """DataTree with a root name of *multiscales*."""
        super().__init__(name, data=data, parent=parent, children=children)

    def to_zarr(self, store, mode: str = "w", encoding=None, **kwargs):
        """
        Write multi-scale spatial image contents to a Zarr store.

        Metadata is added according the OME-NGFF standard.

        store : MutableMapping, str or Path, optional
            Store or path to directory in file system
        mode : {{"w", "w-", "a", "r+", None}, default: "w"
            Persistence mode: “w” means create (overwrite if exists); “w-” means create (fail if exists);
            “a” means override existing variables (create if does not exist); “r+” means modify existing
            array values only (raise an error if any metadata or shapes would change). The default mode
            is “a” if append_dim is set. Otherwise, it is “r+” if region is set and w- otherwise.
        encoding : dict, optional
            Nested dictionary with variable names as keys and dictionaries of
            variable specific encodings as values, e.g.,
            ``{"multiscales/0/image": {"my_variable": {"dtype": "int16", "scale_factor": 0.1}, ...}, ...}``.
            See ``xarray.Dataset.to_zarr`` for available options.
        kwargs :
            Additional keyword arguments to be passed to ``datatree.DataTree.to_zarr``
        """

        multiscales = []
        for name in self.children[0].ds.data_vars.keys():
            ngff_datasets = []
            for child in self.children:
                image = child.ds
                scale_transform = []
                translate_transform = []
                for dim in image.dims:
                    if len(image.coords[dim]) > 1:
                        scale_transform.append(
                            float(image.coords[dim][1] - image.coords[dim][0])
                        )
                    else:
                        scale_transform.append(1.0)
                    if len(image.coords[dim]) > 0:
                        translate_transform.append(float(image.coords[dim][0]))
                    else:
                        translate_transform.append(0.0)

                ngff_datasets.append(
                    {
                        "path": f"{child.name}/{name}",
                        "coordinateTransformations": [
                            {
                                "type": "scale",
                                "scale": scale_transform,
                            },
                            {
                                "type": "translation",
                                "translation": translate_transform,
                            },
                        ],
                    }
                )

            image = self.children[0].ds
            axes = []
            for axis in image.dims:
                if axis == "t":
                    axes.append({"name": "t", "type": "time"})
                elif axis == "c":
                    axes.append({"name": "c", "type": "channel"})
                else:
                    axes.append({"name": axis, "type": "space"})
                if "units" in image.coords[axis].attrs:
                    axes[-1]["unit"] = image.coords[axis].attrs["units"]

            multiscales.append(
                {
                    "version": "0.4",
                    "name": name,
                    "axes": axes,
                    "datasets": ngff_datasets,
                }
            )

        # NGFF v0.4 metadata
        ngff_metadata = {"multiscales": multiscales}
        self.ds.attrs = ngff_metadata

        super().to_zarr(store, **kwargs)


class Methods(Enum):
    XARRAY_COARSEN = "xarray.DataArray.coarsen"
    ITK_BIN_SHRINK = "itk.bin_shrink_image_filter"
    ITK_GAUSSIAN = "itk.discrete_gaussian_image_filter"


def to_multiscale(
    image: SpatialImage,
    scale_factors: Sequence[Union[Dict[str, int], int]],
    method: Optional[Methods] = None,
    chunks: Optional[
        Union[
            int,
            Tuple[int, ...],
            Tuple[Tuple[int, ...], ...],
            Mapping[Any, Union[None, int, Tuple[int, ...]]],
        ]
    ] = None,
) -> MultiscaleSpatialImage:
    """Generate a multiscale representation of a spatial image.

    Parameters
    ----------

    image : SpatialImage
        The spatial image from which we generate a multi-scale representation.

    scale_factors : int per scale or dict of spatial dimension int's per scale
        Integer scale factors to apply uniformly across all spatial dimension or
        along individual spatial dimensions.
        Examples: [2, 2] or [{'x': 2, 'y': 4 }, {'x': 5, 'y': 10}]

    method : spatial_image_multiscale.Methods, optional
        Method to reduce the input image.

    chunks : xarray Dask array chunking specification, optional
        Specify the chunking used in each output scale.

    Returns
    -------

    result : MultiscaleSpatialImage
        Multiscale representation. An xarray DataTree where each node is a SpatialImage Dataset
        named by the integer scale.  Increasing scales are downscaled versions of the input image.
    """

    # IPFS and visualization friendly default chunks
    if "z" in image.dims:
        default_chunks = 64
    else:
        default_chunks = 256
    default_chunks = {d: default_chunks for d in image.dims}
    if "t" in image.dims:
        default_chunks["t"] = 1
    out_chunks = chunks
    if out_chunks is None:
        out_chunks = default_chunks

    current_input = image.chunk(out_chunks)
    # https://github.com/pydata/xarray/issues/5219
    if "chunks" in current_input.encoding:
        del current_input.encoding["chunks"]
    data_objects = {f"multiscales/0": current_input.to_dataset(name=image.name)}

    def dim_scale_factors(scale_factor):
        if isinstance(scale_factor, int):
            dim = {dim: scale_factor for dim in _spatial_dims.intersection(image.dims)}
        else:
            dim = scale_factor
        return dim

    def align_chunks(current_input, dim_factors):
        block_0_shape = [c[0] for c in current_input.chunks]

        rechunk = False
        aligned_chunks = {}
        for dim, factor in dim_factors.items():
            dim_index = current_input.dims.index(dim)
            if block_0_shape[dim_index] % factor:
                aligned_chunks[dim] = block_0_shape[dim_index] * factor
                rechunk = True
            else:
                aligned_chunks[dim] = default_chunks[dim]
        if rechunk:
            current_input = current_input.chunk(aligned_chunks)

        return current_input

    if method is Methods.XARRAY_COARSEN:
        for factor_index, scale_factor in enumerate(scale_factors):
            dim_factors = dim_scale_factors(scale_factor)
            current_input = align_chunks(current_input, dim_factors)

            downscaled = (
                current_input.coarsen(dim=dim_factors, boundary="trim", side="right")
                .mean()
                .astype(current_input.dtype)
            )

            downscaled = downscaled.chunk(out_chunks)

            data_objects[f"multiscales/{factor_index+1}"] = downscaled.to_dataset(
                name=image.name
            )
            current_input = downscaled
    elif method is Methods.ITK_BIN_SHRINK:
        import itk

        for factor_index, scale_factor in enumerate(scale_factors):
            dim_factors = dim_scale_factors(scale_factor)
            current_input = align_chunks(current_input, dim_factors)

            image_dims: Tuple[str, str, str] = ("x", "y", "z", "t")
            shrink_factors = [dim_factors[sf] for sf in image_dims if sf in dim_factors]

            block_0_shape = [c[0] for c in current_input.chunks]
            block_0 = current_input[tuple([slice(0, s) for s in block_0_shape])]
            # For consistency for now, do not utilize direction until there is standardized support for
            # direction cosines / orientation in OME-NGFF
            block_0.attrs.pop("direction", None)
            block_input = itk.image_from_xarray(block_0)
            filt = itk.BinShrinkImageFilter.New(
                block_input, shrink_factors=shrink_factors
            )
            filt.UpdateOutputInformation()
            block_output = filt.GetOutput()
            scale = {
                image_dims[i]: s for (i, s) in enumerate(block_output.GetSpacing())
            }
            translation = {
                image_dims[i]: s for (i, s) in enumerate(block_output.GetOrigin())
            }
            dtype = block_output.dtype
            output_chunks = list(current_input.chunks)
            for i, c in enumerate(output_chunks):
                output_chunks[i] = [
                    block_output.shape[i],
                ] * len(c)

            block_neg1_shape = [c[-1] for c in current_input.chunks]
            block_neg1 = current_input[tuple([slice(0, s) for s in block_neg1_shape])]
            block_neg1.attrs.pop("direction", None)
            block_input = itk.image_from_xarray(block_neg1)
            filt = itk.BinShrinkImageFilter.New(
                block_input, shrink_factors=shrink_factors
            )
            filt.UpdateOutputInformation()
            block_output = filt.GetOutput()
            for i, c in enumerate(output_chunks):
                output_chunks[i][-1] = block_output.shape[i]
                output_chunks[i] = tuple(output_chunks[i])
            output_chunks = tuple(output_chunks)

            downscaled_array = map_blocks(
                itk.bin_shrink_image_filter,
                current_input.data,
                shrink_factors=shrink_factors,
                dtype=dtype,
                chunks=output_chunks,
            )
            downscaled = to_spatial_image(
                downscaled_array,
                dims=image.dims,
                scale=scale,
                translation=translation,
                name=current_input.name,
                axis_names={
                    d: image.coords[d].attrs.get("long_name", d) for d in image.dims
                },
                axis_units={
                    d: image.coords[d].attrs.get("units", "") for d in image.dims
                },
                t_coords=image.coords.get("t", None),
                c_coords=image.coords.get("c", None),
            )
            downscaled = downscaled.chunk(out_chunks)
            data_objects[f"multiscales/{factor_index+1}"] = downscaled.to_dataset(
                name=image.name
            )
            current_input = downscaled

    multiscale = MultiscaleSpatialImage.from_dict(
        name="multiscales", data_objects=data_objects
    )

    return multiscale
