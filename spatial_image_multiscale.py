"""spatial-image-multiscale

Generate a multiscale spatial image."""

__version__ = "0.3.0"

from typing import Union, Sequence, List, Optional, Dict
from enum import Enum

from spatial_image import SpatialImage  # type: ignore

import xarray as xr
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

    def to_zarr(
        self,
        store,
        mode: str = "w",
        encoding=None,
        **kwargs
    ):
        """
        Write spatialimage contents to a Zarr store.

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


        super().to_zarr(store, **kwargs)


class Method(Enum):
    XARRAY_COARSEN = "xarray.coarsen"


def to_multiscale(
    image: SpatialImage,
    scale_factors: Sequence[Union[Dict[str, int], int]],
    method: Optional[Method] = None,
) -> MultiscaleSpatialImage:
    """Generate a multiscale representation of a spatial image.

    Parameters
    ----------

    image : SpatialImage
        The spatial image from which we generate a multi-scale representation.

    scale_factors : int per scale or spatial dimension int's per scale
        Sequence of integer scale factors to apply along each spatial dimension.

    method : spatial_image_multiscale.Method, optional
        Method to reduce the input image.

    Returns
    -------

    result : MultiscaleSpatialImage
        Multiscale representation. An xarray DataTree where each node is a SpatialImage Dataset
        named by the integer scale.  Increasing scales are downscaled versions of the input image.
    """

    data_objects = {f"multiscales/0": image.to_dataset(name=image.name)}

    scale_transform = []
    translate_transform = []
    for dim in image.dims:
        if len(image.coords[dim]) > 1:
            scale_transform.append(float(image.coords[dim][1] - image.coords[dim][0]))
        else:
            scale_transform.append(1.0)
        if len(image.coords[dim]) > 0:
            translate_transform.append(float(image.coords[dim][0]))
        else:
            translate_transform.append(0.0)

    ngff_datasets = [
        {
            "path": f"0/{image.name}",
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
    ]
    current_input = image
    for factor_index, scale_factor in enumerate(scale_factors):
        if isinstance(scale_factor, int):
            dim = {dim: scale_factor for dim in _spatial_dims.intersection(image.dims)}
        else:
            dim = scale_factor
        downscaled = current_input.coarsen(
            dim=dim, boundary="trim", side="right"
        ).mean()
        data_objects[f"multiscales/{factor_index+1}"] = downscaled.to_dataset(name=image.name)

        scale_transform = []
        translate_transform = []
        for dim in image.dims:
            if len(downscaled.coords[dim]) > 1:
                scale_transform.append(
                    float(downscaled.coords[dim][1] - downscaled.coords[dim][0])
                )
            else:
                scale_transform.append(1.0)
            if len(downscaled.coords[dim]) > 0:
                translate_transform.append(float(downscaled.coords[dim][0]))
            else:
                translate_transform.append(0.0)

        ngff_datasets.append(
            {
                "path": f"{factor_index+1}/{image.name}",
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

        current_input = downscaled

    multiscale = MultiscaleSpatialImage.from_dict(
        name="multiscales", data_objects=data_objects
    )

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

    # NGFF v0.4 metadata
    ngff_metadata = {
        "multiscales": [
            {
                "version": "0.4",
                "name": image.name,
                "axes": axes,
                "datasets": ngff_datasets,
            }
        ]
    }
    multiscale.ds.attrs = ngff_metadata

    return multiscale
