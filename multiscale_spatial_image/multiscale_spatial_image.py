from typing import Union

from datatree import DataTree
import numpy as np
from collections.abc import MutableMapping
from pathlib import Path
from zarr.storage import BaseStore
from datatree import register_datatree_accessor


@register_datatree_accessor("msi")
class MultiscaleSpatialImage:
    """A multi-scale representation of a spatial image.

    This is an xarray DataTree, with content compatible with the Open Microscopy Environment-
    Next Generation File Format (OME-NGFF).

    The tree contains nodes in the form: `scale{scale}` where *scale* is the integer scale.
    Each node has a the same named `Dataset` that corresponds to to the NGFF dataset name.
     For example, a three-scale representation of a *cells* dataset would have `Dataset` nodes:

      scale0
      scale1
      scale2
    """

    def __init__(self, xarray_obj: DataTree):
        self._dt = xarray_obj

    def to_zarr(
        self,
        store: Union[MutableMapping, str, Path, BaseStore],
        mode: str = "w",
        encoding=None,
        **kwargs,
    ):
        """
        Write multi-scale spatial image contents to a Zarr store.

        Metadata is added according the OME-NGFF standard.

        store : MutableMapping, str or Path, or zarr.storage.BaseStore
            Store or path to directory in file system
        mode : {{"w", "w-", "a", "r+", None}, default: "w"
            Persistence mode: “w” means create (overwrite if exists); “w-” means create (fail if exists);
            “a” means override existing variables (create if does not exist); “r+” means modify existing
            array values only (raise an error if any metadata or shapes would change). The default mode
            is “a” if append_dim is set. Otherwise, it is “r+” if region is set and w- otherwise.
        encoding : dict, optional
            Nested dictionary with variable names as keys and dictionaries of
            variable specific encodings as values, e.g.,
            ``{"scale0/image": {"my_variable": {"dtype": "int16", "scale_factor": 0.1}, ...}, ...}``.
            See ``xarray.Dataset.to_zarr`` for available options.
        kwargs :
            Additional keyword arguments to be passed to ``datatree.DataTree.to_zarr``
        """

        multiscales = []
        scale0 = self._dt[self._dt.groups[1]]
        for name in scale0.ds.data_vars.keys():
            ngff_datasets = []
            for child in self._dt.children:
                image = self._dt[child].ds
                scale_transform = []
                translate_transform = []
                for dim in image.dims:
                    if len(image.coords[dim]) > 1 and np.issubdtype(
                        image.coords[dim].dtype, np.number
                    ):
                        scale_transform.append(
                            float(image.coords[dim][1] - image.coords[dim][0])
                        )
                    else:
                        scale_transform.append(1.0)
                    if len(image.coords[dim]) > 0 and np.issubdtype(
                        image.coords[dim].dtype, np.number
                    ):
                        translate_transform.append(float(image.coords[dim][0]))
                    else:
                        translate_transform.append(0.0)

                ngff_datasets.append(
                    {
                        "path": f"{self._dt[child].name}/{name}",
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

            image = scale0.ds
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
                    "@type": "ngff:Image",
                    "version": "0.4",
                    "name": name,
                    "axes": axes,
                    "datasets": ngff_datasets,
                }
            )

        # NGFF v0.4 metadata
        ngff_metadata = {"multiscales": multiscales, "multiscaleSpatialImageVersion": 1}
        self._dt.ds = self._dt.ds.assign_attrs(**ngff_metadata)

        self._dt.to_zarr(store, mode=mode, **kwargs)
