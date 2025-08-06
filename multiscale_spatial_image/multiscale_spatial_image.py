from typing import Union, Iterable, Any

from xarray import DataTree, register_datatree_accessor
import numpy as np
from collections.abc import MutableMapping, Hashable
from pathlib import Path
import zarr.storage
from multiscale_spatial_image.operations import (
    transpose,
    reindex_data_arrays,
    assign_coords,
)

# Zarr Python 3
if hasattr(zarr.storage, "StoreLike"):
    StoreLike = zarr.storage.StoreLike
else:
    StoreLike = Union[MutableMapping, str, Path, zarr.storage.BaseStore]


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
        store: StoreLike,
        mode: str = "w",
        encoding=None,
        **kwargs,
    ):
        """
        Write multi-scale spatial image contents to a Zarr store.

        Metadata is added according the OME-NGFF standard.

        store : StoreLike
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

        # Ensure zarr v2 format for NGFF v0.4
        if "zarr_format" not in kwargs:
            kwargs["zarr_format"] = 2
        self._dt.to_zarr(store, mode=mode, **kwargs)

    def transpose(self, *dims: Hashable) -> DataTree:
        """Return a `DataTree` with all dimensions of arrays in datasets transposed.

        This method automatically skips those nodes of the `DataTree` that do not contain
        dimensions. Note that for `Dataset`s themselves, the order of dimensions stays the same.
        In case of a `DataTree` node missing specified dimensions an error is raised.

        Parameters
        ----------
        *dims : Hashable | None
            If not specified, reverse the dimensions on each array. Otherwise,
            reorder the dimensions to the order that the `dims` are specified..
        """
        return self._dt.map_over_datasets(transpose, *dims)

    def reindex_data_arrays(
        self,
        indexers: dict[str, Any],
        method: str | None = None,
        tolerance: float | Iterable[float] | str | None = None,
        copy: bool = False,
        fill_value: int | dict[str, int] | None = None,
        **indexer_kwargs: Any,
    ):
        """
        Reindex the `DataArray`s present in the datasets at each scale level of the MultiscaleSpatialImage.

        From the original xarray docstring: Conform this object onto the indexes of another object, filling in missing
        values with fill_value. The default fill value is NaN.

        Parameters
        ----------
        indexers : dict | None
            Dictionary with keys given by dimension names and values given by arrays of coordinates tick labels.
            Any mis-matched coordinate values will be filled in with NaN, and any mis-matched dimension names will
            simply be ignored. One of indexers or indexers_kwargs must be provided.
        method : str | None
            Method to use for filling index values in indexers not found on this data array:
                - None (default): don’t fill gaps
                - pad / ffill: propagate last valid index value forward
                - backfill / bfill: propagate next valid index value backward
                - nearest: use nearest valid index value
        tolerance: float | Iterable[float] | str | None
            Maximum distance between original and new labels for inexact matches. The values of the index at the
            matching locations must satisfy the equation abs(index[indexer] - target) <= tolerance. Tolerance may
            be a scalar value, which applies the same tolerance to all values, or list-like, which applies variable
            tolerance per element. List-like must be the same size as the index and its dtype must exactly match the
            index’s type.
        copy : bool
            If copy=True, data in the return value is always copied. If copy=False and reindexing is unnecessary, or
            can be performed with only slice operations, then the output may share memory with the input. In either
            case, a new xarray object is always returned.
        fill_value: int | dict[str, int] | None
            Value to use for newly missing values. If a dict-like, maps variable names (including coordinates) to fill
            values. Use this data array’s name to refer to the data array’s values.
        **indexer_kwargs
            The keyword arguments form of indexers. One of indexers or indexers_kwargs must be provided.
        """
        return self._dt.map_over_datasets(
            reindex_data_arrays,
            indexers,
            method,
            tolerance,
            copy,
            fill_value,
            *indexer_kwargs,
        )

    def assign_coords(self, coords, **coords_kwargs):
        """
        Assign new coordinates to all `Dataset`s in the `DataTree` having dimensions.

        Returns a new `Dataset` at each scale level of the `MultiscaleSpatialImage` with all the original data in
        addition to the new coordinates.

        Parameters
        ----------
        coords
            A mapping whose keys are the names of the coordinates and values are the coordinates to assign.
            The mapping will generally be a dict or Coordinates.
                - If a value is a standard data value — for example, a DataArray, scalar, or array — the data is simply
                  assigned as a coordinate.
                - If a value is callable, it is called with this object as the only parameter, and the return value is
                  used as new coordinate variables.
                - A coordinate can also be defined and attached to an existing dimension using a tuple with the first
                  element the dimension name and the second element the values for this new coordinate.
        **coords_kwargs
            The keyword arguments form of coords. One of coords or coords_kwargs must be provided.
        """
        return self._dt.map_over_datasets(assign_coords, coords, *coords_kwargs)
