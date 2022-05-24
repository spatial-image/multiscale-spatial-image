"""multiscale-spatial-image

Generate a multiscale spatial image."""

__version__ = "0.9.0"

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

    This is an xarray DataTree, with content compatible with the Open Microscopy Environment-
    Next Generation File Format (OME-NGFF).

    The tree contains nodes in the form: `scale{scale}` where *scale* is the integer scale.
    Each node has a the same named `Dataset` that corresponds to to the NGFF dataset name.
     For example, a three-scale representation of a *cells* dataset would have `Dataset` nodes:

      scale0
      scale1
      scale2
    """

    def __init__(
        self,
        name: str = "multiscales",
        data: Union[xr.Dataset, xr.DataArray] = None,
        parent: TreeNode = None,
        children: List[TreeNode] = None,
    ):
        """DataTree with a root name of *multiscales*."""
        super().__init__(data=data, name=name, parent=parent, children=children)

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
            ``{"scale0/image": {"my_variable": {"dtype": "int16", "scale_factor": 0.1}, ...}, ...}``.
            See ``xarray.Dataset.to_zarr`` for available options.
        kwargs :
            Additional keyword arguments to be passed to ``datatree.DataTree.to_zarr``
        """

        multiscales = []
        scale0 = self[self.groups[1]]
        for name in scale0.ds.data_vars.keys():

            ngff_datasets = []
            for child in self.children:
                image = self[child].ds
                scale_transform = []
                translate_transform = []
                for dim in image.dims:
                    if len(image.coords[dim]) > 1 and np.issubdtype(image.coords[dim].dtype, np.number):
                        scale_transform.append(
                            float(image.coords[dim][1] - image.coords[dim][0])
                        )
                    else:
                        scale_transform.append(1.0)
                    if len(image.coords[dim]) > 0 and np.issubdtype(image.coords[dim].dtype, np.number):
                        translate_transform.append(float(image.coords[dim][0]))
                    else:
                        translate_transform.append(0.0)

                ngff_datasets.append(
                    {
                        "path": f"{self[child].name}/{name}",
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
        self.ds.attrs = ngff_metadata

        super().to_zarr(store, **kwargs)


class Methods(Enum):
    XARRAY_COARSEN = "xarray.DataArray.coarsen"
    ITK_BIN_SHRINK = "itk.bin_shrink_image_filter"
    ITK_GAUSSIAN = "itk.discrete_gaussian_image_filter"
    ITK_LABEL_GAUSSIAN = "itk.discrete_gaussian_image_filter_label_interpolator"
    DASK_IMAGE_GAUSSIAN = "dask_image.ndfilters.gaussian_filter"


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

    method : multiscale_spatial_image.Methods, optional
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
    data_objects = {f"scale0": current_input.to_dataset(name=image.name, promote_attrs=True)}

    if method is None:
        method = Methods.XARRAY_COARSEN

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

    def get_block(xarray_image, block_index:int):
        '''Helper method for accessing an enumerated chunk from xarray input'''
        block_shape = [c[block_index] for c in current_input.chunks]
        block = current_input[tuple([slice(0, s) for s in block_shape])]
        # For consistency for now, do not utilize direction until there is standardized support for
        # direction cosines / orientation in OME-NGFF
        block.attrs.pop("direction", None)
        return block
        
    def compute_input_spacing(input_image):
        '''Helper method to manually compute image spacing. Assumes even spacing along any axis.
           
        input_image: xarray.core.dataarray.DataArray
            The image for which voxel spacings are computed

        result: Dict
            Spacing along each enumerated image axis
            Example {'x': 1.0, 'y': 0.5}
        '''
        return {dim: float(input_image.coords[dim][1]) - float(input_image.coords[dim][0])
                for dim in input_image.dims}

    def compute_output_spacing(input_image, dim_factors):
        '''Helper method to manually compute output image spacing.
           
        input_image: xarray.core.dataarray.DataArray
            The image for which voxel spacings are computed

        dim_factors: Dict
            Shrink ratio along each enumerated axis

        result: Dict
            Spacing along each enumerated image axis
            Example {'x': 2.0, 'y': 1.0}
        '''
        input_spacing = compute_input_spacing(input_image)
        return {dim: input_spacing[dim] * dim_factors[dim] for dim in input_image.dims}

    def compute_output_origin(input_image, dim_factors):    
        '''Helper method to manually compute output image physical offset.
           Note that this method does not account for an image direction matrix.
           
        input_image: xarray.core.dataarray.DataArray
            The image for which voxel spacings are computed

        dim_factors: Dict
            Shrink ratio along each enumerated axis

        result: Dict
            Offset in physical space of first voxel in output image
            Example {'x': 0.5, 'y': 1.0}
        '''
        import math
        image_dims: Tuple[str, str, str, str] = ("x", "y", "z", "t")
            
        input_spacing = compute_input_spacing(input_image)
        input_origin = {dim: float(input_image.coords[dim][0])
                          for dim in image_dims if dim in dim_factors}

        # Index in input image space corresponding to offset after shrink
        input_index = {dim: 0.5 * (dim_factors[dim] - 1)
                                  for dim in image_dims if dim in dim_factors}
        # Translate input index coordinate to offset in physical space
        # NOTE: This method fails to account for direction matrix
        return {dim: input_index[dim] * input_spacing[dim] + input_origin[dim]
                          for dim in image_dims if dim in dim_factors}

    def compute_sigma(input_spacings, shrink_factors) -> list:
        '''Compute Gaussian kernel sigma values for resampling to isotropic spacing.
        sigma = sqrt((isoSpacing^2 - inputSpacing[0]^2)/(2*sqrt(2*ln(2)))^2)
        Ref https://discourse.itk.org/t/resampling-to-isotropic-signal-processing-theory/1403/16

        input spacings: List
            Input image physical spacings in xyzt order

        shrink_factors: List
            Shrink ratio along each axis in xyzt order

        result: List
            Standard deviation of Gaussian kernel along each axis in xyzt order
        '''
        assert len(input_spacings) == len(shrink_factors)
        import math
        output_spacings = [input_spacing * shrink for input_spacing, shrink in zip(input_spacings, shrink_factors)]
        denominator = (2 * ((2 * math.log(2)) ** 0.5)) ** 2
        return [((output_spacing ** 2 - input_spacing ** 2) / denominator) ** 0.5
                for input_spacing, output_spacing in zip(input_spacings, output_spacings)]

    def compute_itk_gaussian_kernel_radius(input_size, sigma_values, shrink_factors) -> list:
        '''Get kernel radius in xyzt directions'''
        DEFAULT_MAX_KERNEL_WIDTH = 32
        MAX_KERNEL_ERROR = 0.01
        image_dimension = len(input_size)

        # Constrain kernel width to be at most the size of one chunk
        max_kernel_width = min(DEFAULT_MAX_KERNEL_WIDTH, *input_size)
        variance = [sigma ** 2 for sigma in sigma_values]

        def generate_radius(direction:int) -> int:
            '''Follow itk.DiscreteGaussianImageFilter procedure to generate directional kernels'''
            oper = itk.GaussianOperator[itk.F, image_dimension]()
            oper.SetDirection(direction)
            oper.SetMaximumError(MAX_KERNEL_ERROR)
            oper.SetMaximumKernelWidth(max_kernel_width)
            oper.SetVariance(variance[direction])
            oper.CreateDirectional()
            return oper.GetRadius(direction)

        return [generate_radius(dim) for dim in range(image_dimension)]

    def itk_blur_and_downsample(xarray_data, gaussian_filter_name, interpolator_name, shrink_factors, sigma_values, kernel_radius):
        '''Blur and then downsample a given image chunk'''
        import itk
        
        # xarray chunk does not have metadata attached, values are ITK defaults
        image = itk.image_view_from_array(xarray_data)
        input_origin = itk.origin(image)

        # Skip this image block if it has 0 voxels
        block_size = itk.size(image)
        if(any([block_len == 0 for block_len in block_size])):
            return None
        
        # Output values are relative to input
        itk_shrink_factors = shrink_factors  # xyzt
        itk_kernel_radius = kernel_radius
        output_origin = [val + radius for val, radius in zip(input_origin, itk_kernel_radius)]
        output_spacing = [s * f for s, f in zip(itk.spacing(image), itk_shrink_factors)]
        output_size = [max(0,int((image_len - 2 * radius) / shrink_factor))
            for image_len, radius, shrink_factor in zip(itk.size(image), itk_kernel_radius, itk_shrink_factors)]

        # Optionally run accelerated smoothing with itk-vkfft
        if gaussian_filter_name == 'VkDiscreteGaussianImageFilter':
            smoothing_filter_template = itk.VkDiscreteGaussianImageFilter
        elif gaussian_filter_name == 'DiscreteGaussianImageFilter':
            smoothing_filter_template = itk.DiscreteGaussianImageFilter
        else:
            raise ValueError(f'Unsupported gaussian_filter {gaussian_filter_name}')

        # Construct pipeline
        smoothing_filter = smoothing_filter_template.New(image, 
            sigma_array=sigma_values, 
            use_image_spacing=False)            

        if interpolator_name == 'LinearInterpolateImageFunction':
            interpolator_instance = itk.LinearInterpolateImageFunction.New(smoothing_filter.GetOutput())
        elif interpolator_name == 'LabelImageGaussianInterpolateImageFunction':
            interpolator_instance = itk.LabelImageGaussianInterpolateImageFunction.New(smoothing_filter.GetOutput())
            # Similar approach as compute_sigma
            # Ref: https://link.springer.com/content/pdf/10.1007/978-3-319-24571-3_81.pdf
            sigma = [s * 0.7355 for s in output_spacing]
            sigma_max = max(sigma)
            interpolator_instance.SetSigma(sigma)
            interpolator_instance.SetAlpha(sigma_max * 2.5)
        else:
            raise ValueError(f'Unsupported interpolator_name {interpolator_name}')
        
        shrink_filter = itk.ResampleImageFilter.New(smoothing_filter.GetOutput(),
            interpolator=interpolator_instance,
            size=output_size,
            output_spacing=output_spacing,
            output_origin=output_origin)
        shrink_filter.Update()
        
        return shrink_filter.GetOutput()

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

            data_objects[f"scale{factor_index+1}"] = downscaled.to_dataset(
                name=image.name, promote_attrs=True
            )
            current_input = downscaled
    elif method is Methods.ITK_BIN_SHRINK:
        import itk

        for factor_index, scale_factor in enumerate(scale_factors):
            dim_factors = dim_scale_factors(scale_factor)
            current_input = align_chunks(current_input, dim_factors)

            image_dims: Tuple[str, str, str, str] = ("x", "y", "z", "t")
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
            data_objects[f"scale{factor_index+1}"] = downscaled.to_dataset(
                name=image.name, promote_attrs=True
            )
            current_input = downscaled
    elif method is Methods.ITK_GAUSSIAN:
        import itk         

        # Optionally run accelerated smoothing with itk-vkfft
        if 'VkFFTBackend' in dir(itk):
            gaussian_filter_name = 'VkDiscreteGaussianImageFilter'
        else:
            gaussian_filter_name = 'DiscreteGaussianImageFilter'

        interpolator_name = 'LinearInterpolateImageFunction'

        for factor_index, scale_factor in enumerate(scale_factors):
            dim_factors = dim_scale_factors(scale_factor)
            current_input = align_chunks(current_input, dim_factors)

            image_dims: Tuple[str, str, str, str] = ("x", "y", "z", "t")
            shrink_factors = [dim_factors[sf] for sf in image_dims if sf in dim_factors]

            # Compute metadata for region splitting

            # Blocks 0, ..., N-2 have the same shape
            block_0_input = get_block(current_input,0)
            # Block N-1 may be smaller than preceding blocks
            block_neg1_input = get_block(current_input,-1)

            # Compute overlap for Gaussian blurring for all blocks
            block_0_image = itk.image_from_xarray(block_0_input)
            input_spacing = itk.spacing(block_0_image)
            sigma_values = compute_sigma(input_spacing, shrink_factors)
            kernel_radius = compute_itk_gaussian_kernel_radius(itk.size(block_0_image), sigma_values, shrink_factors)

            # Compute output size and spatial metadata for blocks 0, .., N-2
            filt = itk.BinShrinkImageFilter.New(
                block_0_image, shrink_factors=shrink_factors
            )
            filt.UpdateOutputInformation()
            block_output = filt.GetOutput()
            block_0_output_spacing = block_output.GetSpacing()
            block_0_output_origin = block_output.GetOrigin()
            
            block_0_scale = {
                image_dims[i]: s for (i, s) in enumerate(block_0_output_spacing)
            }
            block_0_translation = {
                image_dims[i]: s for (i, s) in enumerate(block_0_output_origin)
            }
            dtype = block_output.dtype
            
            computed_size = [int(block_len / shrink_factor) 
                for block_len, shrink_factor in zip(itk.size(block_0_image), shrink_factors)]
            assert all([itk.size(block_output)[dim] == computed_size[dim]
                        for dim in range(block_output.ndim)])
            output_chunks = list(current_input.chunks)
            for i, c in enumerate(output_chunks):
                output_chunks[i] = [
                    block_output.shape[i],
                ] * len(c)

            # Compute output size for block N-1
            block_neg1_image = itk.image_from_xarray(block_neg1_input)
            filt.SetInput(block_neg1_image)
            filt.UpdateOutputInformation()
            block_output = filt.GetOutput()
            computed_size = [int(block_len / shrink_factor) 
                for block_len, shrink_factor in zip(itk.size(block_neg1_image), shrink_factors)]
            assert all([itk.size(block_output)[dim] == computed_size[dim]
                        for dim in range(block_output.ndim)])
            for i, c in enumerate(output_chunks):
                output_chunks[i][-1] = block_output.shape[i]
                output_chunks[i] = tuple(output_chunks[i])
            output_chunks = tuple(output_chunks)

            downscaled_array = map_overlap(
              itk_blur_and_downsample,
              current_input.data,
              gaussian_filter_name=gaussian_filter_name,
              interpolator_name=interpolator_name,
              shrink_factors=shrink_factors,
              sigma_values=sigma_values,
              kernel_radius=kernel_radius,
              dtype=dtype,
              depth={dim: radius for dim, radius in enumerate(np.flip(kernel_radius))}, # overlap is in tzyx
              boundary='nearest',
              trim=False   # Overlapped region is trimmed in blur_and_downsample to output size
            ).compute()
            
            downscaled = to_spatial_image(
                downscaled_array,
                dims=image.dims,
                scale=block_0_scale,
                translation=block_0_translation,
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
            data_objects[f"scale{factor_index+1}"] = downscaled.to_dataset(
                name=image.name, promote_attrs=True
            )
            current_input = downscaled

    elif method is Methods.ITK_LABEL_GAUSSIAN:
        # Uses the LabelImageGaussianInterpolateImageFunction. More appropriate for integer label images.
        import itk         

        gaussian_filter_name = 'DiscreteGaussianImageFilter'
        interpolator_name = 'LabelImageGaussianInterpolateImageFunction'

        for factor_index, scale_factor in enumerate(scale_factors):
            dim_factors = dim_scale_factors(scale_factor)
            current_input = align_chunks(current_input, dim_factors)

            image_dims: Tuple[str, str, str, str] = ("x", "y", "z", "t")
            shrink_factors = [dim_factors[sf] for sf in image_dims if sf in dim_factors]

            # Compute metadata for region splitting

            # Blocks 0, ..., N-2 have the same shape
            block_0_input = get_block(current_input,0)
            # Block N-1 may be smaller than preceding blocks
            block_neg1_input = get_block(current_input,-1)

            # Compute overlap for Gaussian blurring for all blocks
            block_0_image = itk.image_from_xarray(block_0_input)
            input_spacing = itk.spacing(block_0_image)
            sigma_values = compute_sigma(input_spacing, shrink_factors)
            kernel_radius = compute_itk_gaussian_kernel_radius(itk.size(block_0_image), sigma_values, shrink_factors)

            # Compute output size and spatial metadata for blocks 0, .., N-2
            filt = itk.BinShrinkImageFilter.New(
                block_0_image, shrink_factors=shrink_factors
            )
            filt.UpdateOutputInformation()
            block_output = filt.GetOutput()
            block_0_output_spacing = block_output.GetSpacing()
            block_0_output_origin = block_output.GetOrigin()
            
            block_0_scale = {
                image_dims[i]: s for (i, s) in enumerate(block_0_output_spacing)
            }
            block_0_translation = {
                image_dims[i]: s for (i, s) in enumerate(block_0_output_origin)
            }
            dtype = block_output.dtype
            
            computed_size = [int(block_len / shrink_factor) 
                for block_len, shrink_factor in zip(itk.size(block_0_image), shrink_factors)]
            assert all([itk.size(block_output)[dim] == computed_size[dim]
                        for dim in range(block_output.ndim)])
            output_chunks = list(current_input.chunks)
            for i, c in enumerate(output_chunks):
                output_chunks[i] = [
                    block_output.shape[i],
                ] * len(c)

            # Compute output size for block N-1
            block_neg1_image = itk.image_from_xarray(block_neg1_input)
            filt.SetInput(block_neg1_image)
            filt.UpdateOutputInformation()
            block_output = filt.GetOutput()
            computed_size = [int(block_len / shrink_factor) 
                for block_len, shrink_factor in zip(itk.size(block_neg1_image), shrink_factors)]
            assert all([itk.size(block_output)[dim] == computed_size[dim]
                        for dim in range(block_output.ndim)])
            for i, c in enumerate(output_chunks):
                output_chunks[i][-1] = block_output.shape[i]
                output_chunks[i] = tuple(output_chunks[i])
            output_chunks = tuple(output_chunks)

            downscaled_array = map_overlap(
              itk_blur_and_downsample,
              current_input.data,
              gaussian_filter_name=gaussian_filter_name,
              interpolator_name=interpolator_name,
              shrink_factors=shrink_factors,
              sigma_values=sigma_values,
              kernel_radius=kernel_radius,
              dtype=dtype,
              depth={dim: radius for dim, radius in enumerate(np.flip(kernel_radius))}, # overlap is in tzyx
              boundary='nearest',
              trim=False   # Overlapped region is trimmed in blur_and_downsample to output size
            ).compute()
            
            downscaled = to_spatial_image(
                downscaled_array,
                dims=image.dims,
                scale=block_0_scale,
                translation=block_0_translation,
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
            data_objects[f"scale{factor_index+1}"] = downscaled.to_dataset(
                name=image.name, promote_attrs=True
            )
            current_input = downscaled

    elif method is Methods.DASK_IMAGE_GAUSSIAN:
        import dask_image.ndfilters
        import dask_image.ndinterp

        def get_truncate(xarray_image, sigma_values, truncate_start=4.0) -> float:
            '''Discover truncate parameter yielding a viable kernel width
               for dask_image.ndfilters.gaussian_filter processing. Block overlap
               cannot be greater than image size, so kernel radius is more limited
               for small images. A lower stddev truncation ceiling for kernel
               generation can result in a less precise kernel.

            xarray_image: xarray.core.dataarray.DataArray
               Chunked image to be smoothed

            sigma:values: List
               Gaussian kernel standard deviations in tzyx order

            truncate_start: float
               First truncation value to try.
               "dask_image.ndfilters.gaussian_filter" defaults to 4.0.

            result: float
               Truncation value found to yield largest possible kernel width without
               extending beyond one chunk such that chunked smoothing would fail.
            '''

            from dask_image.ndfilters._gaussian import _get_border

            truncate = truncate_start
            stddev_step = 0.5  # search by stepping down by 0.5 stddev in each iteration

            border = _get_border(xarray_image.data, sigma_values, truncate)
            while any([border_len > image_len for border_len, image_len in zip(border, xarray_image.shape)]):
                truncate = truncate - stddev_step
                if(truncate <= 0.0):
                    break                    
                border = _get_border(xarray_image.data, sigma_values, truncate)

            return truncate

        for factor_index, scale_factor in enumerate(scale_factors):
            dim_factors = dim_scale_factors(scale_factor)
            current_input = align_chunks(current_input, dim_factors)

            image_dims: Tuple[str, str, str, str] = ("x", "y", "z", "t")
            shrink_factors = [dim_factors[sf] for sf in image_dims if sf in dim_factors]

            # Compute/discover region splitting parameters
            input_spacing = compute_input_spacing(current_input)
            input_spacing = [input_spacing[dim] for dim in image_dims if dim in dim_factors]
            sigma_values = compute_sigma(input_spacing, shrink_factors)
            truncate = get_truncate(current_input, np.flip(sigma_values))

            # Compute output shape and metadata
            output_shape = [int(image_len / shrink_factor)
                for image_len, shrink_factor in zip(current_input.shape, np.flip(shrink_factors))]
            output_spacing = compute_output_spacing(current_input, dim_factors)
            output_origin = compute_output_origin(current_input, dim_factors)

            blurred_array = dask_image.ndfilters.gaussian_filter(
                image=current_input.data,
                sigma=np.flip(sigma_values), # tzyx order
                mode='nearest',
                truncate=truncate
            )

            # Construct downsample parameters
            image_dimension = len(dim_factors)
            transform = np.eye(image_dimension)
            for dim, shrink_factor in enumerate(np.flip(shrink_factors)):
                transform[dim,dim] = shrink_factor

            downscaled_array = dask_image.ndinterp.affine_transform(
                blurred_array,
                matrix=transform,
                output_shape=output_shape # tzyx order
            ).compute()
            
            downscaled = to_spatial_image(
                downscaled_array,
                dims=image.dims,
                scale=output_spacing,
                translation=output_origin,
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
            data_objects[f"scale{factor_index+1}"] = downscaled.to_dataset(
                name=image.name, promote_attrs=True
            )
            current_input = downscaled


    multiscale = MultiscaleSpatialImage.from_dict(
        d=data_objects
    )

    return multiscale


def itk_image_to_multiscale(
    image,
    scale_factors: Sequence[Union[Dict[str, int], int]],
    anatomical_axes: bool = False,
    axis_names: List[str] = None,
    axis_units: List[str] = None,
    name: List[str] = None,
    method: Optional[Methods] = None,
    chunks: Optional[
        Union[
            int,
            Tuple[int, ...],
            Tuple[Tuple[int, ...], ...],
            Mapping[Any, Union[None, int, Tuple[int, ...]]],
        ]
    ] = None) -> MultiscaleSpatialImage:

    import itk
    import numpy as np

    if not name:
        object_name = image.GetObjectName().strip()
        if object_name and not object_name.isspace():
            name = object_name
        else:
            name = 'image'

    # Handle anatomical axes
    if anatomical_axes and (axis_names is None):
        axis_names = {"x": "right-left", "y": "anterior-posterior", "z": "inferior-superior"}
    
    # Orient 3D image so that direction is identity wrt RAI coordinates
    image_dimension = image.GetImageDimension()
    input_direction = np.array(image.GetDirection())
    if anatomical_axes and image_dimension == 3 and not (np.eye(image_dimension) == input_direction).all():
        desired_orientation = itk.SpatialOrientationEnums.ValidCoordinateOrientations_ITK_COORDINATE_ORIENTATION_RAI
        oriented_image = itk.orient_image_filter(image, use_image_direction=True, desired_coordinate_orientation=desired_orientation)

    elif anatomical_axes and image_dimension != 3:
        raise ValueError(f'Cannot use anatomical axes for input image of size {image_dimension}')
        
    # TODO remove itk_helper when itk>5.3rc04 is available.
    # itk_helper is a workaround for memory reference fix in itk>5.3rc04.
    from itk_helper import xarray_from_image
    image_da = xarray_from_image(image)
    image_da.name = name

    image_dims: Tuple[str, str, str, str] = ("x", "y", "z", "t") # ITK dims are in xyzt order
    scale = {image_dims[i]: s for (i, s) in enumerate(image.GetSpacing())}
    translation = {image_dims[i]: s for (i, s) in enumerate(image.GetOrigin())}

    spatial_image = to_spatial_image(image_da,
                                     scale=scale,
                                     translation=translation,
                                     name=name,
                                     axis_names=axis_names,
                                     axis_units=axis_units)

    return to_multiscale(spatial_image, 
                         scale_factors, 
                         method=method, 
                         chunks=chunks)
