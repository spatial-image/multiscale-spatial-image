# Copilot Instructions for multiscale-spatial-image

## Project Overview

This library generates multiscale, chunked, multi-dimensional spatial image data
structures serializable to OME-NGFF format. It's built on Xarray DataTree with
spatial-image Dataset nodes, creating image pyramids for visualization and
analysis.

## Core Architecture

### Key Components

- **MultiscaleSpatialImage**: Xarray DataTree accessor
  (`@register_datatree_accessor("msi")`) providing NGFF-compatible multiscale
  operations
- **to_multiscale()**: Main entry point for creating pyramids with configurable
  downsampling methods
- **Methods enum**: Defines downsampling algorithms (XARRAY*COARSEN, ITK*_,
  DASK*IMAGE*_)
- **Operations module**: Provides dataset operations that skip non-dimensional
  nodes

### Data Structure Pattern

```python
# DataTree structure: /scale0, /scale1, /scale2, etc.
# Each scale contains Dataset with same variable name as input
multiscale = to_multiscale(image, [2, 4])  # Creates 3 scales total
multiscale['scale0'].ds  # Original resolution
multiscale['scale1'].ds  # 2x downsampled
multiscale['scale2'].ds  # 8x downsampled (2*4)
```

### Zarr Integration Patterns

- **Always use `dimension_separator='/'`** for NGFF compliance
- Support both Zarr v2 (DirectoryStore) and v3 (LocalStore) with fallback
  pattern in `_data.py`
- NGFF metadata automatically added in `to_zarr()` method with coordinate
  transformations

## Development Workflows

### Environment Management (Pixi-based)

```bash
pixi install -a                    # Install all environments
pixi run -e test test              # Run unit tests
pixi run test-notebooks            # Test example notebooks
pixi shell                         # Activate development shell
```

### Testing Patterns

- **Baseline comparison testing**: Use `verify_against_baseline()` for image
  output validation
- **Test data**: IPFS-hosted via pooch, SHA256-verified downloads
- **Notebook testing**: nbmake integration tests all examples
- **Multiple backends**: Tests run against ITK, dask-image, and xarray methods

### Adding Test Data

```python
# Temporary add this line to generate new baseline:
store_new_image(dataset_name, baseline_name, multiscale)
# Remove after running tests, then update data.tar.gz and SHA256 in _data.py
```

## Critical Patterns

### Dimension Handling

- **skip_non_dimension_nodes decorator**: Essential for operations on DataTree
  root nodes without dimensions
- **Default chunking**: Uses 64 for 3D (with 'z'), 256 for 2D, 1 for 't'
  dimension
- **Coordinate transforms**: Scale and translation automatically computed from
  coordinate spacing

### Multi-Backend Support

```python
# Pattern for optional dependencies
try:
    from zarr.storage import DirectoryStore  # Zarr v2
except ImportError:
    from zarr.storage import LocalStore     # Zarr v3
```

### Downsampling Method Dispatch

- Each method in `to_multiscale/` subdirectory follows `_downsample_{method}`
  naming
- Methods handle chunking alignment via `_align_chunks()` helper
- Scale factors can be uniform int or per-dimension dict:
  `[2, {'x': 2, 'y': 4}]`

## Integration Points

### External Dependencies

- **spatial-image**: Base SpatialImage input type
- **xarray-datatree**: Core DataTree functionality
- **OME-NGFF**: Metadata standard compliance via `multiscales` attribute
- **Optional**: ITK (medical imaging), dask-image (distributed), pyimagej

### File Patterns

- `multiscale_spatial_image.py`: Core accessor class with to_zarr() and
  operations
- `to_multiscale/`: Downsampling method implementations
- `operations/`: Dataset operations with dimension-aware decorators
- `examples/`: Jupyter notebooks demonstrating usage patterns

## Key Conventions

- Use `promote_attrs=True` when converting DataArrays to Datasets to preserve
  metadata
- Coordinate names follow spatial conventions: 't' (time), 'c' (channel),
  'x'/'y'/'z' (space)
- Error handling validates scale factors against current image dimensions before
  processing
- NGFF axes metadata includes type classification (time/channel/space) and
  optional units
