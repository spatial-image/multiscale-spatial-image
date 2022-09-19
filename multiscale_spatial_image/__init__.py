"""multiscale-spatial-image

Generate a multiscale spatial image."""


__all__ = [
  "MultiscaleSpatialImage",
  "Methods",
  "to_multiscale",
  "itk_image_to_multiscale",
  "__version__",
]

from .__about__ import __version__
from .multiscale_spatial_image import MultiscaleSpatialImage
from .to_multiscale import Methods, to_multiscale, itk_image_to_multiscale