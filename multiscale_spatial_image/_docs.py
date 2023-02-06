from textwrap import dedent
from typing import Any, Callable


def inject_docs(**kwargs: Any) -> Callable[..., Any]:
    # taken from scanpy
    def decorator(obj: Any) -> Any:
        obj.__doc__ = dedent(obj.__doc__).format(**kwargs)
        return obj

    return decorator
