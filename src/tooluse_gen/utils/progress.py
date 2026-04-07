"""Progress-bar utilities built on tqdm."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tqdm import tqdm


def create_progress_bar(
    total: int,
    description: str,
    disable: bool = False,
    unit: str = "conv",
    colour: str = "cyan",
    **kwargs: Any,
) -> tqdm:
    """Return a configured :class:`tqdm` progress bar.

    Args:
        total:       Total number of items to process.
        description: Label displayed to the left of the bar.
        disable:     Suppress output entirely (useful in quiet/CI mode).
        unit:        Item label shown in the rate display (default: ``"conv"``).
        colour:      Bar colour (tqdm supports a subset of ANSI colours).
        **kwargs:    Forwarded verbatim to :class:`tqdm`.

    Returns:
        A configured, **not yet started** :class:`tqdm` instance.  Use as a
        context manager or call ``.update()`` / ``.close()`` manually.

    Example::

        with create_progress_bar(100, "Generating") as pbar:
            for item in items:
                process(item)
                pbar.update(1)
    """
    return tqdm(
        total=total,
        desc=description,
        disable=disable,
        unit=unit,
        colour=colour,
        dynamic_ncols=True,
        **kwargs,
    )


def progress_callback(pbar: tqdm) -> Callable[[], None]:
    """Return a zero-argument callback that advances *pbar* by one step.

    Useful for APIs that accept a callback but don't call it with arguments::

        pbar = create_progress_bar(len(items), "Processing")
        for item in items:
            process(item, on_done=progress_callback(pbar))
        pbar.close()
    """

    def _cb() -> None:
        pbar.update(1)

    return _cb
