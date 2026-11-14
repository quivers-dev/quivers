"""Block handler: hide sites from the handlers outside it.

`BlockHandler` answers hidden sites itself, drawing and scoring them
exactly as the run's default handler would, so no handler outside it
sees their requests: an outer trace does not record them and an outer
clamp cannot condition them. Every other site forwards.
"""

from __future__ import annotations

from quivers.effects.base import EffectHandler, Installation, RunContext
from quivers.qiec.builtins import block_handler


class BlockHandler(EffectHandler):
    """Hide sites from outer handlers.

    Parameters
    ----------
    hide : list[str] or None
        Site names to hide; every site when both arguments are absent.
    expose : list[str] or None
        Site names to leave visible, hiding every other.

    Raises
    ------
    ValueError
        If both ``hide`` and ``expose`` are given.
    """

    def __init__(
        self, hide: list[str] | None = None, expose: list[str] | None = None
    ) -> None:
        if hide is not None and expose is not None:
            raise ValueError(
                "BlockHandler: pass at most one of `hide` and `expose`, not both."
            )
        self.hide = list(hide) if hide is not None else None
        self.expose = list(expose) if expose is not None else None

    def _should_block(self, name: object) -> bool:
        """Whether a site is hidden.

        Parameters
        ----------
        name : object
            The site label.

        Returns
        -------
        bool
            True when the site is hidden from outer handlers.
        """
        if self.expose is not None:
            return name not in self.expose
        if self.hide is None:
            return True
        return name in self.hide

    def install(self, run: RunContext) -> tuple[Installation, ...]:
        """Install a blocking handler of the ``random`` instance.

        Parameters
        ----------
        run : RunContext
            The run being prepared.

        Returns
        -------
        tuple[Installation, ...]
            The blocking handler.
        """
        kernel = run.kernel
        return (
            Installation(
                instance=kernel.random,
                runtime=block_handler(
                    self._should_block,
                    score_instance=kernel.score.entry.instance,
                    result_validator=run.validator,
                    answer_type=run.result_type,
                    key=f"block-{id(self):x}",
                ),
            ),
        )


def block(
    hide: list[str] | None = None,
    expose: list[str] | None = None,
) -> BlockHandler:
    """Return a `BlockHandler` hiding or exposing the named sites.

    Parameters
    ----------
    hide : list[str] or None
        Site names to hide.
    expose : list[str] or None
        Site names to leave visible.

    Returns
    -------
    BlockHandler
        The handler.
    """
    return BlockHandler(hide=hide, expose=expose)


__all__ = ["BlockHandler", "block"]
