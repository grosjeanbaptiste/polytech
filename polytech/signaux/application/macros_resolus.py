"""Macros des exercices resolus : les deux moities, reunies."""

from __future__ import annotations

from polytech.signaux.application.macros_resolus_signaux import (
    macros_resolus_signaux,
)
from polytech.signaux.application.macros_resolus_systemes import (
    macros_resolus_systemes,
)


def macros_resolus() -> dict[str, str]:
    """Les macros des trente-cinq exercices."""
    return {**macros_resolus_signaux(), **macros_resolus_systemes()}
