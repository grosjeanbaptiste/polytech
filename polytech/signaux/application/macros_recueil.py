"""Macros du recueil d'exercices : les deux moities, reunies."""

from __future__ import annotations

from polytech.signaux.application.macros_recueil_signaux import (
    macros_du_recueil_signaux,
)
from polytech.signaux.application.macros_recueil_systemes import (
    macros_du_recueil_systemes,
)


def macros_du_recueil() -> dict[str, str]:
    """Les macros des dix-neuf planches."""
    return {**macros_du_recueil_signaux(), **macros_du_recueil_systemes()}
