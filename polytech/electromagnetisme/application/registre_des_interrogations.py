"""Les jeux de macros des interrogations cotees, annee par annee.

Le registre des travaux d'electromagnetisme depassait cent lignes : les
interrogations cotees -- une par annee et par site -- en formaient la
moitie, et elles ont leur propre regle de nommage (`exs-cote-<annee>`,
suivi du site quand il y en a deux). Elles vivent ici ; `registre.py` les
reprend telles quelles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from polytech.electromagnetisme.application import (
    macros_cote_2016_a,
    macros_cote_2016_b,
    macros_cote_2017,
    macros_cote_2018,
    macros_cote_2019,
    macros_cote_2021_mons,
    macros_cote_2022_charleroi,
    macros_cote_2022_mons,
    macros_cote_2023_charleroi,
    macros_cote_2023_mons,
    macros_cote_2024_charleroi,
    macros_cote_2024_mons,
    macros_cote_2025_charleroi,
    macros_cote_2025_mons,
    macros_cote_2026_charleroi,
    macros_cote_2026_mons,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _jeu_cote_seize() -> dict[str, str]:
    """Les six questions de l'exercice cote de 2016, en deux modules."""
    return {
        **macros_cote_2016_a.macros_cote_seize_debut(),
        **macros_cote_2016_b.macros_cote_seize_fin(),
    }


JEUX_DES_INTERROGATIONS: Final[dict[str, Callable[[], dict[str, str]]]] = {
    "exs-cote-2015-2016": _jeu_cote_seize,
    "exs-cote-2016-2017": macros_cote_2017.macros_cote_dix_sept,
    "exs-cote-2017-2018": macros_cote_2018.macros_cote_dix_huit,
    "exs-cote-2018-2019": macros_cote_2019.macros_cote_dix_neuf,
    "exs-cote-2021-2022-charleroi": macros_cote_2022_charleroi.macros_cote_vingt_deux,
    "exs-cote-2022-2023-charleroi": macros_cote_2023_charleroi.macros_cote_vingt_trois,
    "exs-cote-2023-2024-charleroi": macros_cote_2024_charleroi.macros_cote_vingt_quatre,
    "exs-cote-2024-2025-charleroi": macros_cote_2025_charleroi.macros_cote_vingt_cinq,
    "exs-cote-2025-2026-charleroi": macros_cote_2026_charleroi.macros_cote_vingt_six,
    "exs-cote-2020-2021-mons": macros_cote_2021_mons.macros_cote_mons_vingt_et_un,
    "exs-cote-2021-2022-mons": macros_cote_2022_mons.macros_cote_mons_vingt_deux,
    "exs-cote-2022-2023-mons": macros_cote_2023_mons.macros_cote_mons_vingt_trois,
    "exs-cote-2023-2024-mons": macros_cote_2024_mons.macros_cote_mons_vingt_quatre,
    "exs-cote-2024-2025-mons": macros_cote_2025_mons.macros_cote_mons_vingt_cinq,
    "exs-cote-2025-2026-mons": macros_cote_2026_mons.macros_cote_mons_vingt_six,
}
"""Le nom du dossier de travail designe le jeu qui l'alimente."""
