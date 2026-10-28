"""L'atome : numero atomique, nombre de masse, isotopes.

Un atome du tableau de Mendeleev, de **numero atomique** `Z` et de
**nombre de masse** `A`, est constitue de `Z` protons, `Z` electrons et
`A - Z` neutrons. A l'etat normal il est electriquement neutre. Deux
configurations qui ne different que par le nombre de neutrons -- donc par
`A` -- sont des **isotopes**.

Source : `Electromagnetisme/SYLLABUS/Syllabus_Electricite_generale.pdf`,
paragraphe 1.3, page 1-4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

RAYON_DE_BOHR: Final = 5.29177210903e-11
"""`a_0`, rayon de l'atome d'hydrogene, en metres (CODATA 2018)."""


@dataclass(frozen=True, slots=True)
class Atome:
    """Un atome neutre : son symbole, `Z` et `A`."""

    symbole: str
    numero_atomique: int
    nombre_de_masse: int

    def __post_init__(self) -> None:
        """Refuse un noyau qui aurait moins de nucleons que de protons."""
        if self.numero_atomique < 1 or self.nombre_de_masse < self.numero_atomique:
            msg = (
                f"nombre de masse {self.nombre_de_masse} inferieur au numero "
                f"atomique {self.numero_atomique}, ou numero atomique nul"
            )
            raise ValueError(msg)

    @property
    def protons(self) -> int:
        """`Z` protons dans le noyau."""
        return self.numero_atomique

    @property
    def electrons(self) -> int:
        """`Z` electrons autour du noyau : l'atome est neutre."""
        return self.numero_atomique

    @property
    def neutrons(self) -> int:
        """`A - Z` neutrons dans le noyau."""
        return self.nombre_de_masse - self.numero_atomique

    def est_isotope_de(self, autre: Atome) -> bool:
        """Meme element, nombre de neutrons different."""
        meme_element = self.numero_atomique == autre.numero_atomique
        return meme_element and self.nombre_de_masse != autre.nombre_de_masse
