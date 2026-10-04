"""Les ions, et la dissociation d'un electrolyte.

Un atome a qui l'on arrache des electrons devient un **ion positif**, ou
**cation** ; s'il en recoit, un **ion negatif**, ou **anion**. Sa
**valence** est le nombre de charges elementaires qu'il porte, signe
compris.

Dissous dans un solvant convenable, un **electrolyte** voit une partie
de ses molecules se scinder en cations et en anions. La molecule etait
neutre : les charges de ses ions se compensent exactement.

Source : `Electromagnetisme/SYLLABUS/Syllabus_Electricite_generale.pdf`,
paragraphes 1.3 et 1.4, pages 1-4 a 1-7 (exemple 1.4.2, acide sulfurique).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Ion:
    """Un ion : son symbole et sa valence, en charges elementaires."""

    symbole: str
    valence: int

    def __post_init__(self) -> None:
        """Refuse un ion sans charge : ce serait un atome."""
        if self.valence == 0:
            msg = f"valence nulle : {self.symbole} n'est pas un ion"
            raise ValueError(msg)

    @property
    def est_un_cation(self) -> bool:
        """Un ion positif."""
        return self.valence > 0

    @property
    def est_un_anion(self) -> bool:
        """Un ion negatif."""
        return self.valence < 0


@dataclass(frozen=True, slots=True)
class Dissociation:
    """Une molecule neutre et les ions en lesquels elle se scinde."""

    formule: str
    produits: tuple[tuple[int, Ion], ...]

    def __post_init__(self) -> None:
        """Refuse une dissociation qui creerait de la charge."""
        if self.charges_positives != self.charges_negatives:
            msg = (
                f"electroneutralite violee pour {self.formule} : "
                f"+{self.charges_positives} et -{self.charges_negatives}"
            )
            raise ValueError(msg)

    @property
    def charges_positives(self) -> int:
        """Le nombre de charges elementaires portees par les cations."""
        return sum(n * ion.valence for n, ion in self.produits if ion.est_un_cation)

    @property
    def charges_negatives(self) -> int:
        """Le nombre de charges elementaires portees par les anions."""
        return sum(-n * ion.valence for n, ion in self.produits if ion.est_un_anion)
