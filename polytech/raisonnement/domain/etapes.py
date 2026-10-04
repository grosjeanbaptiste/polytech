"""Une etape justifiee, et le raisonnement qu'elles forment.

« Rien ne doit tomber du ciel » : une etape dit ce que l'on obtient,
**pourquoi** le pas est licite, et ou le cours l'etablit. La
justification est obligatoire -- une etape qui affirme sans dire
pourquoi n'enseigne rien de transposable, et sa construction est
refusee. La source est facultative : certaines etapes ne sont que du
calcul.

Un raisonnement repond a une question par une suite d'etapes et une
conclusion. Deux raisonnements s'enchainent : c'est ainsi qu'un cours
s'appuie sur un autre, la conclusion du premier servant de depart au
second.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Etape:
    """Ce que l'on obtient, pourquoi c'est licite, et ou le cours le dit."""

    affirmation: str
    justification: str
    source: str = ""

    def __post_init__(self) -> None:
        """Refuse l'etape muette et l'etape qui tombe du ciel."""
        if not self.affirmation.strip():
            msg = "etape qui n'affirme rien"
            raise ValueError(msg)
        if not self.justification.strip():
            msg = f"etape sans justification : {self.affirmation!r}"
            raise ValueError(msg)

    def en_texte(self) -> str:
        """`affirmation -- parce que justification [source]`."""
        reference = f" [{self.source}]" if self.source else ""
        return f"{self.affirmation} -- parce que {self.justification}{reference}"


@dataclass(frozen=True, slots=True)
class Raisonnement:
    """Une question, les etapes qui y repondent, et la conclusion."""

    question: str
    etapes: tuple[Etape, ...]
    conclusion: str

    def __post_init__(self) -> None:
        """Refuse un raisonnement sans etape ou sans conclusion."""
        if not self.etapes:
            msg = f"aucune etape pour repondre a {self.question!r}"
            raise ValueError(msg)
        if not self.conclusion.strip():
            msg = f"raisonnement sans conclusion : {self.question!r}"
            raise ValueError(msg)

    @property
    def sources(self) -> tuple[str, ...]:
        """Les sources citees, chacune une fois, dans l'ordre d'apparition."""
        return tuple(dict.fromkeys(e.source for e in self.etapes if e.source))

    def en_texte(self) -> str:
        """La question, les etapes numerotees, la conclusion."""
        lignes = [self.question]
        lignes += [f"{rang}. {e.en_texte()}" for rang, e in enumerate(self.etapes, 1)]
        lignes.append(f"Donc : {self.conclusion}")
        return "\n".join(lignes)

    def puis(self, suite: Raisonnement) -> Raisonnement:
        """Ce raisonnement prolonge par un autre : sa conclusion devient un depart."""
        return Raisonnement(
            self.question, (*self.etapes, *suite.etapes), suite.conclusion
        )
