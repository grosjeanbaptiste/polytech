"""Chapitre XIV.2 : le « Branch and Bound », procedure par separation et evaluation.

Trois elements, pour un probleme de **minimisation** `min F(x)`, `x` dans `S` :

1. une **separation** : subdiviser un sous-ensemble en sous-ensembles qui
   n'eliminent aucune solution (regle de conservation), en nombre fini ;
2. une **evaluation** : une borne inferieure de `F` sur le sous-ensemble,
   et les tests de sondage qui en decoulent (`sondage`) ;
3. un **cheminement** : dans quel ordre examiner les noeuds.

Deux cheminements. **En profondeur d'abord** : on descend toujours dans
le sous-ensemble le plus a gauche, et quand un noeud est sonde on
remonte au premier noeud dont un fils reste a examiner. **La meilleure
evaluation d'abord** : quand un noeud est separe tous ses fils sont
evalues, et le noeud actif de plus petite evaluation est separe a son
tour ; `v* = min v_i` encadre alors l'optimum, `v* <= F~ <= F^`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.optimisation.domain.sondage import (
    Evaluation,
    Exploration,
    Noeud,
    Sondage,
    sonder,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


NOEUDS_MAXIMAUX: Final = 500
"""Garde-fou : la regle de finitude est due par la separation, non par l'explorateur."""


class Cheminement(Enum):
    """Les deux regles de cheminement les plus appliquees (XIV.2.3)."""

    PROFONDEUR_D_ABORD = "en profondeur d'abord, avec retour en arriere"
    MEILLEURE_EVALUATION_D_ABORD = "le noeud actif de plus petite evaluation"


@dataclass(frozen=True, slots=True)
class Procedure[P, S]:
    """L'evaluation et la separation d'un probleme ; au besoin un minorant.

    `minorer` rend une borne inferieure bon marche d'un noeud : quand elle
    suffit au test gamma, le noeud est sonde sans etre evalue. `connue` est
    une solution admissible trouvee d'avance, par une heuristique, et sa
    valeur : une bonne borne superieure elague des le debut (XIV.3.5 a).
    """

    evaluer: Callable[[P], Evaluation[S]]
    separer: Callable[[P, Evaluation[S]], Sequence[P]]
    minorer: Callable[[P], sp.Expr | None] | None = None
    connue: tuple[S, sp.Expr] | None = None


@dataclass(slots=True)
class _Registre[P, S]:
    """Ce que l'exploration accumule : la meilleure solution et les noeuds vus."""

    evaluer: Callable[[P], Evaluation[S]]
    minorer: Callable[[P], sp.Expr | None] | None = None
    noeuds_maximaux: int = NOEUDS_MAXIMAUX
    solution: S | None = None
    valeur: sp.Expr | None = None
    noeuds: list[Noeud[P]] = field(default_factory=list)

    def examiner(self, probleme: P, chemin: tuple[int, ...]) -> Evaluation[S] | None:
        """Evalue le noeud, le consigne ; rend l'evaluation s'il reste a separer."""
        if len(self.noeuds) >= self.noeuds_maximaux:
            msg = f"arborescence non elaguee : plus de {self.noeuds_maximaux} noeuds"
            raise RuntimeError(msg)
        borne = self.minorer(probleme) if self.minorer is not None else None
        if borne is not None and self.valeur is not None and self.valeur <= borne:
            self.noeuds.append(Noeud(probleme, chemin, borne, Sondage.DOMINANCE))
            return None
        evaluation = self.evaluer(probleme)
        sondage = sonder(evaluation, self.valeur)
        self.noeuds.append(Noeud(probleme, chemin, evaluation.valeur, sondage))
        admissible = sondage is Sondage.OPTIMALITE
        if admissible and (self.valeur is None or evaluation.valeur < self.valeur):
            self.solution, self.valeur = evaluation.solution, evaluation.valeur
        return evaluation if sondage is None else None


def explorer[P, S](
    racine: P,
    procedure: Procedure[P, S],
    cheminement: Cheminement,
    noeuds_maximaux: int = NOEUDS_MAXIMAUX,
) -> Exploration[P, S]:
    """Deroule la procedure depuis la racine, selon le cheminement choisi."""
    registre: _Registre[P, S] = _Registre(
        procedure.evaluer, procedure.minorer, noeuds_maximaux
    )
    if procedure.connue is not None:
        registre.solution, registre.valeur = procedure.connue
    if cheminement is Cheminement.PROFONDEUR_D_ABORD:
        _en_profondeur(registre, racine, procedure.separer)
    else:
        _la_meilleure_d_abord(registre, racine, procedure.separer)
    return Exploration(registre.solution, registre.valeur, tuple(registre.noeuds))


def _en_profondeur[P, S](
    registre: _Registre[P, S],
    racine: P,
    separer: Callable[[P, Evaluation[S]], Sequence[P]],
) -> None:
    pile: list[tuple[P, tuple[int, ...]]] = [(racine, ())]
    while pile:
        probleme, chemin = pile.pop()
        evaluation = registre.examiner(probleme, chemin)
        if evaluation is not None:
            fils = separer(probleme, evaluation)
            numerotes = [(f, (*chemin, rang)) for rang, f in enumerate(fils, start=1)]
            pile.extend(reversed(numerotes))


def _la_meilleure_d_abord[P, S](
    registre: _Registre[P, S],
    racine: P,
    separer: Callable[[P, Evaluation[S]], Sequence[P]],
) -> None:
    actifs: list[tuple[int, Evaluation[S]]] = []
    premiere = registre.examiner(racine, ())
    if premiere is not None:
        actifs.append((0, premiere))
    while actifs:
        actifs.sort(key=lambda actif: sp.sympify(actif[1].valeur), reverse=True)
        rang, evaluation = actifs.pop()
        noeud = registre.noeuds[rang]
        if sonder(evaluation, registre.valeur) is Sondage.DOMINANCE:
            registre.noeuds[rang] = replace(noeud, sondage=Sondage.DOMINANCE)
            continue
        for numero, fils in enumerate(separer(noeud.probleme, evaluation), start=1):
            suite = registre.examiner(fils, (*noeud.chemin, numero))
            if suite is not None:
                actifs.append((len(registre.noeuds) - 1, suite))
