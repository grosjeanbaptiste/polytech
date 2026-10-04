"""Chapitre I.2.1 : deux des trois modeles classiques de la programmation lineaire.

**Planification de production** : `m` machines fabriquent `n` produits ;
la machine `i` dispose de `d_i` unites de temps, le produit `j` en
occupe `t_ij` et rapporte `c_j`. On maximise le gain total.

**Melange**, ou probleme dietetique : `n` aliments de prix `c_j`
apportent `t_ij` de l'element nutritif `i`, dont il faut au moins `d_i`.
On minimise le cout. Une contrainte de **teneur** -- la part de cendres
du melange ne depasse pas `C` -- porte sur un rapport ; multipliee par la
quantite totale, elle redevient lineaire :
`somme(c_i x_i) <= C somme(x_i)`, soit `somme((c_i - C) x_i) <= 0`.

Le troisieme modele, le transport, a sa structure propre : `transport`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.formulation import ProgrammeGeneral

if TYPE_CHECKING:
    from collections.abc import Sequence


def planification_de_production(
    gains: Sequence[object],
    capacites: Sequence[object],
    temps: Sequence[Sequence[object]],
) -> ProgrammeGeneral:
    """`max c x`, `T x <= d`, `x >= 0` : une contrainte par machine."""
    if len(temps) != len(capacites):
        msg = (
            f"une ligne de temps par machine : {len(temps)} lignes "
            f"pour {len(capacites)} capacites"
        )
        raise ValueError(msg)
    contraintes = tuple(
        Contrainte(tuple(ligne), Sens.INFERIEUR, capacite)
        for ligne, capacite in zip(temps, capacites, strict=True)
    )
    return ProgrammeGeneral(Optimisation.MAX, tuple(gains), contraintes)


def probleme_de_melange(
    prix: Sequence[object],
    apports: Sequence[Sequence[object]],
    besoins: Sequence[object],
) -> ProgrammeGeneral:
    """`min c x`, `T x >= d`, `x >= 0` : une contrainte par element nutritif."""
    contraintes = tuple(
        Contrainte(tuple(ligne), Sens.SUPERIEUR, besoin)
        for ligne, besoin in zip(apports, besoins, strict=True)
    )
    return ProgrammeGeneral(Optimisation.MIN, tuple(prix), contraintes)


def contrainte_de_teneur(
    teneurs: Sequence[object],
    limite: object,
    sens: Sens,
) -> Contrainte:
    """La teneur du melange comparee a `limite`, rendue lineaire."""
    reference = Contrainte((limite,), sens, 0).coefficients[0]
    exactes = Contrainte(tuple(teneurs), sens, 0).coefficients
    return Contrainte(tuple(t - reference for t in exactes), sens, 0)
