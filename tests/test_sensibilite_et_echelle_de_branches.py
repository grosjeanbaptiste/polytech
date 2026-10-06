"""La sensibilite de `Q` d'un Sallen-Key, et l'echelle faite de branches.

Sources : `ElectroniqueInstrumentation/SLIDES/Slides_3_part2.pdf`, planches
24 a 27 -- qui nomment deux methodes « a sensibilite moderee » et « a
sensibilite minimale » sans chiffrer ni l'une ni l'autre -- et
`Slides_3_part1.pdf` pour les echelles et leurs transformations.

La sensibilite est la definition classique, `S = (x / Q) dQ/dx` : une
erreur relative de 1 % sur `x` deplace `Q` de `S` %. Chaque valeur est
refaite ici par une difference finie sur la cellule, et par l'argument
d'echelle : multiplier toutes les resistances par `k` ne change pas `Q`,
donc leurs sensibilites se compensent -- de meme pour les capacites.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electronique.domain.chebyshev import elements_de_chebyshev
from polytech.electronique.domain.conversions_d_elements import (
    Branche,
    en_passe_bande,
)
from polytech.electronique.domain.echelle_de_branches import EchelleTerminee
from polytech.electronique.domain.echelle_lc import (
    Topologie,
    elements_de_butterworth,
    transfert_de_l_echelle,
)
from polytech.electronique.domain.methodes_sallen_key import (
    Methode,
    cellule_normalisee,
)
from polytech.electronique.domain.sallen_key_a_gain import SallenKeyPasseBas
from polytech.electronique.domain.sensibilite_sallen_key import (
    COMPOSANTS,
    sensibilites_de_la_qualite,
)

Q = sp.Symbol("Q", positive=True)
S = sp.Symbol("s")


def test_la_methode_a_sensibilite_minimale_ne_depend_pas_de_q() -> None:
    """Planche 27 : `Q` ne bouge que de la moitie de l'erreur des capacites."""
    cellule = cellule_normalisee(Methode.SENSIBILITE_MINIMALE, Q, 1)
    demi = sp.Rational(1, 2)
    assert sensibilites_de_la_qualite(cellule) == {
        "r1": 0,
        "r2": 0,
        "c1": demi,
        "c2": -demi,
        "gain": 2 * Q**2,
    }


def test_les_sensibilites_des_trois_autres_methodes_croissent_avec_q() -> None:
    """Planches 24 a 26 : en `2 Q`, en `Q^2`, en `Q / racine(3)`."""
    demi = sp.Rational(1, 2)
    egaux = sensibilites_de_la_qualite(cellule_normalisee(Methode.ELEMENTS_EGAUX, Q, 1))
    assert (egaux["r1"], egaux["c1"], egaux["gain"]) == (
        Q - demi,
        2 * Q - demi,
        3 * Q - 1,
    )
    capacites = sensibilites_de_la_qualite(
        cellule_normalisee(Methode.CAPACITES_EGALES, Q, 1)
    )
    assert (capacites["r1"], capacites["c1"]) == (demi, Q**2 + demi)
    moderee = sensibilites_de_la_qualite(
        cellule_normalisee(Methode.SENSIBILITE_MODEREE, Q, 1)
    )
    assert sp.simplify(moderee["c1"] - (Q / sp.sqrt(3) + demi)) == 0
    assert sp.simplify(moderee["gain"] - 4 * Q / sp.sqrt(3)) == 0


@pytest.mark.parametrize("methode", list(Methode))
def test_l_argument_d_echelle(methode: Methode) -> None:
    """`S_R1 + S_R2 = 0` et `S_C1 + S_C2 = 0`, pour toute methode."""
    valeurs = sensibilites_de_la_qualite(cellule_normalisee(methode, Q, 3))
    assert sp.simplify(valeurs["r1"] + valeurs["r2"]) == 0
    assert sp.simplify(valeurs["c1"] + valeurs["c2"]) == 0


@pytest.mark.parametrize("methode", list(Methode))
@pytest.mark.parametrize("rang", range(5))
def test_la_sensibilite_par_une_difference_finie(methode: Methode, rang: int) -> None:
    """Un composant deplace de un millionieme : `Q` suit de `S` millioniemes."""
    cellule = cellule_normalisee(methode, 2, 5)
    nom = COMPOSANTS[rang]
    valeurs = [float(getattr(cellule, n)) for n in COMPOSANTS]
    pas = 1e-6
    plus, moins = list(valeurs), list(valeurs)
    plus[rang] *= 1 + pas
    moins[rang] *= 1 - pas
    haut = float(SallenKeyPasseBas(*plus).qualite())
    bas = float(SallenKeyPasseBas(*moins).qualite())
    mesuree = (haut - bas) / (2 * pas * 2.0)
    attendue = float(sensibilites_de_la_qualite(cellule)[nom])
    assert mesuree == pytest.approx(attendue, rel=1e-6, abs=1e-7)


def test_l_echelle_de_branches_redonne_l_echelle_du_prototype() -> None:
    """Sans transformation, les branches sont l'echelle de `echelle_lc`."""
    elements = elements_de_butterworth(3)
    en_pi = (
        Branche(capacite=elements[0]),
        Branche(inductance=elements[1]),
        Branche(capacite=elements[2]),
    )
    par_branches = EchelleTerminee(
        en_pi, serie_d_abord=False, source=1.0, charge=1.0
    ).transfert(S)
    assert sp.simplify(par_branches - transfert_de_l_echelle(elements, S)) == 0
    en_t = (
        Branche(inductance=elements[0]),
        Branche(capacite=elements[1]),
        Branche(inductance=elements[2]),
    )
    reference = transfert_de_l_echelle(
        elements, S, topologie=Topologie.INDUCTANCE_D_ABORD
    )
    assert (
        sp.simplify(
            EchelleTerminee(en_t, serie_d_abord=True, source=1.0, charge=1.0).transfert(
                S
            )
            - reference
        )
        == 0
    )


def test_un_passe_bande_tient_son_ondulation_aux_bords_de_sa_bande() -> None:
    """Chebyshev 0,1 dB d'ordre 3 : -0,1 dB aux deux bords, 0 dB au centre."""
    g = elements_de_chebyshev(0.1, 3)
    centre, largeur = 10.0, 2.0
    prototype = (
        Branche(inductance=g[0]),
        Branche(capacite=g[1]),
        Branche(inductance=g[2]),
    )
    bande = tuple(en_passe_bande(b, centre, largeur) for b in prototype)
    filtre = EchelleTerminee(bande, serie_d_abord=True, source=1.0, charge=1.0)
    racine = math.sqrt(largeur**2 / 4 + centre**2)
    for bord in (racine - largeur / 2, racine + largeur / 2):
        # 1e-4 : la table du cours arrondit `40 / ln 10` a 17,37.
        assert 20 * math.log10(2 * filtre.module(bord)) == pytest.approx(-0.1, abs=1e-4)
    assert filtre.module(centre) == pytest.approx(0.5, rel=1e-9)


def test_l_echelle_de_branches_refuse_ce_qui_n_est_pas_un_filtre() -> None:
    """Ni echelle vide, ni terminaison negative ou nulle."""
    with pytest.raises(ValueError, match="echelle vide"):
        EchelleTerminee((), serie_d_abord=True, source=1.0, charge=1.0)
    branche = (Branche(inductance=1.0),)
    for source, charge in ((-1.0, 1.0), (1.0, 0.0)):
        with pytest.raises(ValueError, match="terminaisons invalides"):
            EchelleTerminee(branche, serie_d_abord=True, source=source, charge=charge)
    sans_source = EchelleTerminee(branche, serie_d_abord=True, source=0.0, charge=2.0)
    assert sp.simplify(sans_source.transfert(S) - 2 / (S + 2)) == 0
    assert sans_source.module(2.0) == pytest.approx(1 / math.sqrt(2))
    en_derivation = EchelleTerminee(
        branche, serie_d_abord=False, source=1.0, charge=1.0
    )
    assert sp.simplify(en_derivation.transfert(S) - S / (2 * S + 1)) == 0
