"""Le chapitre I du cours de fluides : les proprietes d'un fluide.

Ce chapitre n'etait **pas declare du tout** -- ni couvert, ni a faire, ni
silence motive. Exactement la situation de `TH-01` en thermique : un
fascicule intitule « Introduction » qu'on avait pris pour une
presentation sans l'ouvrir. Il enseigne la viscosite, la compressibilite,
la dilatation isobare, la tension superficielle et les classifications
d'ecoulement.

La verification la plus satisfaisante vient de la planche elle-meme, qui
donne l'air a 20 degres sous deux pressions :

* a 1 atm : `mu = 1,83 . 10^-5 kg/(m s)` et `nu = 1,52 . 10^-5 m^2/s` ;
* a 4 atm : `mu = 1,83 . 10^-5` -- le **meme** -- et `nu = 0,380 . 10^-5`.

La viscosite dynamique d'un gaz ne depend donc pas de la pression, et la
cinematique si : elle est divisee par quatre quand la pression est
quadruplee, parce que la masse volumique l'est. Le rapport `1,52/0,380`
vaut exactement 4, ce qui confirme `nu = mu / rho` sur les chiffres du
cours plutot que sur sa formule.
"""

from __future__ import annotations

import pytest

from polytech.fluides.domain.proprietes import (
    coefficient_de_compressibilite_du_gaz_parfait,
    coefficient_de_dilatation_du_gaz_parfait,
    compressibilite_isotherme,
    energie_de_surface,
    viscosite_cinematique,
)

# Chiffres de la planche I.2.1, air a 20 degres Celsius.
VISCOSITE_DYNAMIQUE_DE_L_AIR = 1.83e-5
CINEMATIQUE_A_UNE_ATMOSPHERE = 1.52e-5
CINEMATIQUE_A_QUATRE_ATMOSPHERES = 0.380e-5


def test_la_viscosite_cinematique_de_l_air_concorde_avec_la_planche() -> None:
    """`nu = mu / rho` doit redonner les deux chiffres du cours.

    La masse volumique de l'air a 20 degres et 1 atm vaut 1,204 kg/m^3 ;
    a 4 atm elle quadruple, le gaz etant parfait a temperature constante.
    Les deux viscosites cinematiques de la planche doivent donc sortir du
    meme `mu`, et c'est le cas a un pour cent pres.
    """
    une_atmosphere = viscosite_cinematique(VISCOSITE_DYNAMIQUE_DE_L_AIR, 1.204)
    quatre = viscosite_cinematique(VISCOSITE_DYNAMIQUE_DE_L_AIR, 4 * 1.204)
    assert une_atmosphere == pytest.approx(CINEMATIQUE_A_UNE_ATMOSPHERE, rel=0.02)
    assert quatre == pytest.approx(CINEMATIQUE_A_QUATRE_ATMOSPHERES, rel=0.02)


def test_le_rapport_des_deux_viscosites_cinematiques_vaut_quatre() -> None:
    """La verification interne a la planche, et elle est exacte.

    `1,52 / 0,380 = 4` : le rapport des pressions. Cela dit deux choses
    d'un coup -- que `mu` d'un gaz ne depend pas de la pression, et que
    `nu` en depend par la masse volumique. Confondre les deux viscosites
    est l'erreur que ce chapitre previent.
    """
    rapport = CINEMATIQUE_A_UNE_ATMOSPHERE / CINEMATIQUE_A_QUATRE_ATMOSPHERES
    assert rapport == pytest.approx(4.0, rel=0.001)


def test_la_compressibilite_d_un_gaz_parfait_vaut_sa_pression() -> None:
    """« kappa_gp = P », ecrit la planche I.2.2.

    Le resultat tombe de `P = rho R T` : a temperature constante
    `(dP/drho)_T = RT = P/rho`, et `kappa = rho (dP/drho)_T` vaut donc
    `P`. C'est pourquoi la compressibilite d'un gaz se mesure en pascals,
    ce qui surprend d'abord.
    """
    assert coefficient_de_compressibilite_du_gaz_parfait(2.5e5) == pytest.approx(2.5e5)


def test_la_compressibilite_isotherme_est_l_inverse_du_coefficient() -> None:
    """`alpha = 1 / kappa`, en pascals a la puissance moins un.

    Les deux grandeurs disent la meme chose en sens inverse, et leurs
    unites inverses sont ce qui empeche de les confondre : un gaz
    fortement compressible a un `kappa` petit et un `alpha` grand.
    """
    assert compressibilite_isotherme(2.0e5) == pytest.approx(5e-6)


def test_la_dilatation_isobare_d_un_gaz_parfait_est_l_inverse_de_sa_temperature() -> (
    None
):
    """« beta_gp = 1/T », planche I.2.3, en kelvins a la puissance moins un.

    A temperature ambiante elle vaut donc environ `1/293`, soit 0,34 % par
    kelvin -- l'ordre de grandeur qui gouverne la convection naturelle,
    ou `beta` apparait dans le nombre de Grashof.
    """
    beta = coefficient_de_dilatation_du_gaz_parfait(293.0)
    assert beta == pytest.approx(1 / 293.0)
    assert beta == pytest.approx(0.00341, rel=0.01)


def test_l_energie_de_surface_est_le_travail_d_expansion() -> None:
    """La planche l'ecrit : `N.m/m^2 -> J/m^2`, numeriquement egal a `N/m`.

    La tension superficielle est donc **deux** grandeurs d'un seul
    nombre : une force par unite de longueur, et une energie par unite
    d'aire. L'identite des deux n'est pas une coincidence mais une
    equivalence dimensionnelle, et c'est ce qui autorise a parler du
    « travail d'expansion qu'il faut faire pour augmenter l'aire de la
    surface liquide d'une unite ».
    """
    tension = 0.0728
    assert energie_de_surface(tension, aire=1.0) == pytest.approx(tension)
    assert energie_de_surface(tension, aire=2.5) == pytest.approx(2.5 * tension)


def test_une_masse_volumique_nulle_est_refusee() -> None:
    """La viscosite cinematique y diverge."""
    with pytest.raises(ValueError, match="masse volumique nulle ou negative"):
        viscosite_cinematique(VISCOSITE_DYNAMIQUE_DE_L_AIR, 0.0)


def test_une_pression_nulle_est_refusee_pour_la_compressibilite() -> None:
    """Un gaz a pression nulle n'a pas de compressibilite definie."""
    with pytest.raises(ValueError, match="pression nulle ou negative"):
        compressibilite_isotherme(0.0)


def test_une_temperature_absolue_nulle_est_refusee() -> None:
    """`beta = 1/T` y diverge, et le zero absolu n'est pas un cas du cours."""
    with pytest.raises(ValueError, match="temperature absolue nulle ou negative"):
        coefficient_de_dilatation_du_gaz_parfait(0.0)


def test_une_tension_superficielle_negative_est_refusee() -> None:
    """Une surface qui gagnerait de l'energie en s'etendant exploserait."""
    with pytest.raises(ValueError, match="tension superficielle negative"):
        energie_de_surface(-0.01, aire=1.0)
