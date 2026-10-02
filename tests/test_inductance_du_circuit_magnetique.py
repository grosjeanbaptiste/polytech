"""L'exercice 6 : l'inductance d'un circuit magnetique a deux branches.

Deux idees que les exercices precedents n'avaient pas.

**Les ampere-tours sont partages.** Le theoreme d'Ampere sur n'importe
quel contour entourant l'enroulement donne la meme chose, `N I`. Les deux
branches du circuit -- celle qui porte l'entrefer et celle qui ne le porte
pas -- consomment donc le **meme** nombre d'ampere-tours, et c'est ce qui
permet de deduire le champ de la seconde sans rien savoir d'autre.

**Le flux total est la somme des flux de branche.** C'est lui qui entre
dans `L = N Phi / I`, et non le flux d'une seule branche -- l'erreur qui
donnerait une inductance seize fois trop petite ici.

Un resultat de l'enonce vaut d'etre releve : `H_T = 64` A/m est
**exact**, et non arrondi. Le rapport des longueurs valant `16/17,5 =
32/35`, on a `70 x 32/35 = 64` tout rond. Le test le verifie, parce qu'un
corrige qui annoncerait « environ 64 » laisserait croire a un arrondi la
ou il y a une egalite.
"""

from __future__ import annotations

import math

import pytest

from polytech.electromagnetisme.domain.magnetostatique.circuits_magnetiques import (
    BrancheFerromagnetique,
    champ_pour_les_amperes_tours,
    inductance_depuis_les_flux,
)

# Donnees de la transcription.
SPIRES = 1000
RAYON_H, RAYON_T = 0.16, 0.175
BRANCHE_H = BrancheFerromagnetique(
    induction=1.6,
    champ_lu=70.0,
    section=2e-2 * 4e-2,
    longueur=2 * math.pi * RAYON_H,
)
BRANCHE_T = BrancheFerromagnetique(
    induction=0.5,
    champ_lu=64.0,
    section=1e-2 * 4e-2,
    longueur=2 * math.pi * RAYON_T,
)


def test_le_flux_de_la_branche_h_vaut_mille_deux_cent_quatre_vingts_microwebers() -> (
    None
):
    """`Phi_H = B_H S_H = 1,6 x 8 cm^2 = 0,00128` Wb."""
    assert BRANCHE_H.flux() == pytest.approx(0.00128)


def test_le_courant_se_lit_sur_la_branche_a_induction_imposee() -> None:
    """`I = H_H l_H / N = 0,07037` A.

    C'est la branche dont l'enonce donne `B`, donc celle qui fixe le
    courant. L'ordre des calculs n'est pas libre : partir de l'autre
    branche laisserait deux inconnues.
    """
    assert BRANCHE_H.courant_pour(SPIRES) == pytest.approx(0.07037, abs=1e-5)


def test_le_champ_de_la_seconde_branche_est_exactement_soixante_quatre() -> None:
    """`H_T = N I / l_T`, et le resultat est **exact**, non arrondi.

    Les ampere-tours etant partages, `H_T l_T = H_H l_H`, donc
    `H_T = H_H l_H / l_T = 70 x 16/17,5 = 70 x 32/35 = 64` tout rond. Les
    `2 pi` se simplifient, et il ne reste qu'un rapport de rayons.
    """
    amperes_tours = BRANCHE_H.amperes_tours()
    champ = champ_pour_les_amperes_tours(amperes_tours, BRANCHE_T.longueur)
    assert champ == pytest.approx(64.0, abs=1e-9)
    assert champ == pytest.approx(70.0 * RAYON_H / RAYON_T, abs=1e-9)


def test_les_deux_branches_consomment_les_memes_amperes_tours() -> None:
    """La propriete qui porte tout l'exercice, et elle se verifie.

    Le theoreme d'Ampere sur deux contours entourant le meme enroulement
    donne la meme force magnetomotrice. Les deux produits `H l` doivent
    donc concorder -- et c'est le controle qui attrape une longueur mal
    lue sur la figure.
    """
    assert BRANCHE_H.amperes_tours() == pytest.approx(
        BRANCHE_T.amperes_tours(),
        rel=1e-6,
    )


def test_l_inductance_vaut_vingt_et_un_henrys() -> None:
    """`L = N (Phi_T + Phi_H) / I = 21,03` H.

    Le flux total, non celui d'une branche : avec le seul flux de `T` on
    obtiendrait `2,8` H, soit sept fois moins.
    """
    courant = BRANCHE_H.courant_pour(SPIRES)
    inductance = inductance_depuis_les_flux(
        SPIRES,
        (BRANCHE_T.flux(), BRANCHE_H.flux()),
        courant,
    )
    assert inductance == pytest.approx(21.03, abs=0.01)


def test_oublier_une_branche_fait_perdre_un_facteur_sept() -> None:
    """L'ordre de grandeur de l'erreur, pour qu'elle ne passe pas.

    Le flux de la branche `H` est plus de six fois celui de `T` -- sa
    section est double et son induction plus de trois fois plus grande.
    Ne garder que `T` donnerait une inductance sept fois trop petite,
    donc visiblement fausse ; ne garder que `H` la sous-estimerait de
    quinze pour cent, ce qui est bien plus insidieux.
    """
    courant = BRANCHE_H.courant_pour(SPIRES)
    complete = inductance_depuis_les_flux(
        SPIRES,
        (BRANCHE_T.flux(), BRANCHE_H.flux()),
        courant,
    )
    sans_h = inductance_depuis_les_flux(SPIRES, (BRANCHE_T.flux(),), courant)
    sans_t = inductance_depuis_les_flux(SPIRES, (BRANCHE_H.flux(),), courant)
    assert complete / sans_h == pytest.approx(7.4, abs=0.1)
    assert complete / sans_t == pytest.approx(1.16, abs=0.01)


def test_une_inductance_a_courant_nul_est_refusee() -> None:
    """`L = N Phi / I` y diverge, et un circuit sans courant n'a pas de flux."""
    with pytest.raises(ValueError, match="courant nul ou negatif"):
        inductance_depuis_les_flux(SPIRES, (0.001,), 0.0)


def test_une_inductance_sans_flux_est_refusee() -> None:
    """Un circuit sans aucune branche ne porte pas d'inductance."""
    with pytest.raises(ValueError, match="aucun flux de branche"):
        inductance_depuis_les_flux(SPIRES, (), 1.0)


def test_une_longueur_nulle_est_refusee_pour_le_champ() -> None:
    """Le champ y diverge : un contour de longueur nulle n'entoure rien."""
    with pytest.raises(ValueError, match="longueur de contour nulle"):
        champ_pour_les_amperes_tours(70.0, 0.0)
