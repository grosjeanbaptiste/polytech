"""Les exercices 15 et 16 : les forces d'un entrefer, tirees de l'energie.

Les deux exercices portent la meme physique et **deux conventions
opposees**, et c'est leur mise en regard qui les rend instructifs.

L'exercice 15 travaille a **flux constant** -- l'enonce fixe `B_H`, donc
le flux -- et la force vaut `-dW/dl`. L'exercice 16 travaille a **courant
constant**, et elle vaut `+dW/da`. Les deux donnent une force qui referme
l'entrefer, ce qui est physiquement obligatoire : un circuit magnetique
tend toujours a se refermer. Les signes opposes des deux formules ne
decrivent donc pas des forces opposees -- c'est ce que le chapitre
enseigne, et ce que les deux exercices ensemble demontrent.

Le fait le plus frappant de l'exercice 15 n'est pas dans le corrige
officiel : les **deux entrefers de 0,1 mm consomment 78 % des
ampere-tours**, contre 22 % pour le metre de fer. C'est toute la raison
d'etre d'un circuit magnetique ferme, et ce test le chiffre.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.magnetostatique.entrefer import (
    CircuitAEntrefers,
    champ_dans_l_entrefer,
    force_a_flux_constant_sur_un_entrefer,
    force_d_un_electroaimant,
)

# Donnees de l'exercice 15.
INDUCTION = 1.6
SECTION = 4e-4
CHAMP_DANS_LE_FER = 70.0
SPIRES = 500
LONGUEUR_DU_FER = 2 * 50e-2
EPAISSEUR_D_ENTREFER = 0.1e-3
NOMBRE_D_ENTREFERS = 2

CIRCUIT = CircuitAEntrefers(
    champ_dans_le_fer=CHAMP_DANS_LE_FER,
    longueur_du_fer=LONGUEUR_DU_FER,
    induction=INDUCTION,
    epaisseur=EPAISSEUR_D_ENTREFER,
    entrefers=NOMBRE_D_ENTREFERS,
)


def test_le_champ_dans_l_entrefer_est_enorme_devant_celui_du_fer() -> None:
    """`H_e = B/mu_0 = 1,27 . 10^6` A/m contre 70 dans le fer.

    Un facteur dix-huit mille. C'est la permeabilite relative du
    materiau qui le produit : a induction egale, l'air demande `mu_r`
    fois plus de champ que le fer.
    """
    champ = champ_dans_l_entrefer(INDUCTION)
    assert champ == pytest.approx(1273239.5, abs=1.0)
    assert champ / CHAMP_DANS_LE_FER > 15000


def test_le_courant_d_excitation_vaut_deux_tiers_d_ampere() -> None:
    """`I = (H_H l_H + 2 H_e l_e) / N = 0,65` A."""
    courant = CIRCUIT.courant_d_excitation(SPIRES)
    assert courant == pytest.approx(0.65, abs=0.005)


def test_les_deux_entrefers_consomment_les_quatre_cinquiemes_des_amperes_tours() -> (
    None
):
    """Deux dixiemes de millimetre d'air contre un metre de fer : 78 % / 22 %.

    C'est le fait le plus frappant de l'exercice, et le corrige officiel
    ne le releve pas. Il explique pourquoi un circuit magnetique se
    ferme : le moindre entrefer coute plus que toute la culasse.
    """
    part = CIRCUIT.part_des_entrefers()
    assert part == pytest.approx(0.784, abs=0.002)


def test_doubler_l_entrefer_renforce_encore_sa_domination() -> None:
    """Et la domination croit : a 0,2 mm, les entrefers passent a 88 %.

    La part du fer est fixe -- elle ne depend que de sa longueur et de
    son champ -- tandis que celle de l'air croit avec l'epaisseur. C'est
    pourquoi un entrefer mal maitrise fait deriver tout le
    dimensionnement.
    """
    double = CircuitAEntrefers(
        champ_dans_le_fer=CHAMP_DANS_LE_FER,
        longueur_du_fer=LONGUEUR_DU_FER,
        induction=INDUCTION,
        epaisseur=2 * EPAISSEUR_D_ENTREFER,
        entrefers=NOMBRE_D_ENTREFERS,
    ).part_des_entrefers()
    assert double == pytest.approx(0.879, abs=0.002)


def test_la_force_d_arrachement_vaut_huit_cent_quinze_newtons() -> None:
    """`F = H_e Phi = 815` N, soit quatre-vingt-trois kilogrammes.

    La grandeur surprend pour un courant de deux tiers d'ampere, et
    c'est tout l'interet de l'electroaimant.
    """
    force = force_a_flux_constant_sur_un_entrefer(INDUCTION, SECTION)
    assert force == pytest.approx(815.0, abs=1.0)
    assert force / 9.81 == pytest.approx(83.0, abs=1.0)


def test_la_force_croit_comme_le_carre_de_l_induction() -> None:
    """`F = B^2 S / mu_0` : doubler `B` quadruple la force.

    La forme `H_e Phi` du manuscrit le cache ; la reecrire en `B^2 S/mu_0`
    le montre, et c'est la forme que tout dimensionnement emploie.
    """
    simple = force_a_flux_constant_sur_un_entrefer(INDUCTION, SECTION)
    double = force_a_flux_constant_sur_un_entrefer(2 * INDUCTION, SECTION)
    assert double == pytest.approx(4 * simple)
    attendue = INDUCTION**2 * SECTION / (4 * math.pi * 1e-7)
    assert simple == pytest.approx(attendue)


def test_la_force_de_l_electroaimant_decroit_comme_le_carre_de_l_entrefer() -> None:
    """`F = -N^2 mu_0 I^2 S / (2 a^2)`, symbolique et verifiee par sympy.

    La decroissance en `1/a^2` est ce qui rend un electroaimant si
    puissant au contact et si faible a un millimetre -- et ce qui le
    rend difficile a asservir.
    """
    spires, courant, section, entrefer = sp.symbols(
        "N_1 I S alpha",
        positive=True,
    )
    force = force_d_un_electroaimant(spires, courant, section, entrefer)
    permeabilite = sp.Symbol("mu_0", positive=True)
    attendue = -(spires**2) * permeabilite * courant**2 * section / (2 * entrefer**2)
    assert sp.simplify(force - attendue) == 0


def test_la_force_de_l_electroaimant_est_negative_donc_refermante() -> None:
    """Le signe du manuscrit, et ce qu'il signifie.

    A courant constant la force vaut `+dW/da` ; l'energie decroit quand
    l'entrefer croit, donc la derivee est negative, donc la force tend a
    **refermer** l'entrefer. Un circuit magnetique se referme toujours,
    et c'est pourquoi les deux exercices -- l'un a flux constant, l'autre
    a courant constant -- aboutissent au meme sens malgre des formules de
    signes opposes.
    """
    spires, courant, section, entrefer = sp.symbols(
        "N_1 I S alpha",
        positive=True,
    )
    force = force_d_un_electroaimant(spires, courant, section, entrefer)
    assert sp.simplify(force).could_extract_minus_sign()


def test_une_section_nulle_est_refusee() -> None:
    """Sans section, il n'y a pas de flux, donc pas de force."""
    with pytest.raises(ValueError, match="section nulle ou negative"):
        force_a_flux_constant_sur_un_entrefer(INDUCTION, 0.0)


def test_un_nombre_d_entrefers_nul_est_refuse() -> None:
    """Un circuit sans entrefer n'est pas le cas de ces exercices.

    Et le refus porte un sens : la part des entrefers y serait nulle, ce
    qui est vrai mais rend la question sans objet.
    """
    with pytest.raises(ValueError, match="nombre d'entrefers nul"):
        CircuitAEntrefers(
            champ_dans_le_fer=CHAMP_DANS_LE_FER,
            longueur_du_fer=LONGUEUR_DU_FER,
            induction=INDUCTION,
            epaisseur=EPAISSEUR_D_ENTREFER,
            entrefers=0,
        )
