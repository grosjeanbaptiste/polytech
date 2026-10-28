"""Les constantes du corrige de l'activite 2 viennent de la bibliotheque.

Trois valeurs etaient tapees dans le `.tex` : la masse volumique de l'eau
de mer, la pression atmospherique et la constante des gaz. Chacune sort
d'une macro que `activite_2_bateau` calcule depuis la bibliotheque, et le
document ne porte plus aucun de ces litteraux.
"""

from __future__ import annotations

import re
from pathlib import Path

from polytech.chemistry.domain.gaz import CONSTANTE_DES_GAZ
from polytech.fluides.application import activite_2 as a2
from polytech.fluides.application import activite_2_hypotheses as hyp
from polytech.fluides.application.macros_activite_2_complements import (
    macros_complements,
)
from polytech.fluides.application.registre import macros_du_travail
from polytech.maths.domain.comparaison import egales_arrondies

TRAVAIL = (
    Path(__file__).resolve().parent.parent
    / "travaux"
    / "MecaniqueFluides"
    / "activite-integration-2"
)


def test_la_masse_volumique_de_l_eau_de_mer_est_celle_de_l_exercice() -> None:
    assert float(macros_complements()["MasseVolumiqueEauDeMer"]) == a2.EAU_DE_MER


def test_la_pression_atmospherique_est_celle_de_l_atmosphere_normale() -> None:
    assert float(macros_complements()["PressionAtmospherique"]) == hyp.AIR_PRESSION


def test_la_constante_des_gaz_est_celle_de_la_chimie_arrondie_a_trois_decimales() -> (
    None
):
    macro = float(macros_complements()["ConstanteDesGaz"])
    assert egales_arrondies(macro, CONSTANTE_DES_GAZ, 3)
    assert macro == round(macro, 3)


def test_le_registre_publie_les_trois_constantes() -> None:
    jeu = macros_du_travail("activite-integration-2")
    for cle in ("MasseVolumiqueEauDeMer", "PressionAtmospherique", "ConstanteDesGaz"):
        assert cle in jeu


def test_le_document_ne_tape_plus_ces_constantes() -> None:
    texte = (TRAVAIL / "activite_integration_2.tex").read_text(encoding="utf-8")
    for litteral in (r"\b1030\b", r"\b101325\b", r"8\.314"):
        assert re.search(litteral, texte) is None, litteral
