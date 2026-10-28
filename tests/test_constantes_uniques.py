"""Une constante physique ne s'ecrit qu'une fois dans la bibliotheque.

`AGENTS.md` nommait le defaut : la charge elementaire etait definie trois
fois, deux en electronique et une en electromagnetisme. Elle n'etait pas
seule -- la vitesse de la lumiere, la constante de Planck, la
permeabilite du vide, la pesanteur et le zero Celsius l'etaient aussi.

Chaque constante vit dans la discipline qui l'enseigne, et les autres
contextes l'importent : c'est ce qui dit que deux cours parlent de la
meme grandeur. Deux gardes : les noms conserves designent le meme objet,
et aucun litteral ne reparait ailleurs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import pytest

from polytech.electromagnetisme.domain import constantes
from polytech.electronique.application import enonce_tp_01
from polytech.electronique.domain import bruit, des_champs_aux_circuits
from polytech.fluides.application import macros_hypotheses
from polytech.fluides.domain import grandeurs as grandeurs_des_fluides
from polytech.mecanique.domain import pesanteur
from polytech.thermique.application import enonce_tp_02, enonce_tp_03
from polytech.thermique.domain import grandeurs as grandeurs_thermiques
from polytech.thermique.domain.rayonnement import corps_noir
from polytech.thermique.domain.tables import eau_saturee

BIBLIOTHEQUE: Final = Path(__file__).resolve().parent.parent / "polytech"

DEMEURES: Final = {
    "1.602176634e-19": "electromagnetisme/domain/constantes.py",
    "2.99792458e8": "electromagnetisme/domain/constantes.py",
    "8.8541878128e-12": "electromagnetisme/domain/constantes.py",
    "1.25663706212e-6": "electromagnetisme/domain/constantes.py",
    "6.62607015e-34": "thermique/domain/rayonnement/corps_noir.py",
    "1.380649e-23": "thermique/domain/rayonnement/corps_noir.py",
    "5.670374419e-8": "thermique/domain/rayonnement/corps_noir.py",
    "9.81": "mecanique/domain/pesanteur.py",
    "273.15": "thermique/domain/grandeurs.py",
    "6.02214076e23": "chemistry/domain/matiere.py",
    "8.31446261815324": "chemistry/domain/gaz.py",
    "6.6743e-11": "mecanique/domain/gravitation.py",
    "5.29177210903e-11": "chemistry/domain/atomes.py",
}
"""Le litteral, et le seul fichier qui a le droit de l'affecter."""


def test_la_charge_de_l_electron_est_la_charge_elementaire() -> None:
    charge = des_champs_aux_circuits.CHARGE_DE_L_ELECTRON
    assert charge is constantes.CHARGE_ELEMENTAIRE


def test_les_constantes_du_vide_sont_celles_de_l_electromagnetisme() -> None:
    assert enonce_tp_01.MU0 is constantes.PERMEABILITE_DU_VIDE
    assert corps_noir.LUMIERE is constantes.VITESSE_DE_LA_LUMIERE


def test_la_constante_de_planck_est_celle_du_corps_noir() -> None:
    assert bruit.CONSTANTE_DE_PLANCK is corps_noir.PLANCK


def test_le_zero_celsius_est_celui_de_la_thermique() -> None:
    reference = grandeurs_thermiques.ZERO_ABSOLU
    assert eau_saturee.ZERO_CELSIUS is reference
    assert enonce_tp_02.KELVIN is reference
    assert enonce_tp_03.KELVIN is reference
    assert macros_hypotheses.TEMPERATURE_MINIMALE is reference


def test_la_pesanteur_n_a_qu_une_valeur() -> None:
    """Les modules qui l'emploient l'importent : seul le litteral est garde."""
    assert pytest.approx(9.81) == grandeurs_des_fluides.PESANTEUR
    assert grandeurs_des_fluides.PESANTEUR is pesanteur.PESANTEUR


AFFECTATION: Final = re.compile(
    r"^\s*[A-Za-z_][A-Za-z_0-9]*(: [^=]+)? = (?P<valeur>\S+)$"
)


@pytest.mark.parametrize("litteral", sorted(DEMEURES))
def test_aucun_litteral_ne_reparait_ailleurs(litteral: str) -> None:
    """Le nombre n'est affecte que dans sa demeure."""
    fautifs = [
        str(fichier.relative_to(BIBLIOTHEQUE))
        for fichier in sorted(BIBLIOTHEQUE.rglob("*.py"))
        for ligne in fichier.read_text(encoding="utf-8").splitlines()
        if (trouve := AFFECTATION.match(ligne)) and trouve["valeur"] == litteral
    ]
    assert fautifs == [DEMEURES[litteral]]


def test_la_permeabilite_ne_se_recalcule_nulle_part() -> None:
    """`4 pi 1e-7` ecrit en clair est une seconde definition de `mu_0`."""
    motif = re.compile(r"4(\.0)? \* (math\.)?pi \* 1e-7")
    fautifs = [
        str(fichier.relative_to(BIBLIOTHEQUE))
        for fichier in sorted(BIBLIOTHEQUE.rglob("*.py"))
        if motif.search(fichier.read_text(encoding="utf-8"))
    ]
    assert fautifs == []
