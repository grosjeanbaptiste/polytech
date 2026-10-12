"""Le chapitre VII du recueil d'algebre, « Les espaces vectoriels », refait.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, exercices 1 a 44 des
sections VII.6 a VII.10 ; l'oracle est « Solutions du 7eme chapitre » (pages
94 a 96 du syllabus). Une base ne s'ecrit pas d'une seule facon : celle du
syllabus est *eprouvee* -- elle est libre et engendre le meme sous-espace
que celle que la bibliotheque trouve.

Les questions « est-ce un sous-espace ? » (1, 2, 24, 30, 41) et les
demonstrations (11, 20, 21, 28, 39, 40, 42, 43) sont dans
`tests/test_recueil_algebre_chapitre_07_raisonnements.py`.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.application import recueil_algebre_ch07_bases as bases
from polytech.maths.application import recueil_algebre_ch07_enonces as enonces
from polytech.maths.application import recueil_algebre_ch07_familles as familles
from polytech.maths.domain.algebre.familles_de_vecteurs import (
    appartient,
    est_libre,
    est_une_base,
)
from polytech.maths.domain.algebre.sous_espaces import engendrent_le_meme_sous_espace

R = sp.Rational
I = sp.I  # noqa: E741  # l'unite imaginaire, que le syllabus note i


def test_exercices_3_4_23_31_libre_ou_liee() -> None:
    libres = familles.libertes()
    assert libres == {"3a": True, "3b": True, "4": False, "23": False, "31": False}


def test_exercice_5_la_condition_sur_a() -> None:
    assert sp.expand(familles.condition_du_5() - (enonces.PARAMETRE - 1)) == 0


@pytest.mark.parametrize(
    ("cle", "attendues"),
    [
        ("6", (R(1, 3), R(1, 2))),
        ("9", (R(1, 4), R(1, 4), R(1, 2), R(1, 2))),
        ("13a", (1, 0, 1)),
        ("13b", (R(1, 3), -R(1, 3), -R(2, 3))),
        ("34", (-1, 1, 2)),
    ],
)
def test_les_coordonnees_demandees(cle: str, attendues: tuple) -> None:
    assert familles.coordonnees_demandees()[cle] == attendues


def test_exercices_7_et_8() -> None:
    assert familles.exercice_7() is False
    assert familles.exercice_8() == 2


def test_exercices_9_12_13_34_ce_sont_des_bases() -> None:
    assert familles.sont_des_bases() == dict.fromkeys(
        ("9", "12", "13a", "13b", "34"), True
    )


NATURES = {
    "10F1": (False, False),
    "10F2": (True, True),
    "10F3": (True, False),
    "10F4": (False, True),
    "10F5": (False, False),
    "32F1": (False, False),
    "32F2": (False, True),
    "32F3": (False, False),
    "32F4": (True, False),
}
"""Cle : libre, generatrice -- les reponses a) et b) du syllabus."""


@pytest.mark.parametrize("cle", sorted(NATURES))
def test_exercices_10_et_32_libre_generatrice(cle: str) -> None:
    nature = familles.natures()[cle]
    assert (nature.libre, nature.generatrice) == NATURES[cle]
    assert nature.base == all(NATURES[cle])


def test_exercice_10_les_bases_completees_ou_extraites() -> None:
    natures = familles.natures()
    assert est_une_base(natures["10F3"].completee)
    assert est_une_base(((1, 2, 1), (1, 0, 1), (0, 0, 1)))
    assert natures["10F4"].extraite == ((1, 1, 1), (2, 1, 0), (2, 0, 1))


def test_exercice_32_les_bases_completees_ou_extraites() -> None:
    natures = familles.natures()
    assert natures["32F2"].extraite == ((3, 2, 1), (5, 3, 4), (1, 1, 1))
    assert est_une_base(natures["32F4"].completee)
    assert est_une_base(((1, 1, 0), (-1, 0, 1), (1, 1, 1)))


ORACLES = {
    "14": ((1, -1, 2, 1), (2, 0, -1, 0)),
    "15": ((1, 5, 2, -1), (0, -3, 0, 1)),
    "18A": ((2, 1, 0), (0, 0, 1)),
    "18B": ((1, 2, -3),),
    "19": ((-3, 7, -5, 1),),
    "25": ((1, 1, 1, 1), (1, 0, 0, 2)),
    "26": ((I, 0, 2, I), (-1 + I, -1, 5 * I, 0)),
    "29": ((0, 1, -1, 0), (1, 2, 3, 4)),
    "35": ((1, 1, 1, 1), (1, 0, 0, 2)),
    "36W1": ((1, 0, 0, 0), (0, 1, 0, -1), (0, 0, 1, -1)),
    "36W2": ((1, 0, 0, -R(1, 2)), (0, 1, 0, -1), (0, 0, 1, -1)),
    "36Inter": ((0, 1, 0, -1), (0, 0, 1, -1)),
    "37": ((1, -1, 0, 0), (0, 0, 1, 1)),
    "38": ((1, 0, 0, 0), (0, -3, 1, 0), (0, -7, 0, 1)),
}
"""Cle : la base que le syllabus imprime, dans les coordonnees du chapitre."""


@pytest.mark.parametrize("cle", sorted(ORACLES))
def test_chaque_base_du_syllabus_engendre_le_meme_sous_espace(cle: str) -> None:
    calculee = bases.toutes()[cle]
    assert est_libre(calculee)
    assert est_libre(ORACLES[cle])
    assert engendrent_le_meme_sous_espace(calculee, ORACLES[cle])


def test_exercice_18_c_le_sous_espace_nul() -> None:
    assert bases.toutes()["18C"] == ()


def test_exercices_16_et_17_les_quatre_dimensions() -> None:
    assert bases.dimensions()["16"] == (2, 2, 3, 1)
    assert bases.dimensions()["17"] == (3, 3, 3, 3)


def test_exercice_22_une_equation_du_sous_espace() -> None:
    """Le syllabus : `y = t`."""
    (equation,) = bases.exercice_22()
    assert sp.Matrix([equation]).rank() == 1
    x, y, z, t = sp.symbols("x y z t")
    relation = sum(c * v for c, v in zip(equation, (x, y, z, t), strict=True))
    assert sp.solve(relation, t) == [y]


def test_exercices_26_27_37_44_les_bases_prolongees() -> None:
    prolongees = bases.prolongees()
    assert all(est_une_base(b) for b in prolongees.values())
    assert set(prolongees) == {"26", "27", "37", "44"}
    assert est_une_base((*enonces.MATRICES["27"], (2, -1, 1, 1)))
    assert est_une_base((*ORACLES["26"], (1, 0, 0, 0), (0, 0, 1, 0)))
    assert est_une_base((*ORACLES["37"], (0, 1, 0, 0), (0, 0, 1, 0)))
    assert est_une_base((*enonces.POLYNOMES["44"], (1, 0, 0, 0)))


def test_exercice_29_les_antisymetriques_sont_deja_dans_w2() -> None:
    assert appartient(enonces.MATRICES["29W1"][0], enonces.MATRICES["29W2"])


def test_exercice_33_la_valeur_de_a() -> None:
    assert bases.exercice_33() == R(11, 4)


def test_les_relations_de_dependance_que_le_corrige_ecrit() -> None:
    relations = familles.relations()
    assert relations["4"] == (3, -2, 1)
    assert relations["23"] == (-1, -3, 2, 1)
    assert relations["25"] == (1, -3, 1)
    assert relations["31"] == (-1, -2, -1, 1)
    assert relations["35"] == (1, -3, 1)
    for cle, coefficients in relations.items():
        famille = familles.famille(cle)
        nulle = sp.zeros(1, len(famille[0]))
        for c, v in zip(coefficients, famille, strict=True):
            nulle += c * sp.Matrix([list(v)])
        assert sp.simplify(nulle) == sp.zeros(1, len(famille[0])), cle


def test_une_famille_inconnue_est_refusee() -> None:
    with pytest.raises(KeyError, match="pas de famille 'absente' au chapitre VII"):
        familles.famille("absente")


def test_les_determinants_que_le_corrige_affiche() -> None:
    """Calcules a la main dans le corrige, donc recalcules ici.

    Celui du 9 avait ete ecrit -8 de tete ; il vaut -4.
    """
    assert familles.determinants() == {
        "3a": -1,
        "3b": -1,
        "9": -4,
        "12": -2 * I,
        "13a": -1,
        "13b": -3,
        "34": -13,
    }
