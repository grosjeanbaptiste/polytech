"""Le chapitre III du recueil d'algebre, « Les systemes lineaires », refait.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, exercices 1 a 24 de
la section III.7 ; l'oracle est « Solutions du 3eme chapitre » (pages 45 a
47 du syllabus). Une parametrisation ne s'ecrit pas d'une seule facon : la
reponse du syllabus n'est pas comparee terme a terme, elle est *eprouvee*
-- elle satisfait le systeme et a le bon nombre de parametres libres.

Deux reponses du syllabus portent une lettre que l'enonce n'a pas : `b` au
17, `c` au 23. Ces sujets sont « d'apres » un examen, dont le second membre
portait la lettre ; l'enonce imprime vaut `b = 1` et `c = 0`.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.application import recueil_algebre_ch03_enonces as enonces
from polytech.maths.application import recueil_algebre_ch03_resolution as resolution
from polytech.maths.application.macros_recueil_algebre_ch03 import (
    macros_du_chapitre_trois,
)
from polytech.maths.application.registre import JEUX_DE_MACROS
from polytech.maths.domain.algebre.discussion_par_le_rang import discuter_par_le_rang
from polytech.maths.domain.algebre.resolution import est_l_ensemble_des_solutions
from polytech.maths.domain.algebre.solutions import PARAMETRES, Nature

AL, BE, GA = PARAMETRES[:3]
R = sp.Rational
K, MU, LAM, T, U, A, B, M = enonces.SYMBOLES

NUMERIQUES = {
    "1": ((0, 1 - 3 * AL, AL), 1),
    "2": None,
    "3": ((0, 1, -1), 0),
    "4": ((1 - AL - BE, AL, BE), 2),
    "5": ((-14 - 2 * AL - 9 * BE, AL, 11 + 7 * BE, BE), 2),
    "6": ((-AL - GA, -AL - BE, AL, -BE + GA, BE, GA), 3),
    "7": ((6 - 2 * AL - 4 * BE, 3 - AL - 2 * BE, AL, BE), 2),
    "8": ((-AL / 2 + BE / 2, -AL / 2 - R(3, 2) * BE, AL, BE), 2),
    "9": ((-R(7, 5), -R(11, 5), R(7, 5), -R(3, 5)), 0),
    "10a": ((2 + AL, -1 - 3 * AL, AL), 1),
    "10b": ((-9 + AL, 12 - 3 * AL, AL), 1),
    "11": None,
    "12": ((R(23, 6) - 4 * AL, R(5, 6), AL), 1),
}
"""Cle de l'enonce : la reponse du syllabus et son nombre de parametres."""


@pytest.mark.parametrize("cle", sorted(NUMERIQUES))
def test_exercices_1_a_12(cle: str) -> None:
    systeme = resolution.systeme(enonces.NUMERIQUES[cle])
    oracle = NUMERIQUES[cle]
    if oracle is None:
        assert resolution.nature(enonces.NUMERIQUES[cle]) is Nature.IMPOSSIBLE
        return
    attendue, libres = oracle
    assert est_l_ensemble_des_solutions(systeme, attendue, PARAMETRES[:libres])
    calculee = resolution.solution(enonces.NUMERIQUES[cle])
    assert est_l_ensemble_des_solutions(systeme, calculee, PARAMETRES[:libres])


SOLUTIONS = {
    (13, "General"): ((K - 6) / (K - 4) - (K + 4) * AL, 1 / (K - 4) + 2 * AL, AL),
    (14, "Compatible"): ((-8 + 7 * AL) / 14, AL, R(5, 7)),
    (15, "TMoinsUn"): (-1 + AL - BE, AL, BE),
    (15, "TUn"): (1 - AL, AL, 0),
    (16, "BUn"): (AL, 1 - A * AL, 0),
    (16, "AZeroBCinq"): (AL, -R(1, 3), R(4, 3)),
    (17, "LambdaUn"): (2 - AL, -3, AL),
    (18, "AUnBUn"): (1 - AL - BE, AL, BE),
    (18, "MoinsDeux"): (AL, -R(1, 2) - AL / 2, AL),
    (19, "AMoinsUn"): (-1 + AL - BE, AL, BE),
    (19, "AUn"): (1 - AL, AL, 0),
    (20, "ATrois"): (R(11, 3) + R(4, 3) * AL, R(1, 3) - R(4, 3) * AL, AL),
    (20, "AZero"): (-R(3, 4), R(7, 4), -R(11, 4)),
    (21, "MZero"): (-2 - 3 * AL, AL, 2 - AL),
    (21, "MQuatre"): (-AL, -R(2, 3) - AL, AL),
    (21, "MDeux"): ((3 * T + 11) / 4, (T + 1) / 4, -(9 + T) / 4),
    (22, "BUnAUn"): (1 - AL, AL, 0),
    (22, "BUnADeux"): (3 - AL, AL, -1),
    (22, "BMoinsUnAZero"): (1 - AL, AL, -1 + 2 * AL),
    (22, "BMoinsUnAMoinsUn"): (1 - AL, AL, -2 + 2 * AL),
    (23, "AMoinsTrois"): (AL + R(1, 2), AL + R(5, 4), AL + R(1, 4), AL),
    (23, "AZero"): ((B + 3) / 3, (3 - 2 * B) / 3, B / 3, (B - 3) / 3),
    (24, "ATrois"): ((-1 + 19 * AL) / 5, (-7 + 3 * AL) / 5, AL),
}
"""Numero et cas : la reponse du syllabus, `b = 1` au 17 et `c = 0` au 23."""


@pytest.mark.parametrize("cas", sorted(SOLUTIONS))
def test_chaque_cas_compatible_des_exercices_13_a_24(cas: tuple[int, str]) -> None:
    numero, cle = cas
    substitution, _ = enonces.SOLUTIONS[numero][cle]
    systeme = resolution.systeme(enonces.PARAMETRIQUES[numero].subs(substitution))
    libres = sorted(
        (s for s in sp.Matrix(SOLUTIONS[cas]).free_symbols if s in PARAMETRES),
        key=PARAMETRES.index,
    )
    assert est_l_ensemble_des_solutions(systeme, SOLUTIONS[cas], libres)
    calculee = resolution.solution_du_cas(numero, cle)
    assert est_l_ensemble_des_solutions(systeme, calculee, libres)


def test_aucun_cas_declare_n_est_sans_oracle() -> None:
    declares = {(n, cle) for n, cas in enonces.SOLUTIONS.items() for cle in cas}
    assert declares == set(SOLUTIONS)


GENERIQUES = {
    15: ((U - T) / (1 - T**2), (1 - U * T) / (1 - T**2), 0),
    16: ((5 - B) / (A * (B + 1)), -2 / (B + 1), 2 * (B - 1) / (B + 1)),
    17: (
        (A * (LAM**2 - 2 * LAM) - 4 * (LAM + 1)) / (LAM * (LAM - 1) * (LAM - 2)),
        (LAM + 2) / (LAM * (LAM - 2)),
        (A * (-(LAM**2) + 2 * LAM) - 2 * LAM**2 + 6 * LAM + 4)
        / (LAM * (LAM - 1) * (LAM - 2)),
    ),
    18: (
        (A - B) / ((1 - B) * (B + 2)),
        (2 - A * B - A) / (A * (1 - B) * (B + 2)),
        (A - B) / ((1 - B) * (B + 2)),
    ),
    19: ((B - A) / (1 - A**2), (1 - A * B) / (1 - A**2), 0),
    22: (
        (A**2 * B**2 - A**2 - 2 * A - A * B + B + 1 + A**2 * B)
        / ((B + 1) * (B**2 - 1)),
        -(B + 1 - A) * (1 - A * B) / ((B - 1) * (B + 1) ** 2),
        (-(A**2) + B - A + 1) / (B + 1) ** 2,
    ),
    24: ((-A + 16) / (5 * (A + 3)), -(7 * A + 18) / (5 * (A + 3)), 1 / (A + 3)),
}
"""La solution unique, hors des valeurs singulieres, telle que le syllabus l'ecrit."""


@pytest.mark.parametrize("numero", sorted(GENERIQUES))
def test_la_solution_unique_hors_des_valeurs_singulieres(numero: int) -> None:
    calculee = sp.Matrix(resolution.generique(numero))
    assert sp.simplify(calculee - sp.Matrix(GENERIQUES[numero])) == sp.zeros(3, 1)


DETERMINANTS = {
    15: (T - 1) * (T + 1) ** 2,
    16: A * (B - 1) * (B + 1),
    17: LAM * (LAM - 1) * (LAM - 2),
    18: A * (B - 1) ** 2 * (B + 2),
    19: -(A - 1) * (A + 1) ** 2,
    20: -(A - 2) * (A - 3) * (A - 4),
    21: M * (M - 1) * (M - 4),
    22: -(B - 1) * (B + 1) ** 2,
    23: -((A - 1) ** 3) * (A + 3),
    24: 5 * (A - 3) * (A + 3),
}


@pytest.mark.parametrize("numero", sorted(DETERMINANTS))
def test_le_determinant_donne_les_valeurs_singulieres(numero: int) -> None:
    assert sp.expand(resolution.determinant(numero) - DETERMINANTS[numero]) == 0


IMPOSSIBLES = [
    (13, {K: 4}),
    (14, {MU: 4, LAM: 0}),
    (14, {MU: 0, LAM: -10}),
    (15, {T: -1, U: 0}),
    (15, {T: 1, U: 0}),
    (16, {B: -1}),
    (16, {A: 0, B: 2}),
    (17, {LAM: 0}),
    (17, {LAM: 1, A: 0}),
    (17, {LAM: 2}),
    (18, {A: 0}),
    (18, {A: 3, B: 1}),
    (18, {A: 3, B: -2}),
    (19, {A: -1, B: 0}),
    (19, {A: 1, B: 0}),
    (20, {A: 2}),
    (20, {A: 4}),
    (21, {M: 0, T: 0}),
    (21, {M: 1}),
    (21, {M: 4, T: 0}),
    (22, {B: 1, A: 0}),
    (22, {B: -1, A: 3}),
    (23, {A: 1}),
    (23, {A: -3, B: 0}),
    (24, {A: -3}),
]


@pytest.mark.parametrize(("numero", "substitution"), IMPOSSIBLES)
def test_les_cas_impossibles(numero: int, substitution: dict) -> None:
    augmentee = enonces.PARAMETRIQUES[numero].subs(substitution)
    assert resolution.nature(augmentee) is Nature.IMPOSSIBLE


SANS_EXCEPTION = [
    (16, {B: -1}, A),
    (17, {LAM: 0}, A),
    (17, {LAM: 2}, A),
    (18, {A: 0}, B),
    (21, {M: 1}, T),
    (23, {A: 1}, B),
]


@pytest.mark.parametrize(("numero", "substitution", "autre"), SANS_EXCEPTION)
def test_impossible_quel_que_soit_l_autre_parametre(
    numero: int, substitution: dict, autre: sp.Symbol
) -> None:
    """Le syllabus dit « impossible » sans condition : aucune valeur n'y echappe."""
    systeme = resolution.systeme(enonces.PARAMETRIQUES[numero].subs(substitution))
    discussion = discuter_par_le_rang(systeme, autre)
    assert discussion.generique is Nature.IMPOSSIBLE
    assert discussion.particulieres == {}


def test_la_condition_de_compatibilite_se_lit_sur_la_forme_echelonnee() -> None:
    """Exercice 14 : deux lignes `0 = mu - 4` et `0 = lambda + 10`."""
    forme = resolution.forme_du_cas(14, "General")
    conditions = {
        sp.factor(forme[i, 3]) for i in range(forme.rows) if forme[i, :3].is_zero_matrix
    }
    assert any(sp.solve(c, MU) == [4] for c in conditions)
    assert any(sp.solve(c, LAM) == [-10] for c in conditions)


def test_un_cas_inconnu_est_refuse() -> None:
    with pytest.raises(KeyError, match="exercice 13 : pas de cas 'Absent'"):
        resolution.solution_du_cas(13, "Absent")
    with pytest.raises(KeyError, match="exercice 13 : pas de cas 'Absent'"):
        resolution.forme_du_cas(13, "Absent")


def test_une_solution_s_ecrit_point_plus_parametres_fois_directions() -> None:
    """`23/6 - 4 alpha`, et non `-(24 alpha - 23)/6` : la forme du syllabus."""
    douze = resolution.solution(enonces.NUMERIQUES["12"])
    assert douze[0] == sp.Add(R(23, 6), -4 * AL, evaluate=False) or (
        douze[0].coeff(AL) == -4 and douze[0].subs(AL, 0) == R(23, 6)
    )
    assert not isinstance(douze[0], sp.Mul)
    treize = resolution.solution_du_cas(13, "General")
    assert sp.simplify(treize[0].coeff(AL) + (K + 4)) == 0


def test_le_jeu_de_macros_du_chapitre_trois() -> None:
    macros = macros_du_chapitre_trois()
    assert all(nom.startswith("AlgTrois") for nom in macros)
    assert all(valeur.strip() for valeur in macros.values())
    assert "AlgTroisDeuxSolution" not in macros
    assert "AlgTroisDeuxEchelonnee" in macros
    assert (
        macros["AlgTroisDixASolution"]
        == r"\left(\alpha + 2,\; - 3 \alpha - 1,\; \alpha\right)"
    )
    assert (
        macros["AlgTroisVingtQuatreDet"] == r"5 \left(a - 3\right) \left(a + 3\right)"
    )
    assert r"\mu - 4" in macros["AlgTroisQuatorzeGeneralForme"]
    assert "algebre-03-systemes-lineaires" in JEUX_DE_MACROS


def test_exercice_16_la_solution_pour_b_egal_un_vaut_aussi_en_a_nul() -> None:
    """On y laisse `x` libre : prendre `x` pour pivot diviserait par `a`."""
    generale = sp.Matrix(resolution.solution_du_cas(16, "BUn"))
    en_zero = enonces.PARAMETRIQUES[16].subs({B: 1, A: 0})
    systeme = resolution.systeme(en_zero)
    assert est_l_ensemble_des_solutions(systeme, list(generale.subs(A, 0)), [AL])
