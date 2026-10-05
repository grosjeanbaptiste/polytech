"""Syllabus d'algebre, chapitre III : les systemes lineaires.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, pages 31 a 47. Les
quatre exemples resolus (III.2 a III.5), par chacune des demarches que le
chapitre deroule, puis les « Solutions du 3eme chapitre » : chaque reponse
officielle des exercices 1 a 24 est recalculee, cas par cas.

Trois ecarts entre un enonce et sa solution sont etablis :

* exercice 14 : « mu != 4, lambda != -10 : impossible » se lit « ou » ;
* exercice 17 : la solution garde une lettre `b` la ou l'enonce imprime 1 ;
* exercice 23 : la solution porte une lettre `c` la ou l'enonce imprime 0.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre.determinants import Determinant
from polytech.maths.domain.algebre.discussion import discuter
from polytech.maths.domain.algebre.echelonnement import echelonner_simplement
from polytech.maths.domain.algebre.methodes import par_l_inverse, par_le_pivot
from polytech.maths.domain.algebre.resolution import (
    est_l_ensemble_des_solutions,
    par_echelonnement_simple,
)
from polytech.maths.domain.algebre.solutions import (
    PARAMETRES,
    Nature,
    cramer_generalise,
    nature_des_solutions,
    nombre_de_parametres,
    solution_generale,
)
from polytech.maths.domain.algebre.systemes import SystemeLineaire

M = sp.Matrix
R = sp.Rational
ALPHA, BETA, GAMMA, _ = PARAMETRES
A, B, C, K, T, U, MM, LAMBDA, MU = sp.symbols("a b c k t u m lambda mu")
Lignes = list[list[sp.Expr]]
Valeurs = dict[sp.Symbol, sp.Expr]

EXEMPLE_1: Lignes = [[1, 1, 2, 9], [2, 4, -3, 1], [3, 6, -5, 0]]
EXEMPLE_2: Lignes = [[3, -2, 7, 1], [-6, 4, 3, 2]]
EXEMPLE_3: Lignes = [
    [1, -1, 2, -1, -1],
    [2, 1, -2, -2, -2],
    [-1, 2, -4, 1, 1],
    [4, -1, 2, -4, -4],
]
EXEMPLE_4: Lignes = [
    [A - 2, 2, -1, 1],
    [2, A, 2, 1],
    [2 * A, 2 * (A + 1), A + 1, 2],
]


def _systeme(lignes: Lignes, valeurs: Valeurs | None = None) -> SystemeLineaire:
    """La derniere colonne est le second membre."""
    complete = M(lignes).subs(valeurs or {})
    return SystemeLineaire(complete[:, :-1], complete[:, -1])


def _est_impossible(lignes: Lignes, valeurs: Valeurs | None = None) -> bool:
    return nature_des_solutions(_systeme(lignes, valeurs)) is Nature.IMPOSSIBLE


def _est_decrit_par(
    lignes: Lignes,
    solution: tuple[sp.Expr, ...],
    parametres: tuple[sp.Symbol, ...] = (),
    valeurs: Valeurs | None = None,
) -> bool:
    point = tuple(sp.sympify(s).subs(valeurs or {}) for s in solution)
    return est_l_ensemble_des_solutions(_systeme(lignes, valeurs), point, parametres)


def test_l_exemple_1_par_inversion() -> None:
    """`A^-1 = [[2, -17, 11], [-1, 11, -7], [0, 3, -2]]`, et `A^-1 b = (1, 2, 3)`."""
    systeme = _systeme(EXEMPLE_1)
    assert systeme.matrice.inv() == M([[2, -17, 11], [-1, 11, -7], [0, 3, -2]])
    assert par_l_inverse(systeme) == M([1, 2, 3])


def test_l_exemple_1_par_echelonnement() -> None:
    """`l2 - 2 l1`, `l3 - 3 l1`, `2 l3 - 3 l2` : le systeme triangulaire, page 32."""
    resultat = par_le_pivot(_systeme(EXEMPLE_1))
    assert resultat.triangulaire.matrice == M([[1, 1, 2], [0, 2, -7], [0, 0, -1]])
    assert resultat.triangulaire.second_membre == M([9, -17, -3])
    assert resultat.solution == M([1, 2, 3])


def test_l_exemple_1_par_echelonnement_simple() -> None:
    """La matrice completee devient `[I | (1, 2, 3)]` : une solution, sans direction."""
    simple = echelonner_simplement(M(EXEMPLE_1))
    assert simple == sp.eye(3).row_join(M([1, 2, 3]))
    ensemble = par_echelonnement_simple(_systeme(EXEMPLE_1))
    assert ensemble.particuliere == (1, 2, 3)
    assert ensemble.directions == ()
    assert ensemble.forme_generale() == (1, 2, 3)


def test_l_exemple_2_par_une_sous_matrice_inversible() -> None:
    """La sous-matrice `[[3, 7], [-6, 3]]`, `y` libre : `((-11 + 34 y)/51, y, 4/17)`."""
    reduction = cramer_generalise(_systeme(EXEMPLE_2))
    assert reduction.sous_matrice == M([[3, 7], [-6, 3]])
    assert reduction.sous_matrice.inv() == M([[3, -7], [6, 3]]) / 51
    attendue = ((-11 + 34 * ALPHA) / 51, ALPHA, R(4, 17))
    assert all(
        sp.simplify(o - a) == 0
        for o, a in zip(reduction.solution, attendue, strict=True)
    )


def test_l_exemple_2_par_echelonnement_simple() -> None:
    """`[[1, -34/51, 0, -11/51], [0, 0, 1, 4/17]]`, puis la forme generale."""
    simple = echelonner_simplement(M(EXEMPLE_2))
    assert simple == M([[1, -R(34, 51), 0, -R(11, 51)], [0, 0, 1, R(4, 17)]])
    ensemble = par_echelonnement_simple(_systeme(EXEMPLE_2))
    assert ensemble.particuliere == (-R(11, 51), 0, R(4, 17))
    assert ensemble.directions == ((R(34, 51), 1, 0),)
    assert ensemble.forme_generale() == (-R(11, 51) + R(2, 3) * ALPHA, ALPHA, R(4, 17))


def test_l_exemple_3_par_echelonnement_simple() -> None:
    """`(-1, 0, 0, 0) + alpha (0, 2, 1, 0) + beta (1, 0, 0, 1)`."""
    simple = echelonner_simplement(M(EXEMPLE_3))
    assert simple[:2, :] == M([[1, 0, 0, -1, -1], [0, 1, -2, 0, 0]])
    assert simple[2:, :] == sp.zeros(2, 5)
    ensemble = par_echelonnement_simple(_systeme(EXEMPLE_3))
    assert ensemble.particuliere == (-1, 0, 0, 0)
    assert ensemble.directions == ((0, 2, 1, 0), (1, 0, 0, 1))
    assert ensemble.forme_generale() == (-1 + BETA, 2 * ALPHA, ALPHA, BETA)


def test_l_exemple_3_n_a_aucune_sous_matrice_d_ordre_trois_inversible() -> None:
    """La remarque de la page 36 : le rang est deux, donc deux degres de liberte."""
    systeme = _systeme(EXEMPLE_3)
    assert systeme.rang() == 2
    assert nombre_de_parametres(systeme) == 2


def test_la_matrice_echelonnee_de_l_exemple_4() -> None:
    """Page 39 : la derniere ligne est `-a(a-1)(a-2) z = 2a(a-3)`."""
    echelonnee = [
        [2, A, 2, 1],
        [0, 2, -(A - 1), 2],
        [0, 0, -A * (A - 1) * (A - 2), 2 * A * (A - 3)],
    ]
    depart, arrivee = _systeme(EXEMPLE_4), _systeme(echelonnee)
    assert sp.simplify(par_l_inverse(depart) - par_l_inverse(arrivee)) == sp.zeros(3, 1)


def test_la_solution_unique_de_l_exemple_4() -> None:
    """`((a-5)/((a-1)(a-2)), 1/(a-2), -2(a-3)/((a-1)(a-2)))`, hors de 0, 1 et 2."""
    attendue = (
        (A - 5) / ((A - 1) * (A - 2)),
        1 / (A - 2),
        -2 * (A - 3) / ((A - 1) * (A - 2)),
    )
    ensemble = par_echelonnement_simple(_systeme(EXEMPLE_4))
    assert ensemble.directions == ()
    assert all(
        sp.simplify(o - a) == 0
        for o, a in zip(ensemble.particuliere, attendue, strict=True)
    )


def test_la_discussion_de_l_exemple_4() -> None:
    """`a = 0` : indetermine ; `a = 1` et `a = 2` : impossible. Ni -1 ni 4 n'y sont."""
    discussion = discuter(_systeme(EXEMPLE_4), A)
    assert discussion == {
        0: Nature.INDETERMINE,
        1: Nature.IMPOSSIBLE,
        2: Nature.IMPOSSIBLE,
    }
    for inutile in (-1, 4):
        assert nature_des_solutions(_systeme(EXEMPLE_4, {A: inutile})) is Nature.UNIQUE


def test_le_cas_a_nul_de_l_exemple_4() -> None:
    """`(1/2, 1, 0) + alpha (-1, -1/2, 1)`."""
    ensemble = par_echelonnement_simple(_systeme(EXEMPLE_4, {A: 0}))
    assert ensemble.particuliere == (R(1, 2), 1, 0)
    assert ensemble.directions == ((-1, -R(1, 2), 1),)


def test_un_systeme_impossible_se_lit_sur_un_pivot_du_second_membre() -> None:
    """Exemple 4, `a = 1` : la derniere equation est `0 z = -4`."""
    with pytest.raises(ValueError, match="systeme impossible"):
        par_echelonnement_simple(_systeme(EXEMPLE_4, {A: 1}))


def test_la_forme_generale_refuse_de_manquer_de_parametres() -> None:
    """Deux directions pour un seul parametre : rien n'est tronque en silence."""
    ensemble = par_echelonnement_simple(_systeme(EXEMPLE_3))
    with pytest.raises(ValueError, match="trop peu de parametres") as refus:
        ensemble.forme_generale((ALPHA,))
    assert "1 pour 2" in str(refus.value)
    assert len(ensemble.forme_generale((ALPHA, BETA))) == 4


def test_la_resolution_concorde_avec_la_solution_generale_existante() -> None:
    """Lien : deux chemins, l'echelonnement simple et `linsolve`, un meme ensemble."""
    for lignes in (EXEMPLE_1, EXEMPLE_2, EXEMPLE_3):
        systeme = _systeme(lignes)
        generale = solution_generale(systeme)
        libres = tuple(sorted(M(generale).free_symbols, key=str))
        assert est_l_ensemble_des_solutions(systeme, generale, libres)
        forme = par_echelonnement_simple(systeme).forme_generale()
        libres = tuple(sorted(M(forme).free_symbols, key=str))
        assert est_l_ensemble_des_solutions(systeme, forme, libres)


def test_une_description_fausse_ou_incomplete_est_refusee() -> None:
    """Un point hors du systeme ; trop peu de parametres ; un parametre redondant."""
    systeme = _systeme([[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2]])
    juste = (1 - ALPHA - BETA, ALPHA, BETA)
    assert est_l_ensemble_des_solutions(systeme, juste, (ALPHA, BETA))
    assert not est_l_ensemble_des_solutions(systeme, (ALPHA, BETA, 0), (ALPHA, BETA))
    assert not est_l_ensemble_des_solutions(systeme, (1 - ALPHA, ALPHA, 0), (ALPHA,))
    redondante = (1 - ALPHA - BETA, ALPHA + BETA, 0)
    assert not est_l_ensemble_des_solutions(systeme, redondante, (ALPHA, BETA))
    assert not est_l_ensemble_des_solutions(systeme, (1, 0, 0), ())


EX5: Lignes = [
    [1, 2, 1, 2, -3],
    [3, 6, 4, -1, 2],
    [4, 8, 5, 1, -1],
    [-2, -4, -3, 3, -5],
]
EX6: Lignes = [
    [1, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, -2, -1, 2, 0],
    [-3, 0, -3, -1, -1, -2, 0],
    [-1, 0, -1, 0, 0, -1, 0],
]
EX7: Lignes = [[1, -1, 1, 2, 3], [2, -1, 3, 6, 9], [3, -2, 4, 8, 12], [1, 0, 2, 4, 6]]
EX9: Lignes = [
    [1, 1, 2, -3, 1],
    [2, 3, 6, -5, 2],
    [3, -1, 2, -7, 5],
    [1, 2, 3, -1, -1],
]
EX11: Lignes = [
    [2, 1, -1, 1, 1],
    [3, -2, 2, -3, 2],
    [5, 1, -1, 2, -1],
    [2, -1, 1, -3, 4],
]


@pytest.mark.parametrize(
    ("lignes", "solution", "parametres"),
    [
        ([[2, 1, 3, 1], [4, 1, 3, 1], [3, 1, 3, 1]], (0, 1 - 3 * ALPHA, ALPHA), 1),
        ([[2, 4, 3, 1], [-1, 2, 1, 1], [2, 3, 2, 1]], (0, 1, -1), 0),
        (
            [[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2]],
            (1 - ALPHA - BETA, ALPHA, BETA),
            2,
        ),
        (EX5, (-14 - 2 * ALPHA - 9 * BETA, ALPHA, 11 + 7 * BETA, BETA), 2),
        (
            EX6,
            (-ALPHA - GAMMA, -ALPHA - BETA, ALPHA, -BETA + GAMMA, BETA, GAMMA),
            3,
        ),
        (EX7, (6 - 2 * ALPHA - 4 * BETA, 3 - ALPHA - 2 * BETA, ALPHA, BETA), 2),
        (
            [[1, 1, 1, 1, 0], [1, 3, 2, 4, 0], [2, 0, 1, -1, 0]],
            (-ALPHA / 2 + BETA / 2, -ALPHA / 2 - 3 * BETA / 2, ALPHA, BETA),
            2,
        ),
        (EX9, (-R(7, 5), -R(11, 5), R(7, 5), -R(3, 5)), 0),
        (
            [[1, 1, 2, 1], [3, 2, 3, 4], [5, 4, 7, 6]],
            (2 + ALPHA, -1 - 3 * ALPHA, ALPHA),
            1,
        ),
        (
            [[1, 1, 2, 3], [3, 2, 3, -3], [5, 4, 7, 3]],
            (-9 + ALPHA, 12 - 3 * ALPHA, ALPHA),
            1,
        ),
        ([[1, -1, 4, 3], [3, -9, 12, 4]], (R(23, 6) - 4 * ALPHA, R(5, 6), ALPHA), 1),
    ],
    ids=["1", "3", "4", "5", "6", "7", "8", "9", "10a", "10b", "12"],
)
def test_les_reponses_des_exercices_1_a_12(
    lignes: Lignes, solution: tuple[sp.Expr, ...], parametres: int
) -> None:
    """L'ensemble officiel est celui des solutions, et l'echelonnement le rend."""
    assert _est_decrit_par(lignes, solution, PARAMETRES[:parametres])
    ensemble = par_echelonnement_simple(_systeme(lignes))
    assert len(ensemble.directions) == parametres
    libres = PARAMETRES[:parametres]
    assert _est_decrit_par(lignes, ensemble.forme_generale(), libres)


def test_les_exercices_2_et_11_sont_impossibles() -> None:
    """Solutions 2 et 11."""
    assert _est_impossible([[1, 1, 0, 3], [2, 3, 1, 9], [-1, 3, 4, 2]])
    assert _est_impossible(EX11)


def test_la_reponse_de_l_exercice_13() -> None:
    """`k = 4` : impossible ; sinon un degre de liberte."""
    lignes: Lignes = [[1, 2, K, 1], [2, K, 8, 3]]
    assert _est_impossible(lignes, {K: 4})
    solution = (
        (K - 6) / (K - 4) - (K + 4) * ALPHA,
        1 / (K - 4) + 2 * ALPHA,
        ALPHA,
    )
    assert _est_decrit_par(lignes, solution, (ALPHA,))


def test_la_reponse_de_l_exercice_14() -> None:
    """`mu = 4` et `lambda = -10` : `((-8 + 7 alpha)/14, alpha, 5/7)`.

    Des que l'une des deux egalites manque, le systeme est impossible : le
    « mu != 4, lambda != -10 » de la solution se lit « ou ».
    """
    lignes: Lignes = [
        [2, -1, 3, 1],
        [-4, 2, 1, 3],
        [-2, 1, 4, MU],
        [10, -5, -6, LAMBDA],
    ]
    solution = ((-8 + 7 * ALPHA) / 14, ALPHA, R(5, 7))
    assert _est_decrit_par(lignes, solution, (ALPHA,), {MU: 4, LAMBDA: -10})
    assert _est_impossible(lignes, {MU: 5, LAMBDA: -10})
    assert _est_impossible(lignes, {MU: 4, LAMBDA: 0})
    assert _est_impossible(lignes)


@pytest.mark.parametrize(
    ("lignes", "p", "q"),
    [
        ([[1, T, 1, U], [T, 1, T, 1], [1, T, -T, U]], T, U),
        ([[2, 2 * A, 1 - A, 2 * B], [A, 1, A, 1], [1, A, 1, B]], A, B),
    ],
    ids=["15", "19"],
)
def test_les_reponses_des_exercices_15_et_19(
    lignes: Lignes, p: sp.Symbol, q: sp.Symbol
) -> None:
    """Deux systemes differents, une meme discussion : en `(t, u)` et en `(a, b)`."""
    assert Determinant(_systeme(lignes).matrice).valeurs_singulieres(p) == (-1, 1)
    plan = (-1 + ALPHA - BETA, ALPHA, BETA)
    assert _est_decrit_par(lignes, plan, (ALPHA, BETA), {p: -1, q: -1})
    assert _est_impossible(lignes, {p: -1})
    assert _est_decrit_par(lignes, (1 - ALPHA, ALPHA, 0), (ALPHA,), {p: 1, q: 1})
    assert _est_impossible(lignes, {p: 1})
    unique = ((q - p) / (1 - p**2), (1 - q * p) / (1 - p**2), 0)
    assert _est_decrit_par(lignes, unique)


def test_la_reponse_de_l_exercice_16() -> None:
    """Aout 94 : cinq cas, selon `b = +-1`, `a = 0` et `b = 5`."""
    lignes: Lignes = [[A, B, 2, 1], [A, 2 * B - 1, 3, 1], [A, B, B + 3, 2 * B - 1]]
    assert _est_impossible(lignes, {B: -1})
    assert _est_decrit_par(lignes, (ALPHA, 1 - A * ALPHA, 0), (ALPHA,), {B: 1})
    assert _est_impossible(lignes, {A: 0})
    droite = (ALPHA, -R(1, 3), R(4, 3))
    assert _est_decrit_par(lignes, droite, (ALPHA,), {A: 0, B: 5})
    unique = ((5 - B) / (A * (B + 1)), -2 / (B + 1), 2 * (B - 1) / (B + 1))
    assert _est_decrit_par(lignes, unique)


def _exercice_17(second: sp.Expr) -> Lignes:
    return [
        [LAMBDA - 2, 2, -1, A],
        [2, LAMBDA, 2, second],
        [2 * LAMBDA, 2 * (LAMBDA + 1), LAMBDA + 1, A],
    ]


def test_la_reponse_de_l_exercice_17_telle_que_l_enonce_la_pose() -> None:
    """D'apres janvier 97, second membre `(a, 1, a)` : la solution lue avec `b = 1`."""
    lignes = _exercice_17(sp.Integer(1))
    assert Determinant(_systeme(lignes).matrice).valeurs_singulieres(LAMBDA) == (
        0,
        1,
        2,
    )
    assert _est_impossible(lignes, {LAMBDA: 0})
    assert _est_impossible(lignes, {LAMBDA: 1})
    assert _est_decrit_par(lignes, (2 - MU, -3, MU), (MU,), {LAMBDA: 1, A: -8})
    assert _est_impossible(lignes, {LAMBDA: 2})


def test_la_solution_de_l_exercice_17_porte_un_b_que_l_enonce_n_a_pas() -> None:
    """La formule officielle resout le second membre `(a, b, a)`, pas `(a, 1, a)`.

    Et ses conditions, elles, sont celles de `b = 1` : avec `b` quelconque,
    le cas `lambda = 1` demande `a = -8b`, et `lambda = 0` n'est plus
    impossible quand `b = 0`.
    """
    produit = LAMBDA * (LAMBDA - 1) * (LAMBDA - 2)
    officielle = (
        (A * (LAMBDA**2 - 2 * LAMBDA) - 4 * B * (LAMBDA + 1)) / produit,
        B * (LAMBDA + 2) / (LAMBDA * (LAMBDA - 2)),
        (A * (-(LAMBDA**2) + 2 * LAMBDA) + B * (-2 * LAMBDA**2 + 6 * LAMBDA + 4))
        / produit,
    )
    assert _est_decrit_par(_exercice_17(B), officielle)
    assert not _est_decrit_par(_exercice_17(sp.Integer(1)), officielle)
    assert _est_decrit_par(_exercice_17(sp.Integer(1)), officielle, (), {B: 1})
    droite = (2 * B - MU, -3 * B, MU)
    assert _est_decrit_par(_exercice_17(B), droite, (MU,), {LAMBDA: 1, A: -8 * B})
    assert not _est_impossible(_exercice_17(B), {LAMBDA: 0, B: 0})


def test_la_reponse_de_l_exercice_18() -> None:
    """Janvier 98 : six cas."""
    lignes: Lignes = [[B, A, 1, 1], [1, A * B, 1, A], [1, A, B, 1]]
    assert _est_impossible(lignes, {A: 0})
    plan = (1 - ALPHA - BETA, ALPHA, BETA)
    assert _est_decrit_par(lignes, plan, (ALPHA, BETA), {A: 1, B: 1})
    droite = (ALPHA, -R(1, 2) - ALPHA / 2, ALPHA)
    assert _est_decrit_par(lignes, droite, (ALPHA,), {A: -2, B: -2})
    assert _est_impossible(lignes, {B: 1})
    assert _est_impossible(lignes, {B: -2})
    commun = (A - B) / ((1 - B) * (B + 2))
    unique = (commun, (2 - A * B - A) / (A * (1 - B) * (B + 2)), commun)
    assert _est_decrit_par(lignes, unique)


def test_la_reponse_de_l_exercice_20() -> None:
    """Janvier 99 : le determinant s'annule en 2, 3 et 4."""
    lignes: Lignes = [
        [A - 2, A - 2, 0, 2 * A - 2],
        [2, A - 1, A - 3, A + 5],
        [3 - A, 2 * A - 3, 2 * A - 2, A - 2],
    ]
    assert discuter(_systeme(lignes), A) == {
        2: Nature.IMPOSSIBLE,
        3: Nature.INDETERMINE,
        4: Nature.IMPOSSIBLE,
    }
    droite = (R(11, 3) + R(4, 3) * ALPHA, R(1, 3) - R(4, 3) * ALPHA, ALPHA)
    assert _est_decrit_par(lignes, droite, (ALPHA,), {A: 3})
    assert _est_decrit_par(lignes, (-R(3, 4), R(7, 4), -R(11, 4)), (), {A: 0})


def test_la_reponse_de_l_exercice_21() -> None:
    """Janvier 00 : le determinant s'annule en `m = 0`, 1 et 4."""
    lignes: Lignes = [
        [MM - 1, -3, 0, 2],
        [2 * MM, MM - 1, 3 * MM - 1, 2 * T],
        [1, 3, MM, T - 1],
    ]
    singulieres = Determinant(_systeme(lignes).matrice).valeurs_singulieres(MM)
    assert singulieres == (0, 1, 4)
    droite = (-2 - 3 * ALPHA, ALPHA, 2 - ALPHA)
    assert _est_decrit_par(lignes, droite, (ALPHA,), {MM: 0, T: -1})
    assert _est_impossible(lignes, {MM: 0})
    assert _est_impossible(lignes, {MM: 1})
    droite = (-ALPHA, -R(2, 3) - ALPHA, ALPHA)
    assert _est_decrit_par(lignes, droite, (ALPHA,), {MM: 4, T: -1})
    assert _est_impossible(lignes, {MM: 4})
    unique = ((3 * T + 11) / 4, (T + 1) / 4, -(9 + T) / 4)
    assert _est_decrit_par(lignes, unique, (), {MM: 2})


def test_la_reponse_de_l_exercice_22() -> None:
    """Janvier 01 : sept cas, selon `b = +-1`."""
    lignes: Lignes = [[1, 1, B + 1, 1], [1, B, 1, A], [B, 1, -1, A**2]]
    assert Determinant(_systeme(lignes).matrice).valeurs_singulieres(B) == (-1, 1)
    un = (ALPHA,)
    assert _est_decrit_par(lignes, (1 - ALPHA, ALPHA, 0), un, {B: 1, A: 1})
    assert _est_decrit_par(lignes, (3 - ALPHA, ALPHA, -1), un, {B: 1, A: 2})
    assert _est_impossible(lignes, {B: 1})
    droite = (1 - ALPHA, ALPHA, -1 + 2 * ALPHA)
    assert _est_decrit_par(lignes, droite, un, {B: -1, A: 0})
    droite = (1 - ALPHA, ALPHA, -2 + 2 * ALPHA)
    assert _est_decrit_par(lignes, droite, un, {B: -1, A: -1})
    assert _est_impossible(lignes, {B: -1})
    numerateur = A**2 * B**2 - A**2 - 2 * A - A * B + B + 1 + A**2 * B
    unique = (
        numerateur / ((B + 1) * (B**2 - 1)),
        -(B + 1 - A) * (1 - A * B) / ((B - 1) * (B + 1) ** 2),
        (-(A**2) + B - A + 1) / (B + 1) ** 2,
    )
    assert _est_decrit_par(lignes, unique)


def _exercice_23(second: sp.Expr) -> Lignes:
    return [
        [1, A, 1, 1, B],
        [A, 1, 1, 1, second],
        [1, 1, A, 1, 1],
        [1, 1, 1, A, 2],
    ]


def test_la_reponse_de_l_exercice_23_telle_que_l_enonce_la_pose() -> None:
    """D'apres aout 02, second membre `(b, 0, 1, 2)` : la solution lue avec `c = 0`."""
    lignes = _exercice_23(sp.Integer(0))
    assert Determinant(_systeme(lignes).matrice).valeurs_singulieres(A) == (-3, 1)
    assert _est_impossible(lignes, {A: 1})
    assert _est_impossible(lignes, {A: -3})
    droite = (ALPHA + R(1, 2), ALPHA + R(5, 4), ALPHA + R(1, 4), ALPHA)
    assert _est_decrit_par(lignes, droite, (ALPHA,), {A: -3, B: -3})
    unique = ((B + 3) / 3, (3 - 2 * B) / 3, B / 3, (B - 3) / 3)
    assert _est_decrit_par(lignes, unique, (), {A: 0})


def test_la_solution_de_l_exercice_23_porte_un_c_que_l_enonce_n_a_pas() -> None:
    """La formule du cas `a = 0` resout le second membre `(b, c, 1, 2)`."""
    officielle = (
        (B - 2 * C + 3) / 3,
        (3 + C - 2 * B) / 3,
        (C + B) / 3,
        (C + B - 3) / 3,
    )
    assert _est_decrit_par(_exercice_23(C), officielle, (), {A: 0})
    assert not _est_decrit_par(_exercice_23(sp.Integer(0)), officielle, (), {A: 0})


def test_la_reponse_de_l_exercice_24() -> None:
    """Aout 04 : `a = 3` indetermine, `a = -3` impossible, sinon une solution."""
    lignes: Lignes = [[1, -3, -2, 4], [3, -4, -9, 5], [4, -7, A**2 - 20, A + 6]]
    assert discuter(_systeme(lignes), A) == {
        -3: Nature.IMPOSSIBLE,
        3: Nature.INDETERMINE,
    }
    droite = ((-1 + 19 * ALPHA) / 5, (-7 + 3 * ALPHA) / 5, ALPHA)
    assert _est_decrit_par(lignes, droite, (ALPHA,), {A: 3})
    unique = (
        (-A + 16) / (5 * (A + 3)),
        -(7 * A + 18) / (5 * (A + 3)),
        1 / (A + 3),
    )
    assert _est_decrit_par(lignes, unique)
