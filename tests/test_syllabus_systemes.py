"""Syllabus de mathematiques appliquees, chapitre 6 : les systemes differentiels.

Source : `Mathematique/SYLLABUS/Syllabus_Mathematiques_appliquees_2016_2017.pdf`,
pages 101 a 122. Les iterees vectorielles de l'exemple 6.1, la resolvante
`(pI - A)^-1` des exemples 6.3, la matrice de Wronsky et la formule de la
trace (exemple 6.4), le noyau `R(t1, t2)`, la mise sous forme normale d'une
equation d'ordre `n` (exemples 6.5 et 6.6), le modele predateur-proie.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.differentielles.forme_normale import (
    force_normale,
    matrice_compagnon,
    particuliere_par_le_wronskien,
)
from polytech.maths.domain.differentielles.operateurs import D, OperateurDifferentiel
from polytech.maths.domain.differentielles.plan_de_phase import (
    Nature,
    nature_du_point_critique,
)
from polytech.maths.domain.differentielles.predateur_proie import PredateurProie
from polytech.maths.domain.differentielles.systemes import SystemeDifferentiel
from polytech.maths.domain.differentielles.systemes_normaux import (
    iterees_vectorielles,
    norme_de_la_matrice,
)
from polytech.maths.domain.differentielles.wronskien import (
    Wronskien,
    determinant_par_la_trace,
)
from polytech.maths.domain.laplace.green import fonction_de_green
from polytech.maths.domain.symboles import CONSTANTE
from polytech.maths.domain.symboles import IMAGE as P
from polytech.maths.domain.symboles import TEMPS as T

R = sp.Rational
exp, cosh, sinh, sqrt = sp.exp, sp.cosh, sp.sinh, sp.sqrt
ECHANGE = sp.Matrix([[0, 1], [1, 0]])
EXEMPLE_6_4 = sp.Matrix([[1, -1], [-4, 1]])
SOLUTIONS_6_6 = (T**2, T**3, T**-2)
COMPAGNON_6_6 = matrice_compagnon((12, -6 * T, 0, T**3))


def _nulle(matrice: sp.Matrix) -> bool:
    return bool(sp.simplify(matrice) == sp.zeros(*matrice.shape))


def _egales(obtenue: sp.Matrix, attendue: sp.Matrix) -> bool:
    """Compare deux matrices une fois les hyperboliques ecrites en exponentielles."""
    ecart = (obtenue - attendue).applyfunc(lambda e: e.rewrite(exp))
    return _nulle(ecart)


def test_les_iterees_vectorielles_de_l_exemple_6_1() -> None:
    """`y1' = y2`, `y2' = y1`, depuis `(1, 0)` : `(1, t)`, `(1 + t^2/2, t)`, ..."""
    iterees = iterees_vectorielles(ECHANGE, sp.Matrix([1, 0]), nombre=3)
    assert iterees[0] == sp.Matrix([0, 0])
    assert iterees[1] == sp.Matrix([1, 0])
    assert iterees[2] == sp.Matrix([1, T])
    assert iterees[3] == sp.Matrix([1 + T**2 / 2, T])


def test_les_iterees_tendent_vers_cosh_et_sinh() -> None:
    """Exemple 6.1 : la limite est `(cosh t, sinh t)`."""
    derniere = iterees_vectorielles(ECHANGE, sp.Matrix([1, 0]), nombre=8)[-1]
    attendue = sp.Matrix([cosh(T), sinh(T)]).applyfunc(
        lambda e: sp.series(e, T, 0, 8).removeO()
    )
    assert _nulle(derniere - attendue)


def test_les_iterees_d_un_systeme_homogene_depuis_t0() -> None:
    """6.2.2 : `y_3 = (I + (t - t0) A + (t - t0)^2 A^2 / 2) C`."""
    depart = sp.Matrix([3, -2])
    troisieme = iterees_vectorielles(EXEMPLE_6_4, depart, nombre=3, t0=2)[-1]
    ecart = T - 2
    attendue = (
        sp.eye(2) + ecart * EXEMPLE_6_4 + ecart**2 * EXEMPLE_6_4**2 / 2
    ) * depart
    assert _nulle(troisieme - attendue)


def test_les_iterees_d_un_systeme_force() -> None:
    """`T(y) = C + int (A y + x)` : `y_1 = C + x t`, puis un terme en `t^2`."""
    depart, force = sp.Matrix([1, 0]), sp.Matrix([1, 0])
    iterees = iterees_vectorielles(ECHANGE, depart, nombre=2, force=force)
    assert _nulle(iterees[1] - (depart + force * T))
    attendue = depart + (ECHANGE * depart + force) * T + ECHANGE * force * T**2 / 2
    assert _nulle(iterees[2] - attendue)
    assert iterees[2] == sp.Matrix([1 + T, T + T**2 / 2])


def test_la_constante_de_lipschitz_de_l_exemple_6_1() -> None:
    """`||A|| = racine(2)` pour la matrice d'echange."""
    assert norme_de_la_matrice(ECHANGE) == sqrt(2)


def test_la_norme_d_une_matrice_a_coefficients_variables() -> None:
    """Page 108 : la racine de la somme des carres des `sup |p_jk(t)|` sur `[a, b]`."""
    matrice = sp.Matrix([[T, 1], [0, -2 * T]])
    assert norme_de_la_matrice(matrice, sp.Interval(0, 2)) == sqrt(4 + 1 + 16)


def test_l_exponentielle_de_l_exemple_6_2() -> None:
    """`e^(At) (1, 0) = (cosh t, sinh t)`."""
    systeme = SystemeDifferentiel(ECHANGE)
    attendue = sp.Matrix([[cosh(T), sinh(T)], [sinh(T), cosh(T)]])
    assert _egales(systeme.exponentielle(), attendue)
    solution = systeme.par_exponentielle(sp.Matrix([1, 0]))
    assert _egales(solution, sp.Matrix([cosh(T), sinh(T)]))


def test_la_resolvante_de_l_exemple_6_3_1() -> None:
    """`(pI - A)^-1 = [[p, 1], [1, p]] / (p^2 - 1)`, image de `e^(At)`."""
    systeme = SystemeDifferentiel(ECHANGE)
    attendue = sp.Matrix([[P, 1], [1, P]]) / (P**2 - 1)
    assert _nulle(systeme.resolvante() - attendue)
    colonnes = [systeme.par_laplace(sp.eye(2)[:, k]) for k in range(2)]
    par_laplace = sp.Matrix.hstack(*colonnes)
    assert _egales(par_laplace, systeme.exponentielle())


def test_la_resolvante_de_l_exemple_6_3_2() -> None:
    """`y1' = y1 + y2 + x1`, `y2' = y1 - y2 + x2` : `det(pI - A) = p^2 - 2`."""
    systeme = SystemeDifferentiel(sp.Matrix([[1, 1], [1, -1]]))
    attendue = sp.Matrix([[P + 1, 1], [1, P - 1]]) / (P**2 - 2)
    assert _nulle(systeme.resolvante() - attendue)


def test_la_matrice_b_de_l_exemple_6_3_2() -> None:
    """`B = e^(At)` : chaque colonne repond a une impulsion sur une entree."""
    systeme = SystemeDifferentiel(sp.Matrix([[1, 1], [1, -1]]))
    c, s = cosh(sqrt(2) * T), sinh(sqrt(2) * T) / sqrt(2)
    attendue = sp.Matrix([[c + s, s], [s, c - s]])
    assert _egales(systeme.exponentielle(), attendue)


def test_la_reponse_forcee_par_les_deux_chemins() -> None:
    """(6.3) : `y = e^(At) C + e^(At) * x`, par convolution et par Laplace."""
    systeme = SystemeDifferentiel(sp.Matrix([[1, 1], [1, -1]]))
    repos, force = sp.Matrix([0, 0]), sp.Matrix([1, T])
    par_convolution = systeme.par_exponentielle(repos, force)
    par_laplace = systeme.par_laplace(repos, force)
    assert _egales(par_convolution, par_laplace)
    residu = sp.diff(par_convolution, T) - systeme.matrice * par_convolution - force
    assert _egales(residu, sp.zeros(2, 1))


def test_l_exponentielle_de_l_exemple_6_4() -> None:
    """`A = [[1, -1], [-4, 1]]` : les colonnes `g_1` et `g_2` de la page 114."""
    systeme = SystemeDifferentiel(EXEMPLE_6_4)
    plus, moins = exp(3 * T) + exp(-T), -exp(3 * T) + exp(-T)
    attendue = sp.Matrix([[plus / 2, moins / 4], [moins, plus / 2]])
    assert _nulle(systeme.exponentielle() - attendue)
    wronskien = Wronskien(attendue)
    assert wronskien.resout(EXEMPLE_6_4)
    assert attendue.subs(T, 0) == sp.eye(2)


def test_la_formule_de_la_trace_sur_l_exemple_6_4() -> None:
    """Propriete 6.3.5 : `det W(t) = det W(0) e^(int trace)` = `e^(2t)`."""
    wronskien = Wronskien(SystemeDifferentiel(EXEMPLE_6_4).exponentielle())
    assert sp.simplify(wronskien.determinant() - exp(2 * T)) == 0
    assert determinant_par_la_trace(EXEMPLE_6_4, t0=0, initial=1) == exp(2 * T)


def test_des_solutions_dependantes_ont_un_determinant_nul_partout() -> None:
    """Propriete 6.3.4 : il suffit qu'il s'annule en un point."""
    premiere = SystemeDifferentiel(EXEMPLE_6_4).exponentielle()[:, 0]
    wronskien = Wronskien(sp.Matrix.hstack(premiere, 2 * premiere))
    assert wronskien.resout(EXEMPLE_6_4)
    assert not wronskien.est_fondamental()
    assert Wronskien(SystemeDifferentiel(EXEMPLE_6_4).exponentielle()).est_fondamental()


def test_le_wronskien_de_l_exemple_6_5() -> None:
    """`(D - 1)(D - 2) y = 0` : `W = [[e^t, e^(2t)], [e^t, 2e^(2t)]]`."""
    operateur = OperateurDifferentiel((D - 1) * (D - 2))
    wronskien = Wronskien.des_solutions(operateur.base_des_solutions())
    assert wronskien.matrice == sp.Matrix(
        [[exp(T), exp(2 * T)], [exp(T), 2 * exp(2 * T)]]
    )
    assert sp.simplify(wronskien.determinant() - exp(3 * T)) == 0
    compagnon = matrice_compagnon((2, -3, 1))
    assert compagnon == sp.Matrix([[0, 1], [-2, 3]])
    assert wronskien.resout(compagnon)
    assert determinant_par_la_trace(compagnon, t0=0, initial=1) == exp(3 * T)


def test_la_forme_normale_de_l_exemple_6_6() -> None:
    """`t^3 y''' - 6t y' + 12 y = x` : la derniere ligne porte `-a_k / a_n`."""
    attendue = sp.Matrix([[0, 1, 0], [0, 0, 1], [-12 / T**3, 6 / T**2, 0]])
    assert attendue == COMPAGNON_6_6
    assert force_normale(T**4, T**3, 3) == sp.Matrix([0, 0, T])


def test_le_wronskien_de_l_exemple_6_6() -> None:
    """`t^2`, `t^3`, `t^-2` : `det W = 20`, constant -- la trace de `P(t)` est nulle."""
    wronskien = Wronskien.des_solutions(SOLUTIONS_6_6)
    attendue = sp.Matrix(
        [
            [T**2, T**3, T**-2],
            [2 * T, 3 * T**2, -2 * T**-3],
            [2, 6 * T, 6 * T**-4],
        ]
    )
    assert wronskien.matrice == attendue
    assert wronskien.resout(COMPAGNON_6_6)
    assert wronskien.determinant() == 20
    assert COMPAGNON_6_6.trace() == 0
    assert determinant_par_la_trace(COMPAGNON_6_6, t0=1, initial=20) == 20


def test_les_cofacteurs_de_l_exemple_6_6() -> None:
    """`W_31 = -5`, `W_32 = 4/t`, `W_33 = t^4`."""
    matrice = Wronskien.des_solutions(SOLUTIONS_6_6).matrice
    cofacteurs = [sp.simplify(matrice.cofactor(2, k)) for k in range(3)]
    assert cofacteurs == [-5, 4 / T, T**4]


def test_la_solution_particuliere_laissee_en_exercice() -> None:
    """Exemple 6.6 avec `x = t^4` : la formule (6.15) rend `t^4 / 12`."""
    tau = sp.Symbol("tau", positive=True)
    particuliere = particuliere_par_le_wronskien(
        SOLUTIONS_6_6, dominant=T**3, force=T**4, t0=tau
    )
    residu = T**3 * sp.diff(particuliere, T, 3) - 6 * T * sp.diff(particuliere, T)
    assert sp.simplify(residu + 12 * particuliere - T**4) == 0
    homogene_pres = sp.simplify(particuliere - T**4 / 12)
    libre = T**3 * sp.diff(homogene_pres, T, 3) - 6 * T * sp.diff(homogene_pres, T)
    assert sp.simplify(libre + 12 * homogene_pres) == 0


def test_la_solution_de_cauchy_par_le_wronskien() -> None:
    """Page 110 : `y(t) = W(t) W^-1(t0) y0`, ici depuis `t0 = 1`."""
    wronskien = Wronskien.des_solutions(SOLUTIONS_6_6)
    depart = sp.Matrix([1, 0, 2])
    solution = wronskien.solution_de_cauchy(1, depart)
    assert solution.subs(T, 1) == depart
    assert _nulle(sp.diff(solution, T) - COMPAGNON_6_6 * solution)
    assert sp.simplify(solution[1] - sp.diff(solution[0], T)) == 0


def test_les_proprietes_du_noyau() -> None:
    """Page 112 : `R(t, t) = I`, composition, `R^-1(t1, t2) = R(t2, t1)`."""
    wronskien = Wronskien.des_solutions(SOLUTIONS_6_6)
    t1, t2, t3 = sp.symbols("t1 t2 t3", positive=True)
    assert _nulle(wronskien.noyau(t1, t1) - sp.eye(3))
    assert _nulle(
        wronskien.noyau(t1, t2) * wronskien.noyau(t2, t3) - wronskien.noyau(t1, t3)
    )
    assert _nulle(wronskien.noyau(t1, t2).inv() - wronskien.noyau(t2, t1))


def test_le_noyau_d_un_systeme_a_coefficients_constants() -> None:
    """6.3.4 : `W = e^(At)` donne `R(t1, t2) = e^(A (t1 - t2))`."""
    exponentielle = SystemeDifferentiel(EXEMPLE_6_4).exponentielle()
    t1, t2 = sp.symbols("t1 t2", positive=True)
    noyau = Wronskien(exponentielle).noyau(t1, t2)
    assert _nulle(noyau - exponentielle.subs(T, t1 - t2))


def test_la_solution_forcee_par_le_noyau() -> None:
    """Page 112 : `y = R(t, t0) K0 + int R(t, tau) x(tau) dtau`."""
    wronskien = Wronskien(SystemeDifferentiel(EXEMPLE_6_4).exponentielle())
    depart, force = sp.Matrix([1, 2]), sp.Matrix([T, 1])
    solution = wronskien.solution_de_cauchy(0, depart, force)
    assert solution.subs(T, 0) == depart
    assert _nulle(sp.diff(solution, T) - EXEMPLE_6_4 * solution - force)


def test_la_fonction_de_green_est_le_coin_de_l_exponentielle() -> None:
    """Remarque 6.6 : `g = g_n / a_n`, ou `g_n` est la derniere colonne de `e^(At)`."""
    for coefficients in ((2, -3, 1), (4, 6, 2), (0, 2, 3, 1)):
        ordre = len(coefficients) - 1
        polynome = sum(a * D**k for k, a in enumerate(coefficients))
        exponentielle = SystemeDifferentiel(
            matrice_compagnon(coefficients)
        ).exponentielle()
        coin = exponentielle[0, ordre - 1] / coefficients[-1]
        green = fonction_de_green(OperateurDifferentiel(polynome))
        assert sp.simplify(coin - green) == 0


def test_un_coefficient_dominant_nul_n_a_pas_de_forme_normale() -> None:
    """(6.11) : « avec `a_n(t) != 0` »."""
    with pytest.raises(ValueError, match="coefficient dominant nul"):
        matrice_compagnon((1, 2, 0))


def test_les_points_critiques_du_modele_predateur_proie() -> None:
    """Page 115 : `(0, 0)` et `(c/gamma, a/alpha)`."""
    a, alpha, c, gamma = sp.symbols("a alpha c gamma", positive=True)
    modele = PredateurProie(a, alpha, c, gamma)
    assert set(modele.points_critiques()) == {(0, 0), (c / gamma, a / alpha)}


def test_l_origine_est_un_point_de_selle() -> None:
    """La matrice y vaut `diag(a, -c)` : valeurs propres de signes opposes."""
    a, alpha, c, gamma = sp.symbols("a alpha c gamma", positive=True)
    modele = PredateurProie(a, alpha, c, gamma)
    assert modele.linearise((0, 0)) == sp.Matrix([[a, 0], [0, -c]])
    assert nature_du_point_critique(modele.linearise((0, 0))) is Nature.SELLE
    assert not Nature.SELLE.est_stable


def test_l_equilibre_de_coexistence_est_un_centre() -> None:
    """Matrice anti-diagonale, valeurs propres `+- i racine(ac)`."""
    a, alpha, c, gamma = sp.symbols("a alpha c gamma", positive=True)
    modele = PredateurProie(a, alpha, c, gamma)
    matrice = modele.linearise((c / gamma, a / alpha))
    assert matrice == sp.Matrix([[0, -alpha * c / gamma], [a * gamma / alpha, 0]])
    assert set(matrice.eigenvals()) == {sp.I * sqrt(a * c), -sp.I * sqrt(a * c)}
    assert nature_du_point_critique(matrice) is Nature.CENTRE
    assert Nature.CENTRE.est_stable


def test_sans_predateur_les_proies_croissent_en_e_puissance_a_t() -> None:
    """Le syllabus ecrit « selon `e^(ax)` » et « `e^(-cy)` » : c'est en `t`."""
    a, alpha, c, gamma = sp.symbols("a alpha c gamma", positive=True)
    proies, predateurs = PredateurProie(a, alpha, c, gamma).champ()
    x, y = PredateurProie.POPULATIONS
    assert proies.subs(y, 0) == a * x
    assert predateurs.subs(x, 0) == -c * y
    assert sp.diff(exp(a * T), T) == a * exp(a * T)


def test_les_populations_isolees_sont_exponentielles_en_t_par_la_bibliotheque() -> None:
    """La lib resout `x' = a x` et `y' = -c y` : `C e^(a t)`, `C e^(-c t)`."""
    a, alpha, c, gamma = sp.symbols("a alpha c gamma", positive=True)
    modele = PredateurProie(a, alpha, c, gamma)
    proies, predateurs = modele.evolution_isolee()
    assert proies == CONSTANTE * exp(a * T)
    assert predateurs == CONSTANTE * exp(-c * T)
    x, y = PredateurProie.POPULATIONS
    assert not proies.has(x, y)
    assert not predateurs.has(x, y)
    assert proies != CONSTANTE * exp(a * x)
    assert predateurs != CONSTANTE * exp(-c * y)


@pytest.mark.parametrize(
    ("matrice", "nature"),
    [
        (sp.Matrix([[-1, 0], [0, -2]]), Nature.NOEUD_STABLE),
        (sp.Matrix([[1, 0], [0, 2]]), Nature.NOEUD_INSTABLE),
        (sp.Matrix([[-1, -2], [2, -1]]), Nature.FOYER_STABLE),
        (sp.Matrix([[1, -2], [2, 1]]), Nature.FOYER_INSTABLE),
        (EXEMPLE_6_4, Nature.SELLE),
        (sp.Matrix([[0, 1], [-1, 0]]), Nature.CENTRE),
    ],
)
def test_la_nature_d_un_point_critique_se_lit_sur_les_valeurs_propres(
    matrice: sp.Matrix, nature: Nature
) -> None:
    assert nature_du_point_critique(matrice) is nature
    assert nature.est_stable == (
        nature in {Nature.NOEUD_STABLE, Nature.FOYER_STABLE, Nature.CENTRE}
    )


def test_une_valeur_propre_nulle_ne_se_classe_pas() -> None:
    with pytest.raises(ValueError, match="valeur propre nulle"):
        nature_du_point_critique(sp.Matrix([[0, 0], [0, -1]]))


def test_la_norme_porte_sur_les_carres_des_coefficients() -> None:
    """`[[2, 0], [0, -3]]` : `racine(4 + 9)`."""
    assert norme_de_la_matrice(sp.Matrix([[2, 0], [0, -3]])) == sqrt(13)


def test_la_solution_forcee_depuis_un_autre_instant() -> None:
    """Le meme probleme pose en `t0 = 1` : l'integrale part de `t0`, non de 0."""
    wronskien = Wronskien(SystemeDifferentiel(EXEMPLE_6_4).exponentielle())
    depart, force = sp.Matrix([1, 2]), sp.Matrix([T, 1])
    solution = wronskien.solution_de_cauchy(1, depart, force)
    assert _nulle(solution.subs(T, 1) - depart)
    assert _nulle(sp.diff(solution, T) - EXEMPLE_6_4 * solution - force)


def test_la_formule_de_la_trace_depuis_un_autre_instant() -> None:
    """`det W(1) = e^2` redonne `e^(2t)` : l'integrale de la trace part de `t0`."""
    depuis_un = determinant_par_la_trace(EXEMPLE_6_4, t0=1, initial=exp(2))
    assert sp.simplify(depuis_un - exp(2 * T)) == 0
