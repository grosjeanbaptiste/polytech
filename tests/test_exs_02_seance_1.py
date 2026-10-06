"""Mathematique, exercices 2, seance I : equations differentielles.

Source : `Mathematique/EXS/Exs_02_Equations_differentielles.pdf`, pages 4
a 9. La feuille donne une reponse par exercice ; chaque test la recopie
et la controle par un chemin qui ne la suppose pas : une solution se
*reporte* dans son equation, et le residu doit etre nul.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.application import equations_exs_02 as eq
from polytech.maths.application import modeles_exs_02 as mo
from polytech.maths.application.resolue_exs_02 import K, Resolue, X, Y
from polytech.maths.domain.differentielles.autonomes import Stabilite
from polytech.maths.domain.symboles import TEMPS as T

OFFICIELLES = {
    1: sp.sqrt(1 + K * sp.exp(X**2)),
    2: sp.Eq(Y * sp.sqrt(Y**2 + 2 * X**2), K),
    3: K * sp.sqrt(1 + sp.exp(2 * X)),
    4: sp.Eq(sp.sinh(Y / X) / X ** sp.Rational(2, 3), K),
    5: sp.sin(X) - 1 + K * sp.exp(-sp.sin(X)),
    6: (K + X) * sp.cos(X),
    7: K * (X**2 - X) + X,
}
"""Page 4, « Reponses » ; les formes implicites sont ecrites `g(x, y) = K`."""

TYPES = {
    1: "variables separees",
    2: "homogene de degre 0",
    3: "variables separees",
    4: "homogene de degre 0",
    5: "lineaire du premier ordre",
    6: "lineaire du premier ordre",
    7: "lineaire du premier ordre",
}


@pytest.mark.parametrize("numero", sorted(OFFICIELLES))
def test_i1_la_solution_verifie_son_equation(numero: int) -> None:
    assert eq.EQUATIONS[numero]().residu() == 0


@pytest.mark.parametrize("numero", sorted(OFFICIELLES))
def test_i1_la_reponse_de_la_feuille_verifie_l_equation(numero: int) -> None:
    resolue = eq.EQUATIONS[numero]()
    officielle = Resolue(resolue.pente, "", (), OFFICIELLES[numero])
    assert officielle.residu() == 0


@pytest.mark.parametrize("numero", sorted(OFFICIELLES))
def test_i1_la_solution_est_celle_de_la_feuille(numero: int) -> None:
    resolue = eq.EQUATIONS[numero]()
    assert resolue.type == TYPES[numero]
    trouvee, attendue = resolue.solution, OFFICIELLES[numero]
    if isinstance(attendue, sp.Equality):
        assert isinstance(trouvee, sp.Equality)
        assert sp.simplify(trouvee.lhs - attendue.lhs) == 0
    else:
        assert sp.simplify(trouvee - attendue) == 0


def test_un_residu_non_nul_trahit_une_fausse_solution() -> None:
    fausse = Resolue(eq.EQUATIONS[6]().pente, "", (), (K - X) * sp.cos(X))
    assert fausse.residu() != 0
    implicite = Resolue(eq.EQUATIONS[2]().pente, "", (), sp.Eq(X * Y, K))
    assert implicite.residu() != 0


def test_i1_1_les_solutions_constantes_sont_dans_la_famille() -> None:
    """`y = 1` et `y = -1` annulent `y^2 - 1` : c'est `K = 0`."""
    assert eq.equation_1().solution.subs(K, 0) == 1
    assert eq.constantes_de_la_1() == (-1, 1)


def test_i1_2_les_etapes_du_changement_z() -> None:
    """`z = y/x` : `f(z) = -z/(1 + z^2)`, puis `x z' = f(z) - z`."""
    z = eq.Z
    en_z, x_z_prime, primitive = eq.equation_2().etapes[:3]
    assert sp.simplify(en_z + z / (1 + z**2)) == 0
    assert sp.simplify(x_z_prime + (z**3 + 2 * z) / (1 + z**2)) == 0
    assert sp.simplify(primitive.diff(z) - 1 / x_z_prime) == 0


def test_i1_2_le_facteur_integrant_donne_la_meme_famille() -> None:
    """Second chemin : multipliee par `y`, l'equation devient exacte."""
    facteur, potentiel = eq.second_chemin_de_la_2()
    assert facteur == Y
    officielle = OFFICIELLES[2].lhs
    assert sp.simplify(potentiel - officielle**2 / 4) == 0


def test_i1_4_les_etapes_du_changement_z() -> None:
    z = eq.Z
    en_z, x_z_prime, primitive = eq.equation_4().etapes[:3]
    assert sp.simplify(en_z - z - 2 * sp.tanh(z) / 3) == 0
    assert sp.simplify(x_z_prime - 2 * sp.tanh(z) / 3) == 0
    assert sp.simplify(primitive.diff(z) - 1 / x_z_prime) == 0


def test_i1_7_la_variation_de_la_constante() -> None:
    """`y_H = K x (x - 1)`, puis `K(x) = 1/(x - 1)`, donc `y_p = x`."""
    homogene, constante, particuliere = eq.equation_7().etapes
    assert sp.simplify(homogene - K * X * (X - 1)) == 0
    assert sp.simplify(constante - 1 / (X - 1)) == 0
    assert sp.simplify(particuliere - X) == 0


# --- I.2 : le champ de pentes ---------------------------------------------


def test_i2_les_points_critiques_du_cadre() -> None:
    """`sin(pi y / 4) = 0` pour `y = -4, 0, 4` : la feuille ne cite que 0."""
    assert mo.points_critiques_du_champ() == (-4, 0, 4)
    natures = [mo.champ().nature(y) for y in (-4, 0, 4)]
    assert natures == [Stabilite.STABLE, Stabilite.INSTABLE, Stabilite.STABLE]


def test_i2_l_isocline_de_la_feuille() -> None:
    """Page 5 : `y = (4/pi) arcsin p`."""
    p = sp.Symbol("p")
    principale, seconde = mo.isoclines(p)
    assert principale == 4 * sp.asin(p) / sp.pi
    assert seconde == 4 - 4 * sp.asin(p) / sp.pi
    for branche in (principale, seconde):
        assert sp.simplify(sp.sin(sp.pi * branche / 4) - p) == 0


@pytest.mark.parametrize("depart", [3, 1, -1, -3])
def test_i2_la_solution_explicite(depart: int) -> None:
    """Ce que la feuille esquisse se calcule : l'equation est a variables separees."""
    solution = mo.solution_du_champ(depart)
    assert solution.subs(T, 0) == depart
    residu = solution.diff(T) - sp.sin(sp.pi * solution / 4)
    assert abs(sp.N(residu.subs(T, sp.Rational(7, 10)))) < 1e-12
    assert sp.limit(solution, T, sp.oo) == 4 * sp.sign(depart)


# --- I.3 : la logistique ----------------------------------------------------


def test_i3_les_points_critiques_et_leur_nature() -> None:
    capacite = mo.CAPACITE
    equation = mo.logistique_de_la_feuille()
    assert equation.points_critiques() == (0, capacite)
    assert equation.nature(0) is Stabilite.INSTABLE
    assert equation.nature(capacite) is Stabilite.STABLE


def test_i3_la_solution_de_la_feuille() -> None:
    """Page 6 : `y = K c e^(lambda K t) / (1 + c e^(lambda K t))`, `c = y0/(K - y0)`."""
    capacite, taux, depart = mo.CAPACITE, mo.TAUX, mo.DEPART
    generale, constante, de_cauchy = mo.logistique_resolue()
    c = mo.LIBRE
    croissance = sp.exp(taux * capacite * T)
    assert sp.simplify(generale - capacite * c * croissance / (1 + c * croissance)) == 0
    assert sp.simplify(constante - depart / (capacite - depart)) == 0
    assert sp.simplify(de_cauchy.subs(T, 0) - depart) == 0
    residu = de_cauchy.diff(T) - taux * de_cauchy * (capacite - de_cauchy)
    assert sp.simplify(residu) == 0
    assert sp.limit(de_cauchy, T, sp.oo) == capacite


def test_i3_bernoulli_donne_la_meme_solution() -> None:
    """Le second chemin de la feuille : `z = 1/y` rend l'equation lineaire."""
    en_z, depuis_z = mo.logistique_par_bernoulli()
    capacite, taux = mo.CAPACITE, mo.TAUX
    assert sp.simplify(en_z.p - taux * capacite) == 0
    assert sp.simplify(en_z.q - taux) == 0
    residu = depuis_z.diff(X) - taux * depuis_z * (capacite - depuis_z)
    assert sp.simplify(residu) == 0


def test_i3_l_inflexion_est_a_mi_capacite() -> None:
    """`y'' = 0` quand `y = K/2` : la croissance y est la plus rapide."""
    assert mo.inflexion_de_la_logistique() == mo.CAPACITE / 2


# --- I.4 : equations exactes ------------------------------------------------


def test_i4_la_valeur_du_parametre() -> None:
    assert mo.parametre_d_exactitude() == 1


def test_i4_les_deux_potentiels() -> None:
    """Page 8 : `sin(x + y) + 2 y^2 + x^2 = c` ; et celui de la premiere equation."""
    premier, second = mo.potentiels()
    assert sp.simplify(second - (sp.sin(X + Y) + 2 * Y**2 + X**2)) == 0
    assert sp.simplify(premier - (X**2 + X * Y + sp.exp(X**2) + Y**2)) == 0


# --- I.5 : le ressort -------------------------------------------------------


def test_i5_la_solution_generale_de_la_feuille() -> None:
    """Page 9 : `sin(omega t)/(1 - omega^2)` hors resonance, `-t cos(t)/2` sinon."""
    omega = mo.PULSATION
    hors, en_resonance = mo.reponses_forcees()
    assert sp.simplify(hors - sp.sin(omega * X) / (1 - omega**2)) == 0
    assert sp.simplify(en_resonance + X * sp.cos(X) / 2) == 0
    assert mo.ressort().pulsation_propre() == 1
    assert mo.ressort().est_en_resonance(1)
    assert not mo.ressort().est_en_resonance(2)


def test_i5_la_resonance_est_la_limite_du_cas_general() -> None:
    """Partie du repos, la reponse hors resonance tend vers la reponse resonante."""
    omega = mo.PULSATION
    du_repos, a_la_limite = mo.reponses_depuis_le_repos()
    assert du_repos.subs(X, 0) == 0
    assert du_repos.diff(X).subs(X, 0) == 0
    assert sp.simplify(du_repos.diff(X, 2) + du_repos - sp.sin(omega * X)) == 0
    assert sp.simplify(a_la_limite - (sp.sin(X) - X * sp.cos(X)) / 2) == 0
    assert sp.simplify(sp.limit(du_repos, omega, 1) - a_la_limite) == 0


def test_i1_4_la_primitive_est_un_logarithme_de_sinus_hyperbolique() -> None:
    """`3/(2 tanh z)` s'integre en `(3/2) ln sinh z` : c'est ce que l'on affiche."""
    assert eq.equation_4().etapes[2] == 3 * sp.log(sp.sinh(eq.Z)) / 2
