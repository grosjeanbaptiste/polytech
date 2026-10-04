"""Mathematique, second jeu : les applications des EDO, chacune dans sa discipline.

Oracles (`Mathematique/SLIDES/Slides_02_Equations_differentielles.pdf`) :
la loi de Newton (planche 2) ; le circuit RLC (planche 5) ; le seau troue
et la datation au carbone 14 (planche 17) ; Malthus et les mulots
(planches 41 a 43), `y = 900 + lambda e^(t/2)` ; la logistique (planche
45) ; le prix d'equilibre (planches 46-47).

Le cours de mathematique emprunte ces exemples : la desintegration vit
dans `chemistry`, la vidange dans `fluides`, le circuit dans
`electromagnetisme`. Les tests de lien verifient que chaque discipline
dit la meme chose que la planche.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.chemistry.domain.radioactivite import (
    age_d_un_echantillon,
    constante_de_desintegration,
    demi_vie,
    quantite_restante,
)
from polytech.electromagnetisme.domain.circuits.rlc import (
    equation_du_circuit_rlc,
    equation_integrale_du_circuit_rlc,
)
from polytech.fluides.domain.fondamentales.bernoulli import vitesse_de_torricelli
from polytech.fluides.domain.fondamentales.vidange import (
    coefficient_de_vidange,
    duree_de_vidange,
    hauteur_pendant_la_vidange,
)
from polytech.maths.domain.differentielles.autonomes import (
    EquationAutonome,
    Stabilite,
)
from polytech.maths.domain.differentielles.caracterisation import caracteriser
from polytech.maths.domain.differentielles.cauchy import ProblemeDeCauchy
from polytech.maths.domain.differentielles.modeles import (
    force,
    logistique,
    malthus,
    prix_de_marche,
)
from polytech.maths.domain.differentielles.premier_ordre import INCONNUE, VARIABLE
from polytech.maths.domain.symboles import CONSTANTE as C

X, Y = VARIABLE, INCONNUE
R = sp.Rational
T = sp.Symbol("t", nonnegative=True)


def _verifie(solution: sp.Expr, second_membre: sp.Expr) -> bool:
    reste = sp.diff(solution, X) - second_membre.subs(Y, solution)
    return bool(sp.simplify(reste) == 0)


def test_la_loi_de_newton_derive_la_quantite_de_mouvement() -> None:
    """Planche 2 : `F = d(m v)/dt` ; a masse constante, `F = m a`."""
    m = sp.Symbol("m", positive=True)
    vitesse = 3 * T**2
    assert force(m, vitesse, T) == 6 * m * T
    assert force(2 * T, vitesse, T) == sp.diff(2 * T * vitesse, T) == 18 * T**2


def test_le_modele_de_malthus_de_la_planche_41() -> None:
    """`y' = k y - r` : `y = (r + lambda e^(k t)) / k`."""
    k, r = sp.symbols("k r", positive=True)
    equation = malthus(k, r)
    attendu = (r + C * sp.exp(k * X)) / k
    assert _verifie(attendu, k * Y - r)
    difference = sp.simplify(equation.solution_generale() - attendu.subs(C, C * k))
    assert difference == 0


def test_le_probleme_des_mulots_de_la_planche_43() -> None:
    """`k = 1/2` par mois, `r = 15 x 30 = 450` : `y = 900 + lambda e^(t/2)`."""
    equation = malthus(R(1, 2), 15 * 30)
    attendu = 900 + C * sp.exp(X / 2)
    assert sp.simplify(equation.solution_generale() - attendu) == 0
    assert _verifie(attendu, Y / 2 - 450)


def test_l_effectif_d_equilibre_des_mulots_est_900_et_il_est_instable() -> None:
    """Planche 43 : `y' = 0` donne 900 ; au-dessus il croit, en dessous il decroit."""
    autonome = EquationAutonome(Y / 2 - 450)
    assert autonome.points_critiques() == (900,)
    assert autonome.nature(900) is Stabilite.INSTABLE
    assert autonome.sens_de_variation(1000) == 1
    assert autonome.sens_de_variation(800) == -1


def test_les_trois_departs_de_la_verification_sous_matlab() -> None:
    """Planche 44 : `y(0)` = 800, 900, 1000 -- `lambda` = -100, 0, 100."""
    for depart, coefficient in ((800, -100), (900, 0), (1000, 100)):
        solution = ProblemeDeCauchy(Y / 2 - 450, x0=0, y0=depart).solution()
        assert sp.simplify(solution - (900 + coefficient * sp.exp(X / 2))) == 0


def test_l_equation_logistique_de_la_planche_45() -> None:
    """`y' = lambda y (K - y)` : positive entre 0 et `K`, negative au-dela."""
    autonome = logistique(R(1, 10), 50)
    assert autonome.points_critiques() == (0, 50)
    assert autonome.sens_de_variation(20) == 1
    assert autonome.sens_de_variation(70) == -1
    assert autonome.nature(50) is Stabilite.STABLE
    assert autonome.nature(0) is Stabilite.INSTABLE


def test_le_prix_d_equilibre_des_planches_46_et_47() -> None:
    """`P' = k (alpha + gamma) - k (beta + delta) P` et son prix d'equilibre."""
    alpha, beta, gamma, delta, k = sp.symbols("alpha beta gamma delta k", positive=True)
    modele = prix_de_marche(alpha, beta, gamma, delta, k)
    equilibre = (alpha + gamma) / (beta + delta)
    assert sp.simplify(modele.prix_d_equilibre() - equilibre) == 0
    attendu = equilibre + C * sp.exp(-k * (beta + delta) * X)
    assert sp.simplify(modele.equation().solution_generale() - attendu) == 0


def test_au_prix_d_equilibre_l_offre_egale_la_demande() -> None:
    """Planche 47 : « intersection des droites `Q_O(P)` et `Q_D(P)` »."""
    modele = prix_de_marche(2, 3, 10, 1, R(1, 2))
    prix = modele.prix_d_equilibre()
    assert prix == 3
    assert modele.offre(prix) == modele.demande(prix) == 7
    assert sp.limit(modele.equation().solution_generale(), X, sp.oo) == 3


def test_la_desintegration_est_la_loi_de_la_planche_17() -> None:
    """`x' = -beta x` : `x(t) = x(t0) exp(-beta (t - t0))`, lu dans `chemistry`."""
    beta, x0 = sp.symbols("beta x_0", positive=True)
    restante = quantite_restante(x0, beta, X)
    assert restante == x0 * sp.exp(-beta * X)
    assert ProblemeDeCauchy(-beta * Y, x0=0, y0=x0).est_solution(restante)


def test_la_demi_vie_est_la_duree_qui_divise_la_quantite_par_deux() -> None:
    beta = sp.Symbol("beta", positive=True)
    periode = demi_vie(beta)
    assert periode == sp.log(2) / beta
    assert sp.simplify(quantite_restante(1, beta, periode) - R(1, 2)) == 0
    assert sp.simplify(constante_de_desintegration(periode) - beta) == 0


def test_l_age_se_lit_sur_la_fraction_restante() -> None:
    """Un quart du carbone 14 restant : deux periodes se sont ecoulees."""
    periode = sp.Symbol("T", positive=True)
    assert sp.simplify(age_d_un_echantillon(R(1, 4), periode) - 2 * periode) == 0
    assert age_d_un_echantillon(1, periode) == 0


def test_le_seau_troue_obeit_a_la_loi_de_torricelli() -> None:
    """Planche 17 : `h' = gamma sqrt(h)`, avec `gamma = -(s/S) sqrt(2 g)` -- negatif.

    La vitesse de sortie est celle de Torricelli, `sqrt(2 g h)` ; le debit
    qui sort par l'orifice `s` fait baisser la surface `S`.
    """
    s, grande, h = sp.symbols("s S h", positive=True)
    gamma = coefficient_de_vidange(s, grande)
    assert gamma.is_negative
    assert sp.simplify(gamma * sp.sqrt(h) + s / grande * vitesse_de_torricelli(h)) == 0


def test_la_hauteur_du_seau_est_la_solution_de_la_planche_17() -> None:
    """`h = gamma^2/4 (t - alpha)^2` jusqu'a `alpha`, la duree de vidange, puis 0."""
    s, grande, h0 = sp.symbols("s S h_0", positive=True)
    gamma = coefficient_de_vidange(s, grande)
    alpha = duree_de_vidange(h0, s, grande)
    hauteur = hauteur_pendant_la_vidange(h0, s, grande, X)
    avant = hauteur.subs(X, alpha / 3)
    assert sp.simplify(avant - gamma**2 / 4 * (alpha / 3 - alpha) ** 2) == 0
    assert sp.simplify(hauteur.subs(X, 0) - h0) == 0
    assert sp.simplify(hauteur.subs(X, alpha)) == 0
    assert hauteur.subs(X, 2 * alpha) == 0


def test_le_seau_vide_reste_vide() -> None:
    """`h_0 = 16`, `s/S = 2/sqrt(2 g)` donc `gamma = -2` : vide en 4 unites de temps."""
    assert duree_de_vidange(16, 2, sp.sqrt(R(981, 50))) == 4
    probleme = ProblemeDeCauchy(-2 * sp.sqrt(Y), x0=0, y0=16)
    assert probleme.est_solution(sp.Piecewise(((X - 4) ** 2, X <= 4), (0, True)))


def test_le_circuit_rlc_de_la_planche_5() -> None:
    """`u = L di/dt + R i + (1/C) int i dt`, puis `du/dt = L i'' + R i' + i/C`."""
    resistance, inductance, capacite = sp.symbols("R L C", positive=True)
    t = sp.Symbol("t", nonnegative=True)
    i, u = sp.Function("i")(t), sp.Function("u")(t)
    integrale = equation_integrale_du_circuit_rlc(resistance, inductance, capacite)
    derivee = equation_du_circuit_rlc(resistance, inductance, capacite)
    assert derivee == sp.Eq(
        u.diff(t), inductance * i.diff(t, 2) + resistance * i.diff(t) + i / capacite
    )
    assert sp.simplify(integrale.rhs.diff(t) - derivee.rhs) == 0


def test_l_equation_du_circuit_rlc_est_lineaire_du_second_ordre() -> None:
    """Planche 9 sur la planche 5 : ordre 2, lineaire, coefficients constants."""
    t = sp.Symbol("t", nonnegative=True)
    equation = equation_du_circuit_rlc(2, 3, sp.Rational(1, 5))
    en_x = equation.subs(sp.Function("u")(t).diff(t), 0).subs(t, X)
    en_y = en_x.subs(sp.Function("i")(X), sp.Function("y")(X))
    caracterisation = caracteriser(en_y)
    assert (caracterisation.ordre, caracterisation.lineaire) == (2, True)
    assert caracterisation.coefficients_constants


def test_une_fraction_restante_hors_de_son_domaine_est_refusee() -> None:
    """Il ne peut rester ni plus que ce qu'il y avait, ni rien du tout."""
    for fraction in (0, R(3, 2), -1):
        with pytest.raises(ValueError, match="fraction restante hors de"):
            age_d_un_echantillon(fraction, 5568)
