"""Fiche de signaux : la boucle fermee, sa stabilite, ses erreurs, la fonction M.

Articles du bloc « boucle » de `cursus/fiches/SignauxSystemes.toml`.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.comparaison import egales
from polytech.signaux.domain.boucle_fermee import Boucle
from polytech.signaux.domain.marges import marges_de_stabilite
from polytech.signaux.domain.stabilite import (
    Stabilite,
    critere_simplifie_de_nyquist,
    equation_caracteristique,
    est_stable_par_les_racines,
)
from polytech.signaux.domain.symboles import GAIN as K
from polytech.signaux.domain.symboles import IMAGE as P
from polytech.signaux.domain.transmittance import Transmittance

C, G1, G2, GS = sp.symbols("C G_1 G_2 G_s")
R0, P0 = sp.symbols("R_0 P_0")


def _boucle() -> Boucle:
    return Boucle(correcteur=C, actionneur=G1, procede=G2, capteur=GS)


def test_la_boucle_ouverte_porte_le_capteur_et_la_chaine_directe() -> None:
    """Article « La boucle ouverte » : `L = G_s C G_1 G_2`, `KG/(1+KG)` en unitaire."""
    assert egales(_boucle().ouverte(), GS * C * G1 * G2)
    g = 1 / (P * (P + 1))
    fermee = Transmittance(expression=K * g).en_boucle_fermee().expression
    assert egales(fermee, K * g / (1 + K * g))


def test_les_transmittances_equivalentes_viennent_de_la_resolution_de_la_boucle() -> (
    None
):
    """Article « Les transmittances en boucle fermee », perturbation d'un signe `s`."""
    y, signe = sp.symbols("Y s")
    equation = sp.Eq(y, G2 * (C * G1 * (R0 - GS * y) + signe * P0))
    solution = sp.solve(equation, y)[0]
    boucle = _boucle()
    assert egales(solution.subs({signe: -1}), boucle.sortie(R0, P0))
    assert egales(sp.diff(solution, R0), boucle.vis_a_vis_de_la_consigne())
    assert egales(
        sp.diff(solution.subs(signe, 1), P0), -boucle.vis_a_vis_de_la_perturbation()
    )


def test_la_figure_du_retour_h_donne_g1g2_sur_un_plus_g1g2h() -> None:
    """Figure : `S/E = G1 G2 / (1 + G1 G2 H)`, `S/P = G2 / (1 + G1 G2 H)`."""
    h = sp.Symbol("H")
    boucle = Boucle(correcteur=1, actionneur=G1, procede=G2, capteur=h)
    assert egales(boucle.vis_a_vis_de_la_consigne(), G1 * G2 / (1 + G1 * G2 * h))
    assert egales(boucle.vis_a_vis_de_la_perturbation(), -G2 / (1 + G1 * G2 * h))


def test_l_equation_caracteristique_est_dl_plus_nl_et_ses_racines() -> None:
    """Article « L'equation caracteristique » : `1 + K/((p+2)(p+1/2))`."""
    ouverte = Transmittance(expression=K / ((P + 2) * (P + sp.Rational(1, 2))))
    numerateur, denominateur = ouverte.numerateur(), ouverte.denominateur()
    assert egales(equation_caracteristique(ouverte), numerateur + denominateur)
    racines = sp.solve(equation_caracteristique(ouverte).subs(K, 2), P)
    assert {sp.re(r) for r in racines} == {sp.Rational(-5, 4)}


def test_la_stabilite_en_boucle_fermee_depend_de_k_avec_une_limite() -> None:
    """Septembre 2016 (4) : une valeur limite de `K` met des racines sur l'axe."""
    ouverte = Transmittance(expression=K / (P * (P + 1) * (P + 2)))
    limite = Stabilite(ouverte).gain_critique(K)
    assert limite == 6
    racines = sp.solve(equation_caracteristique(ouverte).subs(K, limite), P)
    assert {sp.re(r) for r in racines if r != -3} == {0}
    assert est_stable_par_les_racines(
        Transmittance(expression=ouverte.expression.subs(K, 5))
    )
    assert not est_stable_par_les_racines(
        Transmittance(expression=ouverte.expression.subs(K, 7))
    )


def test_nyquist_simplifie_s_accorde_avec_les_racines_de_l_equation() -> None:
    """Article « Le critere de Nyquist simplifie » : le croisement decide."""
    ouverte = Transmittance(expression=K / (P * (P + 1) * (P + 2)))
    stabilite = Stabilite(ouverte)
    for gain in (3, 5, 8):
        croisement = float(stabilite.reel_au_croisement({K: sp.Integer(gain)}))
        racines = est_stable_par_les_racines(
            Transmittance(expression=ouverte.expression.subs(K, gain))
        )
        assert critere_simplifie_de_nyquist(croisement) is racines


def test_la_courbe_de_nyquist_de_la_figure_a_pour_gain_un_virgule_six() -> None:
    """Figure : `L = 1,6 / (p(p+1))`, stable en BF, sans croisement a gauche de `-1`."""
    ouverte = sp.Rational(8, 5) / (P * (P + 1))
    reel = sp.re(sp.simplify(ouverte.subs(P, sp.I * sp.Symbol("w", positive=True))))
    assert egales(reel, -sp.Rational(8, 5) / (1 + sp.Symbol("w", positive=True) ** 2))
    assert est_stable_par_les_racines(Transmittance(expression=ouverte))
    assert marges_de_stabilite(ouverte).de_phase > 0
