"""Cinematique, transport, continuite et Navier-Stokes : la fiche, refaite.

Les formules de `cursus/fiches/MecaniqueFluides.toml` pour la derivee
materielle, le theoreme de transport, la conservation de la masse, le
tenseur des deformations, Couette et le film tombant sont recalculees par
`polytech.fluides.domain`.
"""

from __future__ import annotations

import sympy as sp

from polytech.fluides.domain.cinematique.champ import ChampDeVitesse
from polytech.fluides.domain.cinematique.operateurs import (
    derivee_le_long_de_la_trajectoire,
    derivee_materielle,
    divergence,
)
from polytech.fluides.domain.cinematique.symboles import T, X, Y
from polytech.fluides.domain.fondamentales.film_tombant import vitesse_du_film
from polytech.fluides.domain.fondamentales.masse_locale import (
    residu_de_continuite,
    residu_en_derivee_materielle,
)
from polytech.fluides.domain.fondamentales.navier_stokes import (
    HAUTEUR,
    VISCOSITE,
    VITESSE_DE_PAROI,
    cisaillement,
    profil_de_couette,
)
from polytech.fluides.domain.fondamentales.transport_de_reynolds import (
    flux_a_travers,
    variation_du_systeme,
)
from polytech.maths.domain.comparaison import egales, est_negative

MU, RHO, G, H, XP = sp.symbols("mu rho g h x_p", positive=True)
ETIREMENT = ChampDeVitesse(2 * X, -2 * Y)


def test_un_champ_stationnaire_peut_accelerer_par_son_terme_convectif() -> None:
    assert ETIREMENT.est_stationnaire()
    assert egales(sp.Matrix(ETIREMENT.gradient_temporel()), sp.zeros(2, 1))
    assert egales(sp.Matrix(ETIREMENT.acceleration()), sp.Matrix([4 * X, 4 * Y]))


def test_la_derivee_materielle_est_la_derivee_en_chaine_le_long_de_la_particule() -> (
    None
):
    grandeur = X**2 * Y * T + sp.sin(X) * Y
    u, v = Y * T, X - Y
    assert egales(
        derivee_materielle(grandeur, u, v),
        derivee_le_long_de_la_trajectoire(grandeur, u, v),
    )


def test_l_acceleration_d_euler_est_la_derivee_seconde_de_la_trajectoire() -> None:
    x_t, y_t = ETIREMENT.trajectoire(1.5, 0.5)
    ax, ay = ETIREMENT.acceleration()
    en_t = {X: x_t, Y: y_t}
    assert egales(sp.diff(x_t, T, 2), ax.subs(en_t))
    assert egales(sp.diff(y_t, T, 2), ay.subs(en_t))


def test_transport_de_reynolds_la_normale_est_sortante() -> None:
    entree = flux_a_travers(1, 1000, (2, 0, 0), (-1, 0, 0), 0.5)
    sortie = flux_a_travers(1, 1000, (2, 0, 0), (1, 0, 0), 0.5)
    assert entree < 0 < sortie
    assert egales(entree, -sortie)


def test_b_egal_un_donne_la_conservation_de_la_masse_sous_forme_integrale() -> None:
    entrant, sortant = 1000 * 2 * 0.5, 1000 * 0.5 * 4.0
    stockage = sp.Symbol("dM_vc")
    systeme = variation_du_systeme(stockage, [entrant], [sortant])
    (valeur,) = sp.solve(sp.Eq(systeme, 0), stockage)
    assert egales(valeur, entrant - sortant)
    assert egales(variation_du_systeme(0, [entrant], [entrant]), 0)


def test_les_deux_formes_locales_de_la_continuite_sont_equivalentes() -> None:
    rho, vitesse = X * T + 2, (Y, X * Y, sp.Integer(0))
    assert egales(
        residu_de_continuite(rho, vitesse), residu_en_derivee_materielle(rho, vitesse)
    )


def test_un_champ_qui_conserve_la_masse_annule_les_deux_residus() -> None:
    rho, vitesse = 1 / (1 + T), (X / (1 + T), sp.Integer(0), sp.Integer(0))
    assert egales(residu_de_continuite(rho, vitesse), 0)
    assert egales(residu_en_derivee_materielle(rho, vitesse), 0)
    assert egales(residu_de_continuite(RHO, (2 * X, -2 * Y, sp.Integer(0))), 0)


def test_la_trace_des_deformations_est_la_divergence_et_la_dilatation_volumique() -> (
    None
):
    champ = ChampDeVitesse(3 * X, -Y)
    assert egales(
        champ.tenseur_taux_deformation().trace(), divergence(champ.u, champ.v)
    )
    assert egales(sp.diff(sp.log(champ.loi_de_volume()), T), champ.divergence())
    gradient = sp.Matrix([[sp.diff(c, a) for a in (X, Y)] for c in (champ.u, champ.v)])
    assert egales(champ.tenseur_taux_deformation(), (gradient + gradient.T) / 2)


def test_incompressible_n_est_pas_irrotationnel() -> None:
    cisaille = ChampDeVitesse(Y, sp.Integer(0))
    assert cisaille.est_incompressible()
    assert cisaille.est_rotationnel()
    dilate = ChampDeVitesse(X, Y)
    assert not dilate.est_incompressible()
    assert not dilate.est_rotationnel()
    assert ETIREMENT.est_incompressible()
    assert not ETIREMENT.est_rotationnel()


def test_couette_est_lineaire_de_cisaillement_uniforme_mu_u_sur_h() -> None:
    assert egales(profil_de_couette(), VITESSE_DE_PAROI * Y / HAUTEUR)
    assert egales(cisaillement(), VISCOSITE * VITESSE_DE_PAROI / HAUTEUR)


def test_le_film_tombant_descend_et_adhere_sans_cisaillement_en_surface() -> None:
    w = vitesse_du_film(XP, H, RHO, MU, G)
    assert est_negative(w.subs(XP, H / 2))
    assert egales(w.subs(XP, 0), 0)
    assert egales(sp.diff(w, XP).subs(XP, H), 0)
    assert egales(MU * sp.diff(w, XP, 2), RHO * G)
