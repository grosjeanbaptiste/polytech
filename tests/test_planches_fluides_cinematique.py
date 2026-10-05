"""Planches de mecanique des fluides, chapitre III : cinematique.

Source : `MecaniqueFluides/SLIDES/Slides_02_Statique_Cinematique_2025_2026.pdf`,
planches 21 a 60, lues sur les pages rendues.

Les oracles du cours : l'exemple lagrangien `x = xi + xi t^2` (planche 25),
le champ `(0,5 + 0,8 x ; 1,5 - 0,8 y)` et ses lignes de courant (planche
28), les deux tourbillons de la planche 40, et les deux exercices de la
derniere planche. Le theoreme de transport est recoupe par la regle de
Leibniz, qu'il ne fait que traduire.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.fluides.domain.cinematique.champ import ChampDeVitesse
from polytech.fluides.domain.cinematique.deplacement import (
    contributions_au_deplacement,
)
from polytech.fluides.domain.cinematique.descriptions import (
    acceleration_lagrangienne,
    etiquette,
    vitesse_eulerienne,
    vitesse_lagrangienne,
)
from polytech.fluides.domain.cinematique.espace import (
    dilatation_volumique,
    taux_de_rotation_dans_l_espace,
    tenseur_des_deformations,
    vorticite_dans_l_espace,
)
from polytech.fluides.domain.cinematique.ligne_de_temps import ligne_de_temps
from polytech.fluides.domain.cinematique.lignes import (
    fonction_de_courant,
    trajectoire,
)
from polytech.fluides.domain.cinematique.operateurs import (
    derivee_materielle,
    taux_de_rotation,
    tenseur_taux_deformation,
    vorticite,
)
from polytech.fluides.domain.cinematique.symboles import T, X, Y, Z
from polytech.fluides.domain.fondamentales.transport_de_reynolds import (
    debit_de_grandeur,
    flux_a_travers,
    variation_du_systeme,
)

XI = sp.Symbol("xi", real=True)
POSITION = XI + XI * T**2
"""Planche 25 : la position de la particule `xi` a l'instant `t`."""


def test_la_vitesse_et_l_acceleration_lagrangiennes_de_la_planche_25() -> None:
    """`v_L = 2 xi t` et `a_L = 2 xi`."""
    assert vitesse_lagrangienne(POSITION) == 2 * XI * T
    assert acceleration_lagrangienne(POSITION) == 2 * XI


def test_la_vitesse_eulerienne_que_la_planche_25_demande() -> None:
    """« v_E ? » : `xi = x / (1 + t^2)`, donc `v_E = 2 x t / (1 + t^2)`."""
    assert sp.simplify(etiquette(POSITION, XI) - X / (1 + T**2)) == 0
    eulerienne = vitesse_eulerienne(POSITION, XI)
    assert sp.simplify(eulerienne - 2 * X * T / (1 + T**2)) == 0
    # Planche 24 : `v = v_L(xi(x, t), t) = v_E(x(xi, t), t)`.
    retour = eulerienne.subs(X, POSITION)
    assert sp.simplify(retour - vitesse_lagrangienne(POSITION)) == 0


def test_la_derivee_materielle_redonne_l_acceleration_lagrangienne() -> None:
    """Planche 26 : `D/Dt = d/dt + V . grad`, appliquee a `v_E`."""
    eulerienne = vitesse_eulerienne(POSITION, XI)
    acceleration = derivee_materielle(eulerienne, eulerienne, sp.Integer(0))
    en_xi = acceleration.subs(X, POSITION)
    assert sp.simplify(en_xi - acceleration_lagrangienne(POSITION)) == 0


def test_un_mouvement_qui_replie_les_particules_n_a_pas_d_etiquette() -> None:
    with pytest.raises(ValueError, match="pas une etiquette unique"):
        etiquette(XI**2 * T, XI)


PLANCHE_28 = ChampDeVitesse(
    sp.Rational(1, 2) + sp.Rational(4, 5) * X,
    sp.Rational(3, 2) - sp.Rational(4, 5) * Y,
)
"""`V = (0,5 + 0,8 x) i + (1,5 - 0,8 y) j`."""


def test_l_acceleration_du_champ_de_la_planche_28() -> None:
    """`a_x = 0,4 + 0,64 x` et `a_y = -1,2 + 0,64 y`, en m/s^2."""
    a_x, a_y = PLANCHE_28.acceleration()
    assert sp.simplify(a_x - (sp.Rational(2, 5) + sp.Rational(16, 25) * X)) == 0
    assert sp.simplify(a_y - (-sp.Rational(6, 5) + sp.Rational(16, 25) * Y)) == 0
    assert PLANCHE_28.est_stationnaire()


def test_les_lignes_de_courant_de_la_planche_28() -> None:
    """`y = C / (0,8 (0,5 + 0,8 x)) + 1,875`, soit `(0,5 + 0,8 x)(y - 1,875) = C`."""
    psi = fonction_de_courant(PLANCHE_28.u, PLANCHE_28.v)
    invariant = (sp.Rational(1, 2) + sp.Rational(4, 5) * X) * (Y - sp.Rational(15, 8))
    assert sp.simplify(sp.diff(psi - invariant, X)) == 0
    assert sp.simplify(sp.diff(psi - invariant, Y)) == 0
    # Planche 27 : `dx / u = dy / v`.
    pente = sp.simplify(
        PLANCHE_28.pente_lignes_de_courant() - PLANCHE_28.v / PLANCHE_28.u
    )
    assert pente == 0


def test_en_stationnaire_la_trajectoire_suit_la_ligne_de_courant() -> None:
    """Resume : trajectoire et ligne de courant se confondent alors."""
    abscisse, ordonnee = trajectoire(
        PLANCHE_28.u, PLANCHE_28.v, sp.Integer(1), sp.Integer(3)
    )
    invariant = (sp.Rational(1, 2) + sp.Rational(4, 5) * abscisse) * (
        ordonnee - sp.Rational(15, 8)
    )
    assert sp.simplify(sp.diff(invariant, T)) == 0


def test_la_ligne_de_temps_dessine_le_profil_de_vitesse() -> None:
    """Planche 31 : des particules marquees au meme instant, sur une droite.

    Dans un ecoulement parallele `u(y)`, elles sont en `x = u(y) t` : la
    ligne de temps prend la forme du profil.
    """
    parametre = sp.Symbol("s", real=True)
    profil = 1 - Y**2
    abscisse, ordonnee = ligne_de_temps(
        profil, sp.Integer(0), (sp.Integer(0), parametre)
    )
    assert sp.simplify(abscisse - (1 - parametre**2) * T) == 0
    assert ordonnee == parametre


def test_les_quatre_contributions_au_deplacement_d_un_coin() -> None:
    """Planche 33 : translation, dilatation, cisaillement, rotation.

    Leur somme est le deplacement du coin `D`, soit sa vitesse fois `dt` ;
    pour un champ lineaire, le developpement au premier ordre est exact.
    """
    u, v = 2 * X - Y + 3, X + 3 * Y - 1
    dx, dy, dt = sp.symbols("dx dy dt", positive=True)
    parts = contributions_au_deplacement(u, v, (dx, dy), dt)
    assert set(parts) == {"translation", "dilatation", "cisaillement", "rotation"}
    total_x = sum(part[0] for part in parts.values())
    total_y = sum(part[1] for part in parts.values())
    en_d = {X: X + dx, Y: Y + dy}
    assert sp.expand(total_x - u.subs(en_d, simultaneous=True) * dt) == 0
    assert sp.expand(total_y - v.subs(en_d, simultaneous=True) * dt) == 0
    assert parts["translation"] == (u * dt, v * dt)
    assert parts["dilatation"] == (2 * dx * dt, 3 * dy * dt)


def test_un_champ_uniforme_ne_fait_que_translater() -> None:
    """Planche 34 : toutes les particules ont au meme instant la meme vitesse."""
    dx, dy, dt = sp.symbols("dx dy dt", positive=True)
    parts = contributions_au_deplacement(sp.Integer(2), sp.Integer(-1), (dx, dy), dt)
    assert parts["translation"] == (2 * dt, -dt)
    for nom in ("dilatation", "cisaillement", "rotation"):
        assert parts[nom] == (0, 0)


def test_un_cisaillement_simple_deforme_et_fait_tourner_a_parts_egales() -> None:
    """Planche 35 : un gradient `du/dy` seul cisaille **et** fait tourner.

    Pour `u = y`, chacun deplace le coin de `dy dt / 2` selon `x` ; selon
    `y`, les deux parts s'annulent.
    """
    dx, dy, dt = sp.symbols("dx dy dt", positive=True)
    parts = contributions_au_deplacement(Y, sp.Integer(0), (dx, dy), dt)
    assert parts["cisaillement"] == (dy * dt / 2, dx * dt / 2)
    assert parts["rotation"] == (dy * dt / 2, -dx * dt / 2)
    assert parts["dilatation"] == (0, 0)


def test_la_vitesse_de_rotation_est_la_moyenne_de_deux_aretes() -> None:
    """Planche 37 : `omega_OA = dv/dx`, `omega_OB = -du/dy`, et leur moyenne."""
    u, v = sp.Function("u")(X, Y), sp.Function("v")(X, Y)
    moyenne = (sp.diff(v, X) - sp.diff(u, Y)) / 2
    assert sp.simplify(taux_de_rotation(u, v) - moyenne) == 0
    assert sp.simplify(vorticite(u, v) - 2 * taux_de_rotation(u, v)) == 0


CHAMP_3D = (X * Y + Z**2, Y * Z - X, X * Z + Y**2)


def test_la_vorticite_et_le_taux_de_rotation_dans_l_espace() -> None:
    """Planche 38 : `zeta = rot V`, `omega = zeta / 2`, les trois composantes."""
    u, v, w = CHAMP_3D
    attendue = (
        sp.diff(w, Y) - sp.diff(v, Z),
        sp.diff(u, Z) - sp.diff(w, X),
        sp.diff(v, X) - sp.diff(u, Y),
    )
    assert vorticite_dans_l_espace(CHAMP_3D) == attendue
    rotation = taux_de_rotation_dans_l_espace(CHAMP_3D)
    assert rotation == tuple(composante / 2 for composante in attendue)
    # Dans le plan, la composante selon z est celle du chapitre a deux dimensions.
    plan = (CHAMP_3D[0].subs(Z, 0), CHAMP_3D[1].subs(Z, 0), sp.Integer(0))
    assert vorticite_dans_l_espace(plan)[2] == vorticite(plan[0], plan[1])


def test_les_deux_tourbillons_de_la_planche_40() -> None:
    """`u_theta = omega r` est rotationnel ; `u_theta = K / r` ne l'est pas."""
    vitesse, constante = sp.symbols("omega K", positive=True)
    solide = ChampDeVitesse(-vitesse * Y, vitesse * X)
    assert solide.est_rotationnel()
    assert solide.vorticite() == 2 * vitesse
    carre = X**2 + Y**2
    libre = ChampDeVitesse(-constante * Y / carre, constante * X / carre)
    assert not libre.est_rotationnel()


def test_le_tenseur_des_deformations_et_la_dilatation_volumique() -> None:
    """Planches 42 a 45 : `eps_ij = (du_i/dx_j + du_j/dx_i) / 2`, trace = `div V`."""
    tenseur = tenseur_des_deformations(CHAMP_3D)
    assert tenseur == tenseur.T
    u, v, w = CHAMP_3D
    assert tenseur[0, 0] == sp.diff(u, X)
    assert tenseur[0, 1] == (sp.diff(u, Y) + sp.diff(v, X)) / 2
    assert tenseur[1, 2] == (sp.diff(v, Z) + sp.diff(w, Y)) / 2
    divergence = sp.diff(u, X) + sp.diff(v, Y) + sp.diff(w, Z)
    assert sp.simplify(dilatation_volumique(CHAMP_3D) - divergence) == 0
    assert sp.simplify(tenseur.trace() - divergence) == 0
    # Dans le plan, c'est le tenseur du chapitre a deux dimensions.
    plan = (2 * X - Y, X + 3 * Y, sp.Integer(0))
    assert tenseur_des_deformations(plan)[:2, :2] == tenseur_taux_deformation(
        plan[0], plan[1]
    )


def test_le_premier_exercice_de_la_derniere_planche() -> None:
    """`u = 2x - y`, `v = x + 3y` : rotationnel, et `a(-1, 2) = (-13, 11)`."""
    champ = ChampDeVitesse(2 * X - Y, X + 3 * Y)
    assert champ.est_stationnaire()
    assert champ.est_rotationnel()
    assert champ.vorticite() == 2
    assert champ.acceleration_en(-1.0, 2.0, 0.0) == pytest.approx((-13.0, 11.0))


def test_le_second_exercice_de_la_derniere_planche() -> None:
    """`V = -y i + x j` : incompressible, et ses lignes de courant sont des cercles."""
    champ = ChampDeVitesse(-Y, X)
    assert champ.est_incompressible()
    psi = champ.fonction_de_courant()
    # `psi` ne varie pas le long d'un cercle : son gradient est radial.
    assert sp.simplify(X * sp.diff(psi, Y) - Y * sp.diff(psi, X)) == 0
    assert sp.diff(psi, X) != 0


def test_le_debit_d_une_grandeur_extensive() -> None:
    """Planche 54 : `B = b m`, `B_point = b rho v A`."""
    assert debit_de_grandeur(1, 1000.0, 2.0, 0.01) == pytest.approx(20.0)
    assert debit_de_grandeur(3.0, 1000.0, 2.0, 0.01) == pytest.approx(60.0)


def test_le_theoreme_de_transport_est_la_regle_de_leibniz() -> None:
    """Planche 56 : `dB_sys/dt = dB_CV/dt - b_1 rho_1 v_1 A_1 + b_2 rho_2 v_2 A_2`.

    Le systeme occupe `[x_1(t), x_2(t)]` dans une conduite ; `f(x, t)` est
    `rho b A`, la grandeur par unite de longueur.
    """
    lineique = sp.Function("f")(X, T)
    amont, aval = sp.Function("x_1")(T), sp.Function("x_2")(T)
    du_systeme = sp.diff(sp.Integral(lineique, (X, amont, aval)), T).doit()
    debut, fin = sp.symbols("a b", real=True)
    du_volume = sp.Integral(sp.diff(lineique, T), (X, debut, fin))
    entrant = lineique.subs(X, debut) * sp.diff(amont, T)
    sortant = lineique.subs(X, fin) * sp.diff(aval, T)
    par_le_theoreme = variation_du_systeme(du_volume, [entrant], [sortant])
    a_l_instant = {debut: amont, fin: aval}
    assert sp.simplify(du_systeme - par_le_theoreme.subs(a_l_instant)) == 0


def test_la_conservation_de_la_masse_en_regime_permanent() -> None:
    """`b = 1`, masse du systeme constante, volume de controle stationnaire."""
    entree = debit_de_grandeur(1, 1000.0, 2.0, 0.03)
    sortie = debit_de_grandeur(1, 1000.0, 6.0, 0.01)
    assert variation_du_systeme(0.0, [entree], [sortie]) == pytest.approx(0.0)


def test_le_signe_du_flux_vient_de_la_normale_sortante() -> None:
    """Planche 57 : `V . n > 0` en sortie (`theta < 90`), `< 0` en entree."""
    sortie = flux_a_travers(1, 1000.0, (2.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.01)
    entree = flux_a_travers(1, 1000.0, (2.0, 0.0, 0.0), (-1.0, 0.0, 0.0), 0.01)
    rasant = flux_a_travers(1, 1000.0, (2.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.01)
    assert sortie == pytest.approx(20.0)
    assert entree == pytest.approx(-20.0)
    assert rasant == 0


def test_une_normale_non_unitaire_est_refusee() -> None:
    with pytest.raises(ValueError, match="normale unitaire attendue"):
        flux_a_travers(1, 1000.0, (2.0, 0.0, 0.0), (2.0, 0.0, 0.0), 0.01)
