"""Planches d'electromagnetisme, chapitre IV.2 : energie et calcul de forces.

Source : `Electromagnetisme/SLIDES/Slides_04_Etats_variables_partie_2_Forces.pdf`,
planches 2 a 15, lues sur les pages rendues.

Le fascicule resume les chapitres 5 et 11 du syllabus et le theoreme de
Poynting du chapitre 14. Ses deux applications servent d'oracle : le
condensateur plan a potentiel constant (planche 11) et le couple du
paramagnetisme, `C = m ^ B`, obtenu par les travaux virtuels (planche 15).
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.conduction import (
    champ_d_ohm,
    puissance_volumique_de_joule,
)
from polytech.electromagnetisme.domain.electrostatique.coefficients import (
    SystemeDeConducteurs,
)
from polytech.electromagnetisme.domain.electrostatique.cylindre import PERMITTIVITE
from polytech.electromagnetisme.domain.energie_et_forces import (
    DEPLACEMENT,
    energie_d_un_condensateur,
    energie_d_une_inductance,
    force_a_charges_constantes,
    force_a_courant_constant,
    force_a_flux_constant,
    force_a_potentiels_constants,
)
from polytech.electromagnetisme.domain.magnetostatique.couple_des_bobines import (
    energie_de_deux_bobines,
)
from polytech.electromagnetisme.domain.magnetostatique.diamagnetisme import (
    couple_sur_un_moment,
)
from polytech.electromagnetisme.domain.magnetostatique.forces_magnetiques import (
    couple_par_l_energie,
)
from polytech.electromagnetisme.domain.magnetostatique.inductances import (
    SystemeDeCircuits,
)
from polytech.electromagnetisme.domain.onde import celerite_dans_un_milieu
from polytech.electromagnetisme.domain.poynting import (
    puissance_emmagasinee,
    residu_de_poynting,
    vecteur_de_poynting,
)
from polytech.electromagnetisme.domain.symboles import ABSCISSE, TEMPS

NUL = (0, 0, 0)


def test_la_puissance_fournie_aux_charges_est_l_effet_joule() -> None:
    """Planche 3 : `dP = E . i dV`, la force magnetique ne travaillant pas."""
    resistivite, courant = sp.Integer(3), (2, 0, 1)
    champ = champ_d_ohm(resistivite, courant)
    fournie = sum(e * i for e, i in zip(champ, courant, strict=True))
    assert fournie == puissance_volumique_de_joule(resistivite, courant)


def test_le_bilan_de_poynting_sur_une_onde_dans_un_dielectrique() -> None:
    """Planche 5 : `d(W_em)/dt = -P_j - flux de E ^ H`, ici sans effet Joule."""
    eps, mu = sp.symbols("varepsilon mu", positive=True)
    vitesse = celerite_dans_un_milieu(eps, mu)
    profil = sp.Function("g")(ABSCISSE - vitesse * TEMPS)
    champ, magnetique = (0, profil, 0), (0, 0, profil / (mu * vitesse))
    induit = tuple(eps * e for e in champ)
    induction = tuple(mu * h for h in magnetique)
    assert residu_de_poynting(champ, magnetique, induit, induction, NUL) == 0
    # Le vecteur de Poynting pointe dans le sens de la propagation.
    assert vecteur_de_poynting(champ, magnetique)[1:] == (0, 0)
    # Planche 4 : en milieu lineaire, c'est la derivee de (E.D + H.B) / 2.
    emmagasinee = puissance_emmagasinee(champ, magnetique, induit, induction)
    densite = (eps * profil**2 + mu * magnetique[2] ** 2) / 2
    assert sp.simplify(emmagasinee - sp.diff(densite, TEMPS)) == 0


def test_les_energies_electrique_et_magnetique_d_un_systeme() -> None:
    """Planche 6 : `W_e = (1/2) somme Q_k V_k` et `W_m = (1/2) somme I_k phi_k`."""
    coefficients = sp.Matrix([[3, 1], [1, 2]])
    charges = sp.Matrix([2, 5])
    conducteurs = SystemeDeConducteurs(coefficients)
    electrique = conducteurs.energie(charges)
    assert electrique == charges.dot(conducteurs.potentiels(charges)) / 2
    circuits = SystemeDeCircuits(coefficients)
    magnetique = circuits.energie(charges)
    assert magnetique == charges.dot(circuits.flux(charges)) / 2


def test_les_energies_du_condensateur_et_des_bobines() -> None:
    """Planche 7 : `Q^2 / (2 C) = C U^2 / 2`, `L I^2 / 2`, et deux bobines couplees."""
    capacite, charge = sp.symbols("C Q", positive=True)
    electrique = energie_d_un_condensateur(capacite, charge / capacite)
    assert sp.simplify(electrique - charge**2 / (2 * capacite)) == 0
    l11, l22, l12, i1, i2 = sp.symbols("L_11 L_22 L_12 I_1 I_2", real=True)
    assert energie_d_une_inductance(l11, i1) == l11 * i1**2 / 2
    couplees = energie_de_deux_bobines(l11, l22, l12, i1, i2)
    matrice = sp.Matrix([[l11, l12], [l12, l22]])
    assert sp.expand(couplees - SystemeDeCircuits(matrice).energie([i1, i2])) == 0


def test_le_condensateur_plan_de_la_planche_11() -> None:
    """`W = C U^2 / 2`, `C = eps_0 S / x` : `F = (1/2) U^2 eps_0 S (-1 / x^2)`."""
    tension, aire = sp.symbols("U_12 S", positive=True)
    position = sp.Symbol("x", positive=True)
    capacite = PERMITTIVITE * aire / position
    energie = energie_d_un_condensateur(capacite, tension).subs(position, DEPLACEMENT)
    force = force_a_potentiels_constants(energie).subs(DEPLACEMENT, position)
    attendue = tension**2 * PERMITTIVITE * aire * (-1 / position**2) / 2
    assert sp.simplify(force - attendue) == 0


def test_les_deux_theoremes_donnent_la_meme_force_sur_l_armature() -> None:
    """Planche 10 : `F = -(dW_e/dl)_Q = +(dW_e/dl)_U`, les armatures s'attirent."""
    charge, tension, aire = sp.symbols("Q U_12 S", positive=True)
    position = sp.Symbol("x", positive=True)
    capacite = PERMITTIVITE * aire / DEPLACEMENT
    a_charges = force_a_charges_constantes(charge**2 / (2 * capacite))
    a_potentiels = force_a_potentiels_constants(
        energie_d_un_condensateur(capacite, tension)
    )
    meme_etat = {charge: capacite * tension}
    ecart = (a_charges.subs(meme_etat) - a_potentiels).subs(DEPLACEMENT, position)
    assert sp.simplify(ecart) == 0
    assert a_charges.subs(DEPLACEMENT, position).is_negative


def test_les_deux_theoremes_magnetiques_de_la_planche_13() -> None:
    """`F = -(dW_m/dl)_phi = +(dW_m/dl)_I`, sur une bobine d'inductance `L(x)`."""
    courant, flux = sp.symbols("I phi", positive=True)
    inductance = sp.Function("L")(DEPLACEMENT)
    a_courant = force_a_courant_constant(energie_d_une_inductance(inductance, courant))
    a_flux = force_a_flux_constant(flux**2 / (2 * inductance))
    assert sp.simplify(a_flux.subs(flux, inductance * courant) - a_courant) == 0


def test_le_couple_du_paramagnetisme_par_les_travaux_virtuels() -> None:
    """Planche 15 : `L_12 I_2 = B S cos(alpha)`, d'ou `C = m ^ B`."""
    courant, induction, aire = sp.symbols("I_1 B S_i", positive=True)
    angle = sp.Symbol("alpha", real=True)
    l11, l22, i2 = sp.symbols("L_11 L_22 I_2", positive=True)
    mutuelle = induction * aire * sp.cos(angle) / i2
    energie = energie_de_deux_bobines(l11, l22, mutuelle, courant, i2)
    couple = couple_par_l_energie(energie, angle, a_courants_constants=True)
    assert sp.simplify(couple + courant * induction * aire * sp.sin(angle)) == 0
    moment = (0, courant * aire, 0)
    incline = (induction * sp.sin(angle), induction * sp.cos(angle), 0)
    assert couple_sur_un_moment(moment, incline) == (0, 0, couple)
