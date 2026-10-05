"""Syllabus d'electricite generale, chapitre 14 : induction et potentiels variables.

Source : `Electromagnetisme/SYLLABUS/Syllabus_Electricite_generale.pdf`,
pages 14-2 a 14-10 et 14-16 a 14-18, lues sur les pages rendues.

Le chapitre pose des relations sans les chiffrer. Elles sont eprouvees
sur des potentiels **quelconques** -- des fonctions indeterminees de
l'espace et du temps -- puis sur trois cas dont le resultat est connu : le
solenoide long, le rail de Laplace dans une induction variable, et une
onde plane.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.electrostatique.lois import (
    divergence_cartesienne,
    est_electrostatique,
)
from polytech.electromagnetisme.domain.electrostatique.potentiel import COULOMB
from polytech.electromagnetisme.domain.induction_totale import (
    fem_induite_dynamique,
    fem_induite_statique,
    sens_de_la_fem_induite,
)
from polytech.electromagnetisme.domain.magnetostatique.forces_magnetiques import (
    force_volumique,
)
from polytech.electromagnetisme.domain.magnetostatique.induction import (
    champ_induit_dynamique,
    force_electromotrice,
)
from polytech.electromagnetisme.domain.magnetostatique.potentiel_vecteur import (
    changement_de_jauge,
    est_en_jauge_de_coulomb,
    induction_du_potentiel,
)
from polytech.electromagnetisme.domain.maxwell import (
    residu_de_gauss_magnetique,
    residus_de_faraday,
)
from polytech.electromagnetisme.domain.onde import est_une_onde
from polytech.electromagnetisme.domain.potentiels_variables import (
    champ_des_potentiels,
    champ_induit_statique,
    changer_de_jauge,
    potentiel_retarde,
    residu_de_la_jauge_de_lorenz,
)
from polytech.electromagnetisme.domain.symboles import CARTESIENNES, TEMPS

X, Y, Z = CARTESIENNES
ARGUMENTS = (X, Y, Z, TEMPS)
SCALAIRE = sp.Function("V")(*ARGUMENTS)
VECTEUR = tuple(sp.Function(nom)(*ARGUMENTS) for nom in ("A_x", "A_y", "A_z"))
JAUGE = sp.Function("varphi")(*ARGUMENTS)
NUL = (0, 0, 0)


def test_l_induction_d_un_potentiel_vecteur_n_a_pas_de_source() -> None:
    """(14.2.1) et (14.2.3) : `B = rot A` entraine `div B = 0`."""
    assert residu_de_gauss_magnetique(induction_du_potentiel(VECTEUR)) == 0


def test_le_champ_induit_statique_tourne_autour_de_l_induction_variable() -> None:
    """(14.3.5) `E_i = -dA/dt`, d'ou (14.5.1) `rot E_i = -dB/dt`."""
    induit = champ_induit_statique(VECTEUR)
    assert induit == tuple(-sp.diff(a, TEMPS) for a in VECTEUR)
    assert residus_de_faraday(induit, induction_du_potentiel(VECTEUR)) == NUL


def test_les_potentiels_satisfont_d_office_la_loi_de_faraday() -> None:
    """(14.5.3) `E_s + E_i = -grad V - dA/dt` verifie (14.5.2)."""
    champ = champ_des_potentiels(SCALAIRE, VECTEUR)
    assert residus_de_faraday(champ, induction_du_potentiel(VECTEUR)) == NUL


def test_sans_potentiel_vecteur_le_champ_reste_electrostatique() -> None:
    """(2.5.3) : `rot E_s = 0`."""
    assert est_electrostatique(champ_des_potentiels(SCALAIRE, NUL))


def test_un_changement_de_jauge_ne_change_ni_le_champ_ni_l_induction() -> None:
    """(14.8.1) `A_1 = A + grad(phi)` et (14.8.2) `V_1 = V - d(phi)/dt`."""
    scalaire, vecteur = changer_de_jauge(SCALAIRE, VECTEUR, JAUGE)
    assert vecteur == changement_de_jauge(VECTEUR, JAUGE)
    avant = champ_des_potentiels(SCALAIRE, VECTEUR)
    apres = champ_des_potentiels(scalaire, vecteur)
    assert [sp.simplify(a - b) for a, b in zip(avant, apres, strict=True)] == [0] * 3
    inductions = zip(
        induction_du_potentiel(VECTEUR), induction_du_potentiel(vecteur), strict=True
    )
    assert [sp.simplify(a - b) for a, b in inductions] == [0] * 3


def test_le_potentiel_scalaire_seul_n_est_pas_invariant() -> None:
    """Page 14-8 : les deux termes de `E_s + E_i` changent avec la jauge."""
    scalaire, _ = changer_de_jauge(SCALAIRE, VECTEUR, X * TEMPS)
    assert sp.simplify(scalaire - SCALAIRE) == -X


def test_en_jauge_de_coulomb_le_potentiel_scalaire_suit_poisson() -> None:
    """(14.10.4) `div A = 0`, donc `div E = -laplacien V` comme en statique."""
    induction = sp.Function("B")(TEMPS)
    vecteur = (-Y * induction / 2, X * induction / 2, 0)
    assert est_en_jauge_de_coulomb(vecteur)
    champ = champ_des_potentiels(SCALAIRE, vecteur)
    laplacien = sum(sp.diff(SCALAIRE, axe, 2) for axe in CARTESIENNES)
    assert sp.simplify(divergence_cartesienne(champ) + laplacien) == 0


def test_le_solenoide_long_induit_la_fem_de_faraday() -> None:
    """(14.3.3) : la circulation de `E_i` sur un cercle vaut `-d(phi)/dt`.

    Dans un solenoide long, `A = (B(t) / 2) (-y, x, 0)` ; le cercle de
    rayon `r` embrasse le flux `pi r^2 B(t)`.
    """
    induction, rayon = sp.Function("B")(TEMPS), sp.Symbol("r", positive=True)
    angle = sp.Symbol("theta", real=True)
    induit = champ_induit_statique((-Y * induction / 2, X * induction / 2, 0))
    sur_le_cercle = {X: rayon * sp.cos(angle), Y: rayon * sp.sin(angle)}
    tangente = (-rayon * sp.sin(angle), rayon * sp.cos(angle), 0)
    integrande = sum(
        e.subs(sur_le_cercle) * d for e, d in zip(induit, tangente, strict=True)
    )
    circulation = sp.integrate(integrande, (angle, 0, 2 * sp.pi))
    flux = sp.pi * rayon**2 * induction
    assert sp.simplify(circulation - force_electromotrice(flux)) == 0


def test_la_fem_totale_est_la_somme_de_la_statique_et_de_la_dynamique() -> None:
    """(14.3.2) sur un rail : `phi = B(t) l x`, la barre etant en `x(t)`."""
    induction, longueur = sp.Function("B")(TEMPS), sp.Symbol("l", positive=True)
    abscisse, position = sp.Symbol("x_b", real=True), sp.Function("x_b")(TEMPS)
    flux = induction * longueur * abscisse
    statique = fem_induite_statique(flux)
    dynamique = fem_induite_dynamique(flux, abscisse, sp.diff(position, TEMPS))
    assert statique == -sp.diff(induction, TEMPS) * longueur * abscisse
    assert dynamique == -induction * longueur * sp.diff(position, TEMPS)
    totale = force_electromotrice(flux.subs(abscisse, position))
    assert sp.simplify(totale - (statique + dynamique).subs(abscisse, position)) == 0


def test_dans_une_induction_constante_seule_la_fem_dynamique_subsiste() -> None:
    """Page 14-3 : « il ne subsisterait que `(E_12)_d` »."""
    abscisse = sp.Symbol("x_b", real=True)
    flux = sp.Rational(1, 2) * 3 * abscisse
    assert fem_induite_statique(flux) == 0
    assert fem_induite_dynamique(flux, abscisse, 4) == -6


def test_la_loi_de_lenz_oppose_la_fem_a_la_variation_du_flux() -> None:
    """Figure 14.4.1, (a) et (b) : le signe de la variation importe, non du flux."""
    assert sens_de_la_fem_induite(0.2) == -1
    assert sens_de_la_fem_induite(-0.2) == 1
    assert sens_de_la_fem_induite(0.0) == 0


def test_la_force_sur_le_courant_induit_freine_le_mouvement() -> None:
    """Figure 14.4.1c : `i_cd` suit `E_d = v ^ B`, et `i_cd ^ B` s'oppose a `v`."""
    vitesse, induction = (2, 1, 0), (0, 3, 5)
    dynamique = champ_induit_dynamique(vitesse, induction)
    force = force_volumique(dynamique, NUL, induction)
    puissance = sum(f * v for f, v in zip(force, vitesse, strict=True))
    assert puissance < 0
    # Page 13-21 : `E_d . i_cd = -f . v`, la puissance electrique vient du mouvement.
    assert puissance == -sum(e * e for e in dynamique)


def test_les_potentiels_d_une_onde_plane_sont_en_jauge_de_lorenz() -> None:
    """(14.18.4) : `div A + (1 / c_0^2) dV/dt = 0`, et `E / B = c_0`."""
    celerite = sp.Symbol("c_0", positive=True)
    profil = sp.Function("g")(TEMPS - X / celerite)
    vecteur = (0, profil, 0)
    assert residu_de_la_jauge_de_lorenz(0, vecteur, celerite) == 0
    assert est_une_onde(profil, celerite)
    champ = champ_des_potentiels(0, vecteur)
    induction = induction_du_potentiel(vecteur)
    assert sp.simplify(champ[1] - celerite * induction[2]) == 0


def test_un_potentiel_statique_hors_jauge_laisse_un_residu() -> None:
    """Un potentiel scalaire qui croit seul viole la condition de Lorenz."""
    celerite = sp.Symbol("c_0", positive=True)
    assert residu_de_la_jauge_de_lorenz(TEMPS, NUL, celerite) == 1 / celerite**2


def test_le_potentiel_retarde_se_propage_a_la_celerite() -> None:
    """(14.18.2a) pour une charge ponctuelle : `q(t - r / c_0) / (4 pi eps_0 r)`.

    Il satisfait l'equation des ondes spheriques, et redonne le potentiel
    instantane (14.10.1a) quand la celerite est infinie.
    """
    celerite, distance = sp.symbols("c_0 r", positive=True)
    charge = sp.Function("q")
    retarde = potentiel_retarde(charge(TEMPS), distance, celerite)
    assert retarde == COULOMB * charge(TEMPS - distance / celerite) / distance
    onde = sp.diff(distance * retarde, distance, 2) / distance
    assert sp.simplify(onde - sp.diff(retarde, TEMPS, 2) / celerite**2) == 0
    pulsation = sp.Symbol("omega", positive=True)
    sinusoidal = potentiel_retarde(sp.cos(pulsation * TEMPS), distance, celerite)
    instantane = COULOMB * sp.cos(pulsation * TEMPS) / distance
    assert sp.limit(sinusoidal, celerite, sp.oo) == instantane


def test_une_celerite_nulle_n_a_pas_de_retard() -> None:
    with pytest.raises(ValueError, match="celerite nulle"):
        potentiel_retarde(sp.cos(TEMPS), sp.Symbol("r", positive=True), 0)
