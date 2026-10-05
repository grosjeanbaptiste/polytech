"""Planches d'electromagnetisme, chapitre IV.1 : equations generales.

Sources : `Electromagnetisme/SLIDES/Slides_04_Etats_variables_partie_1.pdf`,
planches 2 a 22, lues sur les pages rendues, et la note manuscrite des
transparents 21 et 22 (`Slides_04_Note_T21_T22_Equation_d_onde.pdf`).

Le fascicule resume le chapitre 14 du syllabus, et sa fin va plus loin :
l'equation d'onde dans un dielectrique et la vitesse `1 / racine(eps mu)`,
que la note demontre par le changement de variable `w = z - v t`.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.conduction import (
    champ_d_ohm,
    puissance_volumique_de_joule,
)
from polytech.electromagnetisme.domain.constantes import VITESSE_DE_LA_LUMIERE
from polytech.electromagnetisme.domain.courant_total import (
    courant_de_deplacement,
    courant_total,
)
from polytech.electromagnetisme.domain.electrostatique.cylindre import PERMITTIVITE
from polytech.electromagnetisme.domain.electrostatique.lois import (
    divergence_cartesienne,
)
from polytech.electromagnetisme.domain.electrostatique.polarisation import (
    deplacement,
)
from polytech.electromagnetisme.domain.induction_totale import (
    fem_induite_dynamique,
    fem_induite_statique,
    sens_de_la_fem_induite,
)
from polytech.electromagnetisme.domain.magnetostatique.biot_et_savart import (
    PERMEABILITE,
)
from polytech.electromagnetisme.domain.magnetostatique.champ_magnetique import (
    champ_magnetique,
)
from polytech.electromagnetisme.domain.magnetostatique.induction import (
    force_electromotrice,
)
from polytech.electromagnetisme.domain.magnetostatique.potentiel_vecteur import (
    induction_du_potentiel,
)
from polytech.electromagnetisme.domain.maxwell import (
    residu_de_gauss_electrique,
    residu_de_gauss_magnetique,
    residus_d_ampere_maxwell,
    residus_de_faraday,
)
from polytech.electromagnetisme.domain.onde import (
    celerite_dans_un_milieu,
    est_une_onde,
)
from polytech.electromagnetisme.domain.potentiels_variables import (
    champ_des_potentiels,
)
from polytech.electromagnetisme.domain.symboles import CARTESIENNES, TEMPS

X, Y, Z = CARTESIENNES
ARGUMENTS = (X, Y, Z, TEMPS)
NUL = (0, 0, 0)


def test_l_induction_reste_a_divergence_nulle() -> None:
    """Planche 4 : `div B = 0` dans les etats variables, donc `B = rot A`."""
    vecteur = tuple(sp.Function(n)(*ARGUMENTS) for n in ("A_x", "A_y", "A_z"))
    assert residu_de_gauss_magnetique(induction_du_potentiel(vecteur)) == 0


def test_l_alternateur_ne_produit_qu_une_fem_induite_statique() -> None:
    """Planches 6 a 8 : les bobines sont fixes, c'est l'aimant qui tourne."""
    crete, pulsation = sp.symbols("Phi_0 omega", positive=True)
    position = sp.Symbol("x_b", real=True)
    flux = crete * sp.cos(pulsation * TEMPS)
    statique = fem_induite_statique(flux)
    dynamique = fem_induite_dynamique(flux, position, 0)
    assert statique == crete * pulsation * sp.sin(pulsation * TEMPS)
    assert dynamique == 0
    assert sp.simplify(force_electromotrice(flux) - statique - dynamique) == 0


def test_les_deux_exemples_de_la_loi_de_lenz() -> None:
    """Planche 10 : `delta phi > 0` donne `E_12 < 0`, et inversement."""
    assert sens_de_la_fem_induite(0.3) == -1
    assert sens_de_la_fem_induite(-0.3) == 1


def test_le_champ_electrique_depend_des_deux_potentiels() -> None:
    """Planche 12 : `E_s + E_i = -grad V - dA/dt`, et `rot = -dB/dt`."""
    scalaire = sp.Function("V")(*ARGUMENTS)
    vecteur = tuple(sp.Function(n)(*ARGUMENTS) for n in ("A_x", "A_y", "A_z"))
    champ = champ_des_potentiels(scalaire, vecteur)
    assert residus_de_faraday(champ, induction_du_potentiel(vecteur)) == NUL


def test_la_derivee_du_deplacement_joue_le_role_d_un_courant() -> None:
    """Planche 14 : `i_d = dD/dt`, et `i_cd + i_cv + i_d` est a divergence nulle."""
    charge = sp.exp(-TEMPS) * sp.cos(X)
    induction = (sp.exp(-TEMPS) * sp.sin(X), 0, 0)
    conduction = (sp.exp(-TEMPS) * sp.sin(X), 0, 0)
    assert residu_de_gauss_electrique(induction, charge) == 0
    assert courant_de_deplacement(induction) == (-sp.exp(-TEMPS) * sp.sin(X), 0, 0)
    assert divergence_cartesienne(courant_total(conduction, NUL, induction)) == 0


def test_les_trois_relations_complementaires() -> None:
    """Planches 16 a 18 : `D = eps_0 E + P`, `H = B / mu_0 - M`, `E = rho i`."""
    champ, polarise = sp.symbols("E P", real=True)
    assert deplacement((champ, 0, 0), (polarise, 0, 0))[0] == (
        PERMITTIVITE * champ + polarise
    )
    induction, aimante = sp.symbols("B M", real=True)
    assert champ_magnetique((induction, 0, 0), (aimante, 0, 0))[0] == (
        induction / PERMEABILITE - aimante
    )
    assert champ_d_ohm(3, (2, 0, 0)) == (6, 0, 0)
    assert puissance_volumique_de_joule(3, (2, 0, 0)) == 12


EPS, MU = sp.symbols("varepsilon mu", positive=True)
VITESSE = celerite_dans_un_milieu(EPS, MU)
PROFIL = sp.Function("g")(X - VITESSE * TEMPS)
CHAMP = (0, PROFIL, 0)
MAGNETIQUE = (0, 0, PROFIL / (MU * VITESSE))
"""Une onde plane dans un dielectrique : `E_y = g(x - v t)`, `B_z = E_y / v`."""


def test_la_vitesse_de_l_onde_est_l_inverse_de_racine_de_eps_mu() -> None:
    """Planche 22 et note : `eps mu v^2 = 1`."""
    assert sp.simplify(EPS * MU * VITESSE**2 - 1) == 0


def test_dans_le_vide_cette_vitesse_est_celle_de_la_lumiere() -> None:
    """Note : `1 / racine(8,85e-12 . 4 pi e-7) = 2,998 . 10^8 m/s`."""
    calculee = celerite_dans_un_milieu(8.85e-12, 4e-7 * sp.pi)
    assert float(calculee) == pytest.approx(2.998e8, rel=5e-4)
    assert float(calculee) == pytest.approx(VITESSE_DE_LA_LUMIERE, rel=5e-4)


def test_un_milieu_sans_permittivite_n_a_pas_de_vitesse() -> None:
    with pytest.raises(ValueError, match="nulle ou negative"):
        celerite_dans_un_milieu(0, 4e-7)
    with pytest.raises(ValueError, match="nulle ou negative"):
        celerite_dans_un_milieu(8.85e-12, -1)
    with pytest.raises(ValueError, match="nulle ou negative"):
        celerite_dans_un_milieu(-8.85e-12, -1)


def test_l_onde_plane_satisfait_les_quatre_equations_de_la_planche_20() -> None:
    """Dielectrique non charge : `eta = 0`, `i = 0`, `D = eps E`, `B = mu H`."""
    induit = tuple(EPS * e for e in CHAMP)
    induction = tuple(MU * h for h in MAGNETIQUE)
    assert residus_de_faraday(CHAMP, induction) == NUL
    assert residu_de_gauss_magnetique(induction) == 0
    assert residu_de_gauss_electrique(induit, sp.Integer(0)) == 0
    assert residus_d_ampere_maxwell(MAGNETIQUE, NUL, induit) == NUL


def test_le_profil_translate_resout_l_equation_d_onde() -> None:
    """Planche 21 et note : `E(z, t) = E(z - v t, 0)`, a la bonne vitesse seulement."""
    assert est_une_onde(PROFIL, VITESSE)
    trop_vite = sp.Function("g")(X - 2 * VITESSE * TEMPS)
    assert not est_une_onde(trop_vite, VITESSE)


def test_le_changement_de_variable_de_la_note() -> None:
    """`w = z - v t` : `dE/dt = -v dE/dw`, `d2E/dt2 = v^2 d2E/dw2`."""
    vitesse, variable = sp.Symbol("v", positive=True), sp.Symbol("w", real=True)
    profil = sp.Function("g")
    onde = profil(X - vitesse * TEMPS)
    en_w = {X - vitesse * TEMPS: variable}
    premiere = sp.diff(onde, TEMPS).subs(en_w).doit()
    seconde = sp.diff(onde, TEMPS, 2).subs(en_w).doit()
    assert sp.simplify(premiere + vitesse * sp.diff(profil(variable), variable)) == 0
    attendue = vitesse**2 * sp.diff(profil(variable), variable, 2)
    assert sp.simplify(seconde - attendue) == 0
