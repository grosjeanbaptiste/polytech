"""L'ecoulement potentiel autour d'un cylindre : un doublet dans un ecoulement uniforme.

Le cours ne derive pas ce champ ; Cleynen, *Fluid Dynamics for
Engineers*, chapitre 11 (`sources/externes/CleynenFluidDynamics/11/
chap11.tex`), en donne la fonction de courant, les vitesses, la pression
parietale par Bernoulli et le paradoxe de d'Alembert. Les tests verifient
ces formules, et ce que le modele doit tenir par construction : le
laplacien nul, la paroi imperméable, l'ecoulement uniforme au loin.
"""

from __future__ import annotations

import sympy as sp

import polytech.fluides.domain.cylindre_potentiel as cyl

R, U = sp.symbols("R U", positive=True)
r = sp.Symbol("r", positive=True)
theta = sp.Symbol("theta", real=True)


def _psi() -> sp.Expr:
    return cyl.fonction_de_courant(r, theta, U, R)


def test_la_fonction_de_courant_est_celle_de_cleynen() -> None:
    """`psi = U sin(theta) (r - R^2 / r)`, chapitre 11."""
    attendu = U * sp.sin(theta) * (r - R**2 / r)
    assert sp.simplify(_psi() - attendu) == 0


def test_le_champ_est_irrotationnel_et_incompressible() -> None:
    """Le laplacien polaire de `psi` est nul : il n'y a pas de tourbillon."""
    psi = _psi()
    laplacien = sp.diff(r * sp.diff(psi, r), r) / r + sp.diff(psi, theta, 2) / r**2
    assert sp.simplify(laplacien) == 0


def test_la_paroi_est_une_ligne_de_courant() -> None:
    assert sp.simplify(_psi().subs(r, R)) == 0
    assert sp.simplify(cyl.vitesse_radiale(R, theta, U, R)) == 0


def test_loin_du_cylindre_l_ecoulement_redevient_uniforme() -> None:
    """`v_r -> U cos(theta)`, `v_theta -> -U sin(theta)`."""
    assert sp.limit(cyl.vitesse_radiale(r, theta, U, R), r, sp.oo) == U * sp.cos(theta)
    lointaine = sp.limit(cyl.vitesse_tangentielle(r, theta, U, R), r, sp.oo)
    assert sp.simplify(lointaine + U * sp.sin(theta)) == 0


def test_les_vitesses_sont_celles_de_cleynen() -> None:
    radiale = U * sp.cos(theta) * (1 - R**2 / r**2)
    tangentielle = -U * sp.sin(theta) * (1 + R**2 / r**2)
    assert sp.simplify(cyl.vitesse_radiale(r, theta, U, R) - radiale) == 0
    assert sp.simplify(cyl.vitesse_tangentielle(r, theta, U, R) - tangentielle) == 0


def test_a_la_paroi_la_vitesse_vaut_deux_u_sinus() -> None:
    parietale = cyl.vitesse_tangentielle(R, theta, U, R)
    assert sp.simplify(parietale + 2 * U * sp.sin(theta)) == 0


def test_le_coefficient_de_pression_parietal() -> None:
    """Bernoulli le long de la ligne de courant de paroi : `1 - 4 sin^2(theta)`."""
    cp = cyl.coefficient_de_pression_parietal(theta)
    assert sp.simplify(cp - (1 - 4 * sp.sin(theta) ** 2)) == 0
    assert cp.subs(theta, 0) == 1
    assert cp.subs(theta, sp.pi / 2) == -3
    assert sp.simplify(cp.subs(theta, sp.pi / 6)) == 0


def test_le_coefficient_de_pression_du_champ_rejoint_celui_de_la_paroi() -> None:
    champ = cyl.coefficient_de_pression(r, theta, R)
    assert (
        sp.simplify(champ.subs(r, R) - cyl.coefficient_de_pression_parietal(theta)) == 0
    )
    assert sp.limit(champ, r, sp.oo) == 0


def test_les_points_d_arret_sont_a_zero_et_pi() -> None:
    assert cyl.points_d_arret() == (sp.Integer(0), sp.pi)
    for angle in cyl.points_d_arret():
        assert cyl.vitesse_tangentielle(R, angle, U, R) == 0


def test_le_paradoxe_de_d_alembert() -> None:
    """Ni trainee ni portance : la pression parietale est symetrique."""
    trainee, portance = cyl.efforts_de_pression_reduits()
    assert trainee == 0
    assert portance == 0


def test_une_pression_dissymetrique_donne_une_trainee() -> None:
    """`C_p = cos(theta)` : trainee `-int cos^2 = -pi`, portance nulle."""
    angle = cyl.ANGLE
    trainee, portance = cyl.efforts_de_pression_reduits(sp.cos(angle))
    assert trainee == -sp.pi
    assert portance == 0


def test_le_potentiel_des_vitesses() -> None:
    """`phi = U cos(theta) (r + R^2 / r)`, dont les vitesses derivent aussi."""
    phi = cyl.potentiel_des_vitesses(r, theta, U, R)
    assert sp.simplify(phi - U * sp.cos(theta) * (r + R**2 / r)) == 0
    assert sp.simplify(sp.diff(phi, r) - cyl.vitesse_radiale(r, theta, U, R)) == 0
    tangentielle = sp.diff(phi, theta) / r - cyl.vitesse_tangentielle(r, theta, U, R)
    assert sp.simplify(tangentielle) == 0


def test_le_coefficient_de_pression_du_champ_en_forme_close() -> None:
    """`C_p = 2 (R/r)^2 cos(2 theta) - (R/r)^4`."""
    a = R**2 / r**2
    attendu = 2 * a * sp.cos(2 * theta) - a**2
    assert sp.simplify(cyl.coefficient_de_pression(r, theta, R) - attendu) == 0
