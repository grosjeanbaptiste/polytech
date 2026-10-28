"""Syllabus de mathematiques appliquees, section 1.4 : les EDO lineaires d'ordre n.

Source : `Mathematique/SYLLABUS/Syllabus_Mathematiques_appliquees_2016_2017.pdf`,
pages 18 a 31. Oracles : les exemples 1.7 a 1.17 -- l'algebre des
operateurs, les solutions homogenes selon les racines, les solutions
particulieres par l'inverse de l'operateur, le principe de superposition.

Deux coquilles du syllabus sont etablies ici : la relation entre les deux
dephasages (page 24) et la solution particuliere en cosinus de l'exemple
1.17 (page 30).
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.comparaison import egales
from polytech.maths.domain.differentielles.formes_sinusoidales import (
    en_cosinus,
    en_sinus,
)
from polytech.maths.domain.differentielles.operateurs import (
    D,
    OperateurDifferentiel,
)
from polytech.maths.domain.differentielles.particulieres import (
    pour_un_polynome,
    pour_un_polynome_module,
    pour_une_exponentielle,
    pour_une_sinusoide,
)
from polytech.maths.domain.symboles import TEMPS as T

R = sp.Rational
exp, cos, sin, sqrt = sp.exp, sp.cos, sp.sin, sp.sqrt


def _annule(operateur: OperateurDifferentiel, fonction: sp.Expr) -> bool:
    return egales(operateur.appliquer(fonction), 0)


def _rend(operateur: OperateurDifferentiel, fonction: sp.Expr, second: sp.Expr) -> bool:
    return egales(operateur.appliquer(fonction), second)


def test_la_composition_des_operateurs_est_le_produit_des_polynomes() -> None:
    """Exemple 1.7 : `(D + 1)(D^2 + D + 1) = D^3 + 2 D^2 + 2 D + 1`."""
    produit = OperateurDifferentiel(D + 1).composer(OperateurDifferentiel(D**2 + D + 1))
    assert produit == OperateurDifferentiel(D**3 + 2 * D**2 + 2 * D + 1)


def test_composer_deux_operateurs_c_est_les_appliquer_l_un_apres_l_autre() -> None:
    """Page 19 : `A(D)(B(D) y) = (A.B)(D) y`, sur `y = t^3 e^(2t)`."""
    a, b = OperateurDifferentiel(D + 1), OperateurDifferentiel(D**2 + D + 1)
    fonction = T**3 * exp(2 * T)
    l_un_apres_l_autre = a.appliquer(b.appliquer(fonction))
    assert egales(l_un_apres_l_autre, a.composer(b).appliquer(fonction))


def test_un_operateur_differentiel_est_lineaire() -> None:
    """Page 19 : `K(D)(c1 y1 + c2 y2) = c1 K(D) y1 + c2 K(D) y2`."""
    operateur = OperateurDifferentiel(D**2 + 3 * D + 2)
    y1, y2 = sin(T), T**2
    ensemble = operateur.appliquer(5 * y1 - 7 * y2)
    separes = 5 * operateur.appliquer(y1) - 7 * operateur.appliquer(y2)
    assert egales(ensemble, separes)


@pytest.mark.parametrize(
    ("polynome", "racines", "base"),
    [
        (D**2 - 3 * D + 2, {1: 1, 2: 1}, {exp(T), exp(2 * T)}),
        (D**3 - 3 * D + 2, {1: 2, -2: 1}, {exp(T), T * exp(T), exp(-2 * T)}),
        (D**3 + D**2, {0: 2, -1: 1}, {sp.Integer(1), T, exp(-T)}),
        (D**2 - 2 * D + 1, {1: 2}, {exp(T), T * exp(T)}),
    ],
)
def test_les_racines_et_la_base_des_exemples_1_9(
    polynome: sp.Expr, racines: dict[int, int], base: set[sp.Expr]
) -> None:
    """Racine `b` de multiplicite `k` : `e^(bt), t e^(bt), ..., t^(k-1) e^(bt)`."""
    operateur = OperateurDifferentiel(polynome)
    assert operateur.racines() == racines
    assert set(operateur.base_des_solutions()) == base
    assert all(_annule(operateur, fonction) for fonction in base)


def test_l_espace_des_solutions_est_de_dimension_n() -> None:
    """Page 22 : `dim X = n`, la somme des multiplicites."""
    operateur = OperateurDifferentiel(D**3 - 3 * D + 2)
    assert len(operateur.base_des_solutions()) == operateur.ordre == 3


def test_des_racines_complexes_donnent_des_sinus_et_des_cosinus_amortis() -> None:
    """Exemple 1.10.1 : `y'' + y' + y = 0`, racines `(-1 +- j sqrt 3)/2`."""
    operateur = OperateurDifferentiel(D**2 + D + 1)
    attendue = {
        exp(-T / 2) * cos(sqrt(3) * T / 2),
        exp(-T / 2) * sin(sqrt(3) * T / 2),
    }
    assert set(operateur.base_des_solutions()) == attendue
    assert all(_annule(operateur, fonction) for fonction in attendue)


def test_des_racines_complexes_triples_portent_des_polynomes_de_degre_deux() -> None:
    """Exemple 1.10.2 : `(D^2 + D + 1)^3 y = 0`, six fonctions de base."""
    operateur = OperateurDifferentiel((D**2 + D + 1) ** 3)
    base = operateur.base_des_solutions()
    assert len(base) == 6
    assert T**2 * exp(-T / 2) * cos(sqrt(3) * T / 2) in base
    assert all(_annule(operateur, fonction) for fonction in base)


def test_la_solution_generale_de_l_exemple_1_8() -> None:
    """`y'' + y = t` : `c1 cos t + c2 sin t + t`."""
    operateur = OperateurDifferentiel(D**2 + 1)
    assert set(operateur.base_des_solutions()) == {cos(T), sin(T)}
    assert _rend(operateur, T, T)
    assert pour_un_polynome(operateur, T) == T


def test_l_inverse_de_l_operateur_par_puissances_croissantes() -> None:
    """Exemple 1.12 : `1/(1 + D^2) = 1 - D^2 + D^4 - ...`, tronque au degre 4."""
    operateur = OperateurDifferentiel(D**2 + 1)
    assert operateur.inverse_tronque(4) == 1 - D**2 + D**4
    assert operateur.inverse_tronque(1) == 1


def test_la_solution_particuliere_de_l_exemple_1_12() -> None:
    """`y'' + y = 1 + t + t^4` : `25 + t - 12 t^2 + t^4`."""
    operateur = OperateurDifferentiel(D**2 + 1)
    particuliere = pour_un_polynome(operateur, 1 + T + T**4)
    assert sp.expand(particuliere - (25 + T - 12 * T**2 + T**4)) == 0
    assert _rend(operateur, particuliere, 1 + T + T**4)


def test_la_solution_particuliere_de_l_exemple_1_13() -> None:
    """`y'''' - y'' = 1 + t + t^2`, `K(0) = 0` : `-3 t^2/2 - t^3/6 - t^4/12`."""
    operateur = OperateurDifferentiel(D**4 - D**2)
    particuliere = pour_un_polynome(operateur, 1 + T + T**2)
    assert sp.expand(particuliere - (-R(3, 2) * T**2 - T**3 / 6 - T**4 / 12)) == 0
    assert _rend(operateur, particuliere, 1 + T + T**2)
    assert set(operateur.base_des_solutions()) == {1, T, exp(T), exp(-T)}


def test_l_inverse_d_un_operateur_nul_en_zero_n_existe_pas() -> None:
    """Page 27 : la division par puissances croissantes demande `K(0) != 0`."""
    with pytest.raises(
        ValueError, match="division par puissances croissantes est impossible"
    ):
        OperateurDifferentiel(D**4 - D**2).inverse_tronque(2)


def test_la_solution_particuliere_de_l_exemple_1_14() -> None:
    """`y'' + y = t e^(-t)` : `y = e^(-t) z`, `K(D - 1) z = t`, `z = (t + 1)/2`."""
    operateur = OperateurDifferentiel(D**2 + 1)
    assert operateur.decaler(-1) == OperateurDifferentiel(D**2 - 2 * D + 2)
    particuliere = pour_un_polynome_module(operateur, T, -1)
    assert egales(particuliere, exp(-T) * (T + 1) / 2)
    assert _rend(operateur, particuliere, T * exp(-T))


def test_les_solutions_particulieres_des_exemples_1_15_et_1_16() -> None:
    """Une constante : `alpha / K(0) = 3`. Une exponentielle : `e^t / K(1)`."""
    operateur = OperateurDifferentiel(D**2 + 1)
    assert pour_un_polynome(operateur, sp.Integer(3)) == 3
    assert pour_une_exponentielle(operateur, 1) == exp(T) / 2
    assert _rend(operateur, exp(T) / 2, exp(T))


def test_une_exponentielle_racine_de_l_equation_caracteristique_est_refusee() -> None:
    """`e^(alpha t) / K(alpha)` n'a pas de sens si `K(alpha) = 0`."""
    with pytest.raises(ValueError, match="racine de l'equation caracteristique"):
        pour_une_exponentielle(OperateurDifferentiel(D**2 - 1), 1)


def test_une_exponentielle_racine_se_traite_par_le_polynome_module() -> None:
    """`y'' - y = e^t` : `y = e^t z`, `K(D + 1) z = 1`, `z = t/2`."""
    operateur = OperateurDifferentiel(D**2 - 1)
    particuliere = pour_un_polynome_module(operateur, sp.Integer(1), 1)
    assert egales(particuliere, T * exp(T) / 2)
    assert _rend(operateur, particuliere, exp(T))


def test_la_premiere_sollicitation_de_l_exemple_1_17() -> None:
    """`y'' + 3y' + 2y = cos t` : `Re(e^(jt) / K(j))`, avec `K(j) = 1 + 3j`."""
    operateur = OperateurDifferentiel(D**2 + 3 * D + 2)
    assert operateur.racines() == {-1: 1, -2: 1}
    particuliere = pour_une_sinusoide(operateur, cos, 1)
    assert egales(particuliere, (cos(T) + 3 * sin(T)) / 10)
    assert _rend(operateur, particuliere, cos(T))


def test_la_solution_imprimee_page_30_porte_un_signe_faux() -> None:
    """Le syllabus ecrit `(cos t - 3 sin t)/10` ; c'est `(cos t + 3 sin t)/10`.

    `1/(1 + 3j) = (1 - 3j)/10`, et la partie reelle de `(1 - 3j) e^(jt)` est
    `cos t + 3 sin t`. La meme page ecrit aussi `K(D) = D^2 + 3D + 3` pour
    l'equation `y'' + 3y' + 2y`, dont les racines sont bien -1 et -2.
    """
    operateur = OperateurDifferentiel(D**2 + 3 * D + 2)
    imprimee = (cos(T) - 3 * sin(T)) / 10
    assert not _rend(operateur, imprimee, cos(T))
    assert egales(operateur.appliquer(imprimee), -(4 * cos(T) + 3 * sin(T)) / 5)
    assert OperateurDifferentiel(D**2 + 3 * D + 3).racines() != {-1: 1, -2: 1}


def test_le_principe_de_superposition_de_l_exemple_1_17() -> None:
    """Les trois sollicitations `cos t`, `t e^(-t)`, `t^2`, puis leur somme."""
    operateur = OperateurDifferentiel(D**2 + 3 * D + 2)
    deuxieme = pour_un_polynome_module(operateur, T, -1)
    troisieme = pour_un_polynome(operateur, T**2)
    assert egales(deuxieme, (T**2 / 2 - T) * exp(-T))
    assert sp.expand(troisieme - (T**2 / 2 - R(3, 2) * T + R(7, 4))) == 0
    somme = pour_une_sinusoide(operateur, cos, 1) + deuxieme + troisieme
    assert _rend(operateur, somme, cos(T) + T * exp(-T) + T**2)


def test_la_troisieme_sollicitation_par_l_inverse_de_l_operateur() -> None:
    """Page 31 : `1/(2 + 3D + D^2) = 1/2 - 3D/4 + 7D^2/8`."""
    operateur = OperateurDifferentiel(D**2 + 3 * D + 2)
    assert operateur.inverse_tronque(2) == R(1, 2) - R(3, 4) * D + R(7, 8) * D**2


def test_un_sinus_se_traite_par_la_partie_imaginaire() -> None:
    operateur = OperateurDifferentiel(D**2 + 3 * D + 2)
    particuliere = pour_une_sinusoide(operateur, sin, 2)
    assert _rend(operateur, particuliere, sin(2 * T))


def test_une_sinusoide_a_la_pulsation_propre_donne_une_reponse_qui_croit() -> None:
    """`y'' + y = cos t` : `K(j) = 0`, c'est la resonance -- `y = (t/2) sin t`."""
    operateur = OperateurDifferentiel(D**2 + 1)
    particuliere = pour_une_sinusoide(operateur, cos, 1)
    assert egales(particuliere, T / 2 * sin(T))
    assert _rend(operateur, particuliere, cos(T))


def test_les_deux_formes_dephasees_de_l_exemple_1_11() -> None:
    """`A = B = 1` : `sqrt 2 sin(wt + pi/4) = sqrt 2 cos(wt - pi/4)`."""
    assert en_sinus(1, 1) == (sqrt(2), sp.pi / 4)
    assert en_cosinus(1, 1) == (sqrt(2), -sp.pi / 4)


@pytest.mark.parametrize(("a", "b"), [(1, 1), (3, 4), (-2, 5), (1, -1), (-3, -4)])
def test_les_formes_dephasees_redonnent_la_forme_en_cosinus_et_sinus(
    a: int, b: int
) -> None:
    """(1.33) et (1.34) : `A cos wt + B sin wt = K sin(wt + phi) = K cos(wt + psi)`."""
    reference = a * cos(T) + b * sin(T)
    amplitude, phi = en_sinus(a, b)
    autre_amplitude, psi = en_cosinus(a, b)
    assert amplitude == autre_amplitude == sqrt(a**2 + b**2)
    assert egales(sp.expand_trig(amplitude * sin(T + phi)), reference)
    assert egales(sp.expand_trig(amplitude * cos(T + psi)), reference)


def test_les_deux_dephasages_different_de_pi_sur_deux() -> None:
    """Page 24 : le syllabus imprime `phi = psi - pi/2` ; c'est `phi = psi + pi/2`.

    `sin(x) = cos(x - pi/2)`, donc `psi = phi - pi/2` ; l'exemple 1.11 le
    confirme avec `phi = pi/4` et `psi = -pi/4`.
    """
    for a, b in ((1, 1), (3, 4), (-2, 5)):
        _, phi = en_sinus(a, b)
        _, psi = en_cosinus(a, b)
        ecart = phi - psi
        assert egales(sin(ecart), 1)
        assert egales(cos(ecart), 0)


def test_une_amplitude_nulle_n_a_pas_de_dephasage() -> None:
    with pytest.raises(ValueError, match="amplitude nulle"):
        en_sinus(0, 0)


def test_une_sollicitation_qui_n_est_ni_cosinus_ni_sinus_est_refusee() -> None:
    with pytest.raises(ValueError, match="ni cosinus ni sinus"):
        pour_une_sinusoide(OperateurDifferentiel(D**2 + 1), sp.tan, 2)


def test_un_operateur_nul_est_refuse() -> None:
    """`0 . y = x(t)` n'est pas une equation differentielle."""
    with pytest.raises(ValueError, match="operateur nul"):
        OperateurDifferentiel(D - D)


def test_les_racines_reelles_distinctes_s_ecrivent_en_hyperboliques() -> None:
    """Page 25 : `b = alpha +- w` donne `e^(alpha t) cosh(wt)` et `sinh(wt)`."""
    operateur = OperateurDifferentiel(D**2 - 3 * D + 2)
    alpha, omega = R(3, 2), R(1, 2)
    for forme in (sp.cosh, sp.sinh):
        assert _annule(operateur, exp(alpha * T) * forme(omega * T))
