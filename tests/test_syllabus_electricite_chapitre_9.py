"""Syllabus d'electricite generale, chapitre 9 : flux, inductances, champ dynamique.

Source : `Electromagnetisme/SYLLABUS/Syllabus_Electricite_generale.pdf`,
pages 9-1 a 9-19, lues sur les pages rendues.

Le chapitre etablit les inductances sans les chiffrer. L'oracle est un
systeme dont le champ est connu exactement : deux solenoides longs et
coaxiaux, de meme longueur `l`. Dans chacun l'induction vaut
`mu_0 N I / l` ; le flux par spire de l'un a travers l'autre est limite a la
section du plus etroit. D'ou `L_11 = mu_0 N_1^2 S_1 / l`,
`L_22 = mu_0 N_2^2 S_2 / l` et `L_12 = mu_0 N_1 N_2 S_2 / l` si le second est
a l'interieur du premier.

Ce chapitre est le miroir du chapitre 3 : inductances pour coefficients de
potentiel, flux pour potentiels, courants pour charges. L'identite de
Gauss y est la meme operation, et le test l'emploie telle quelle.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.constantes import PERMEABILITE_DU_VIDE
from polytech.electromagnetisme.domain.electrostatique.coefficients import (
    residu_de_l_identite_de_gauss,
)
from polytech.electromagnetisme.domain.electrostatique.potentiel import circulation
from polytech.electromagnetisme.domain.energie_et_forces import (
    densite_d_energie_magnetique,
    energie_d_une_inductance,
)
from polytech.electromagnetisme.domain.magnetostatique.circuits_magnetiques import (
    inductance_depuis_les_flux,
)
from polytech.electromagnetisme.domain.magnetostatique.couplage import (
    CouplageMagnetique,
    inductance_propre_d_un_solenoide,
)
from polytech.electromagnetisme.domain.magnetostatique.inductances import (
    SystemeDeCircuits,
    flux_a_travers_des_aires,
    flux_d_un_circuit_de_section_finie,
    flux_total,
)
from polytech.electromagnetisme.domain.magnetostatique.induction import (
    champ_induit_dynamique,
    fem_de_deplacement,
    force_electromotrice,
)
from polytech.electromagnetisme.domain.symboles import TEMPS

MU, LONGUEUR = sp.symbols("mu_0 l", positive=True)
N_1, N_2, S_1, S_2 = sp.symbols("N_1 N_2 S_1 S_2", positive=True)
COAXIAUX = SystemeDeCircuits(
    sp.Matrix(
        [
            [MU * N_1**2 * S_1 / LONGUEUR, MU * N_1 * N_2 * S_2 / LONGUEUR],
            [MU * N_1 * N_2 * S_2 / LONGUEUR, MU * N_2**2 * S_2 / LONGUEUR],
        ]
    )
)
"""Deux solenoides coaxiaux, le second (de section S_2) a l'interieur du premier."""


# --- 9.2 Flux a travers un circuit de section finie -----------------------------


def test_le_flux_d_un_circuit_de_section_finie_est_une_moyenne_ponderee() -> None:
    """(9.2.1) : phi = (1 / I) somme des phi_0 dI, sur les tubes elementaires."""
    assert flux_d_un_circuit_de_section_finie([(10, 1), (20, 3)]) == sp.Rational(35, 2)
    assert flux_d_un_circuit_de_section_finie([(7, 2), (7, 5)]) == 7


def test_un_circuit_sans_courant_n_a_pas_de_flux_moyen() -> None:
    """La moyenne (9.2.1) est rapportee au courant total, qui ne peut etre nul."""
    with pytest.raises(ValueError, match="courant total nul"):
        flux_d_un_circuit_de_section_finie([(10, 1), (20, -1)])


# --- 9.6 Equations electromagnetiques ------------------------------------------


def test_les_flux_sont_lineaires_en_les_courants() -> None:
    """(9.6.2) : phi_i = somme des L_ik I_k."""
    i_1, i_2 = sp.symbols("I_1 I_2", real=True)
    phi_1, phi_2 = COAXIAUX.flux([i_1, i_2])
    assert (
        sp.simplify(phi_1 - MU * N_1 * (N_1 * S_1 * i_1 + N_2 * S_2 * i_2) / LONGUEUR)
        == 0
    )
    assert sp.simplify(phi_2 - MU * N_2 * S_2 * (N_1 * i_1 + N_2 * i_2) / LONGUEUR) == 0


def test_les_inductances_propres_recoupent_le_solenoide_du_fascicule() -> None:
    """L_11 = mu_0 N^2 S / l, que `magnetostatique.couplage` portait deja en nombres."""
    valeurs = {MU: PERMEABILITE_DU_VIDE, N_1: 500, S_1: 4e-4, LONGUEUR: 0.25}
    assert math.isclose(
        float(COAXIAUX.inductances[0, 0].subs(valeurs)),
        inductance_propre_d_un_solenoide(500, 4e-4, 0.25),
    )


def test_la_matrice_des_inductances_est_symetrique() -> None:
    """(9.6.6) : L_ij = L_ji, consequence de l'identite de Gauss."""
    with pytest.raises(ValueError, match="inductances non symetriques"):
        SystemeDeCircuits(sp.Matrix([[2, 1], [0, 1]]))


def test_une_inductance_propre_n_est_jamais_negative() -> None:
    """Theoreme de la page 9-11 : « toujours positive ou nulle »."""
    with pytest.raises(ValueError, match="inductance propre negative"):
        SystemeDeCircuits(sp.Matrix([[-1, 0], [0, 1]]))
    assert SystemeDeCircuits(sp.Matrix([[0, 0], [0, 1]])).flux([5, 2]) == sp.Matrix(
        [0, 2]
    )


def test_une_inductance_mutuelle_peut_etre_negative() -> None:
    """Theoreme de la page 9-11, et figure 9.8.3 : L_12, L_23 > 0 mais L_13 < 0."""
    coplanaires = SystemeDeCircuits(
        sp.Matrix([[5, 1, -sp.Rational(1, 4)], [1, 5, 1], [-sp.Rational(1, 4), 1, 5]])
    )
    assert coplanaires.inductances[0, 2] < 0
    assert coplanaires.flux([0, 0, 4])[0] == -1


# --- 9.4 Identite de Gauss -----------------------------------------------------


def test_l_identite_de_gauss_des_circuits() -> None:
    """(9.4.1) : somme des I'_k phi_k = somme des I_k phi'_k."""
    premier, second = [4, 2], [-3, 5]
    residu = residu_de_l_identite_de_gauss(
        premier, COAXIAUX.flux(premier), second, COAXIAUX.flux(second)
    )
    assert sp.simplify(residu) == 0


# --- 9.5 Energie magnetique ----------------------------------------------------


def test_l_energie_magnetique_de_deux_circuits() -> None:
    """(9.5.4) et (9.6.10) : W_m = (L_11 I_1^2 + L_22 I_2^2 + 2 L_12 I_1 I_2) / 2."""
    i_1, i_2 = sp.symbols("I_1 I_2", real=True)
    l_11, l_12, l_22 = (
        COAXIAUX.inductances[0, 0],
        COAXIAUX.inductances[0, 1],
        COAXIAUX.inductances[1, 1],
    )
    attendu = (l_11 * i_1**2 + l_22 * i_2**2 + 2 * l_12 * i_1 * i_2) / 2
    assert sp.simplify(COAXIAUX.energie([i_1, i_2]) - attendu) == 0
    demi_somme = (
        sum(
            i * phi
            for i, phi in zip((i_1, i_2), COAXIAUX.flux([i_1, i_2]), strict=True)
        )
        / 2
    )
    assert sp.simplify(COAXIAUX.energie([i_1, i_2]) - demi_somme) == 0


def test_l_energie_d_un_seul_circuit() -> None:
    """(9.6.11) : phi = L I et W_m = L I^2 / 2."""
    seul = SystemeDeCircuits(sp.Matrix([[sp.Symbol("L", positive=True)]]))
    courant = sp.Symbol("I", real=True)
    assert seul.energie([courant]) == energie_d_une_inductance(
        seul.inductances[0, 0], courant
    )


def test_l_energie_du_solenoide_est_celle_de_son_champ() -> None:
    """(9.5.5) et (9.5.6) : w_m = H . B / 2, fois le volume ou regne le champ."""
    spires, section, longueur, courant = 500, 4e-4, 0.25, 2.0
    champ = spires * courant / longueur
    par_le_champ = (
        densite_d_energie_magnetique(PERMEABILITE_DU_VIDE, champ) * section * longueur
    )
    inductance = inductance_propre_d_un_solenoide(spires, section, longueur)
    assert math.isclose(
        float(energie_d_une_inductance(inductance, courant)), par_le_champ
    )


def test_l_energie_magnetique_n_est_jamais_negative() -> None:
    """(9.3.4) : l'integrale de H . B, donc de B^2 / mu_0, n'est jamais negative."""
    valeurs = {MU: 1, N_1: 3, N_2: 5, S_1: 4, S_2: 1, LONGUEUR: 2}
    for courants in ([1, -1], [1, 1], [-2, 7], [5, -3]):
        assert COAXIAUX.energie(courants).subs(valeurs) > 0
    assert COAXIAUX.energie([0, 0]) == 0


# --- 9.9 Association d'inductances en serie -------------------------------------


def test_deux_inductances_couplees_en_serie() -> None:
    """(9.9.7) : L_1 + L_2 + 2 L_12 en concordance, - 2 L_12 en discordance."""
    courant = sp.Symbol("I", positive=True)
    l_11, l_12, l_22 = (
        COAXIAUX.inductances[0, 0],
        COAXIAUX.inductances[0, 1],
        COAXIAUX.inductances[1, 1],
    )
    concordance = 2 * COAXIAUX.energie([courant, courant]) / courant**2
    discordance = 2 * COAXIAUX.energie([courant, -courant]) / courant**2
    assert sp.simplify(concordance - (l_11 + l_22 + 2 * l_12)) == 0
    assert sp.simplify(discordance - (l_11 + l_22 - 2 * l_12)) == 0


def test_la_serie_recoupe_le_couplage_du_fascicule() -> None:
    """(9.9.7), que `CouplageMagnetique.en_serie` portait deja en nombres."""
    systeme = SystemeDeCircuits(sp.Matrix([[0.5, 0.2], [0.2, 0.3]]))
    couplage = CouplageMagnetique(0.5, 0.3, 0.2)
    assert math.isclose(
        float(2 * systeme.energie([1, 1])), couplage.en_serie(aidant=True)
    )
    assert math.isclose(
        float(2 * systeme.energie([1, -1])), couplage.en_serie(aidant=False)
    )


def test_la_mutuelle_est_bornee_par_la_moyenne_des_inductances_propres() -> None:
    """(9.9.8) : |L_12| <= (L_1 + L_2) / 2, sinon la serie discordante est negative."""
    l_11, l_12, l_22 = (
        COAXIAUX.inductances[0, 0],
        COAXIAUX.inductances[0, 1],
        COAXIAUX.inductances[1, 1],
    )
    valeurs = {MU: 1, N_1: 3, N_2: 5, S_1: 4, S_2: 1, LONGUEUR: 2}
    assert abs(l_12.subs(valeurs)) <= (l_11 + l_22).subs(valeurs) / 2
    with pytest.raises(ValueError, match="inductance mutuelle trop grande"):
        SystemeDeCircuits(sp.Matrix([[1, sp.Rational(3, 2)], [sp.Rational(3, 2), 1]]))
    a_la_limite = SystemeDeCircuits(sp.Matrix([[1, 1], [1, 1]]))
    assert a_la_limite.energie([1, -1]) == 0


# --- 9.10 et 9.11 Flux d'un circuit bobine --------------------------------------


def test_le_flux_total_d_une_bobine() -> None:
    """(9.11.1) : phi = N Phi, Phi etant le flux par spire."""
    assert flux_total(spires=200, par_spire=1.5e-4) == 0.03
    with pytest.raises(
        ValueError, match="nombre de spires strictement positif attendu"
    ):
        flux_total(spires=0, par_spire=1.5e-4)


def test_l_inductance_d_une_bobine_par_son_flux_total() -> None:
    """(9.11.2) : N Phi = L I, que le circuit magnetique du fascicule employait deja."""
    total = flux_total(spires=200, par_spire=1.5e-4)
    assert math.isclose(total / 2.0, inductance_depuis_les_flux(200, (1.5e-4,), 2.0))


def test_le_flux_d_un_circuit_de_forme_quelconque() -> None:
    """Page 9-15 : chaque aire compte autant de fois que le courant en fait le tour."""
    aires = [(1, sp.Rational(3, 2)), (2, 1), (3, sp.Rational(1, 4))]
    assert flux_a_travers_des_aires(2, aires) == 2 * (
        sp.Rational(3, 2) + 2 + sp.Rational(3, 4)
    )


def test_un_contour_parcouru_deux_fois_enlace_deux_fois_le_flux() -> None:
    """9.10, par la circulation du potentiel vecteur (8.6.4) : deux tours, deux flux."""
    induction, rayon = sp.symbols("B a", positive=True)
    x, y = sp.symbols("x y", real=True)
    potentiel = (-induction * y / 2, induction * x / 2, sp.Integer(0))
    cercle = (rayon * sp.cos(TEMPS), rayon * sp.sin(TEMPS), sp.Integer(0))
    deux_tours = circulation(potentiel, cercle, TEMPS, 0, 4 * sp.pi)
    assert deux_tours == flux_a_travers_des_aires(induction, [(2, sp.pi * rayon**2)])


# --- 9.13 et 9.14 Champ et force electromotrice induits dynamiques --------------


def test_le_champ_electrique_induit_dynamique() -> None:
    """(9.13.3) : E_d = v ^ B, v etant la vitesse d'entrainement du conducteur."""
    assert champ_induit_dynamique((2.0, 0.0, 0.0), (0.0, 0.0, 0.5)) == (0.0, -1.0, 0.0)
    assert champ_induit_dynamique((0.0, 0.0, 3.0), (0.0, 0.0, 0.5)) == (0.0, 0.0, 0.0)


def test_la_fem_dynamique_d_un_conducteur_rectiligne() -> None:
    """(9.14.1) et (9.14.4) : la circulation de E_d le long du conducteur vaut B l v."""
    induction, longueur, vitesse = 0.5, 0.2, 2.0
    e_x, e_y, e_z = champ_induit_dynamique((vitesse, 0.0, 0.0), (0.0, 0.0, induction))
    barre = (sp.Integer(0), -TEMPS, sp.Integer(0))
    fem = circulation(
        (sp.Float(e_x), sp.Float(e_y), sp.Float(e_z)), barre, TEMPS, 0, longueur
    )
    assert math.isclose(float(fem), fem_de_deplacement(induction, longueur, vitesse))


def test_la_fem_dynamique_est_l_oppose_du_flux_coupe_par_unite_de_temps() -> None:
    """(9.14.3) : E = -d(phi)/dt, pour une barre qui glisse sur deux rails."""
    induction, longueur, vitesse = sp.symbols("B l v", positive=True)
    flux = induction * longueur * vitesse * TEMPS
    assert force_electromotrice(flux) == -induction * longueur * vitesse
    assert sp.Abs(force_electromotrice(flux)).subs(
        {induction: 0.5, longueur: 0.2, vitesse: 2.0}
    ) == (fem_de_deplacement(0.5, 0.2, 2.0))
