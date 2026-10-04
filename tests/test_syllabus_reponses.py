"""Syllabus de mathematiques appliquees, chapitre 3 : convolution et reponses.

Source : `Mathematique/SYLLABUS/Syllabus_Mathematiques_appliquees_2016_2017.pdf`,
pages 60 a 71. Le systeme masse-amortisseur-ressort et les circuits RL et
RLC du 3.6, le produit de convolution (3.7), la reponse impulsionnelle
comme limite de reponses a des creneaux (3.8), l'impulsion de Dirac
(3.8.2), la fonction de Green (3.8.3 et 3.9), l'exemple 3.2.

La fonction de transfert est une notion de signaux et d'electronique :
les tests de lien la prennent la ou elle vit.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.circuits.enclenchement import enclenchement
from polytech.electromagnetisme.domain.symboles import TEMPS as TEMPS_DES_CIRCUITS
from polytech.electronique.domain.laplace import (
    reponse_a_un_echelon,
    reponse_impulsionnelle,
)
from polytech.maths.domain.differentielles.operateurs import D, OperateurDifferentiel
from polytech.maths.domain.differentielles.oscillateurs import (
    masse_amortisseur_ressort,
)
from polytech.maths.domain.laplace.convolution import (
    IMPULSION_DE_DIRAC,
    produit_de_convolution,
)
from polytech.maths.domain.laplace.green import (
    conditions_initiales_equivalentes,
    fonction_de_green,
    reponse,
    reponse_au_creneau,
)
from polytech.maths.domain.laplace.heaviside import par_fractions_simples
from polytech.maths.domain.laplace.ordre_n import ProblemeLineaire
from polytech.maths.domain.laplace.proprietes import de_la_convolution
from polytech.maths.domain.laplace.transformee import Transformee
from polytech.maths.domain.symboles import IMAGE as P
from polytech.maths.domain.symboles import TEMPS as T
from polytech.signaux.domain.symboles import IMAGE as P_DES_SIGNAUX
from polytech.signaux.domain.symboles import TEMPS as TEMPS_DES_SIGNAUX
from polytech.signaux.domain.systemes import convolution
from polytech.signaux.domain.transmittance import Transmittance

R = sp.Rational
exp, sin, cos = sp.exp, sp.sin, sp.cos
MASSE = masse_amortisseur_ressort(1, 3, 2)
CIRCUIT_RL = OperateurDifferentiel(D + 1)
RLC = P / (P**2 + 3 * P + 2)


def test_un_n_est_pas_neutre_pour_la_convolution() -> None:
    """Page 63 : `1 * e^t = e^t - 1`."""
    assert produit_de_convolution(sp.Integer(1), exp(T)) == exp(T) - 1


def test_les_proprietes_du_produit_de_convolution() -> None:
    """Proposition 3.2 : commutatif, distributif sur la somme, associatif."""
    f, g, h = T, exp(-T), sin(T)
    star = produit_de_convolution
    assert sp.simplify(star(f, g) - star(g, f)) == 0
    assert sp.simplify(star(f, g + h) - star(f, g) - star(f, h)) == 0
    assert sp.simplify(star(f, star(g, h)) - star(star(f, g), h)) == 0


def test_l_image_d_une_convolution_est_le_produit_des_images() -> None:
    """Proposition 3.3, sur l'exemple 1 : `L(1 * e^t) = 1/p . 1/(p - 1)`."""
    produit = produit_de_convolution(sp.Integer(1), exp(T))
    attendue = de_la_convolution(1 / P, 1 / (P - 1))
    assert sp.simplify(Transformee.de(produit).image - attendue) == 0


def test_l_original_de_un_sur_p2_p2_plus_un_par_deux_chemins() -> None:
    """Exemple 2, page 64 : `t * sin t = t - sin t`, comme par fractions simples."""
    par_convolution = produit_de_convolution(T, sin(T))
    assert sp.simplify(par_convolution - (T - sin(T))) == 0
    assert (
        sp.simplify(par_fractions_simples(1 / (P**2 * (P**2 + 1))) - (T - sin(T))) == 0
    )


def test_l_impulsion_de_dirac_est_l_element_neutre() -> None:
    """(3.4) : `f * delta = f`, et `L(delta) = 1`."""
    for f in (exp(-T), sin(T), T**2):
        assert produit_de_convolution(f, IMPULSION_DE_DIRAC) == f
        assert produit_de_convolution(IMPULSION_DE_DIRAC, f) == f
    assert Transformee.de(IMPULSION_DE_DIRAC).image == 1


def test_la_convolution_des_maths_est_celle_des_signaux() -> None:
    """Lien : les deux contextes ecrivent la meme integrale, de 0 a `t`."""
    t = TEMPS_DES_SIGNAUX
    cote_signaux = convolution(t, sp.exp(-t)).subs(t, T)
    assert sp.simplify(cote_signaux - produit_de_convolution(T, exp(-T))) == 0


def test_la_fonction_de_green_du_circuit_rl() -> None:
    """3.8.1 : `i' + i = u`, fonction de transfert `1/(p + 1)`, `g = e^(-t)`."""
    assert fonction_de_green(CIRCUIT_RL) == exp(-T)


def test_la_reponse_est_la_convolution_avec_la_fonction_de_green() -> None:
    """3.8.3 : `phi = G * f`. Sur `i' + i = t`, au repos : `t - 1 + e^(-t)`."""
    obtenue = reponse(CIRCUIT_RL, T)
    assert sp.simplify(obtenue - (T - 1 + exp(-T))) == 0
    assert ProblemeLineaire((1, 1), T, (0,)).est_solution(obtenue)


def test_la_reponse_du_circuit_rl_a_un_creneau() -> None:
    """Page 65 : `n (1 - e^(-t))` pendant le creneau, `n e^(-t) (e^(1/n) - 1)` apres."""
    n = sp.Symbol("n", positive=True)
    pendant, apres = reponse_au_creneau(CIRCUIT_RL, n)
    assert sp.simplify(pendant - n * (1 - exp(-T))) == 0
    assert sp.simplify(apres - n * exp(-T) * (exp(1 / n) - 1)) == 0


def test_la_reponse_impulsionnelle_est_la_limite_des_reponses_aux_creneaux() -> None:
    """Page 66 : `lim n e^(-t) (e^(1/n) - 1) = e^(-t)` ; et de meme a l'ordre deux."""
    n = sp.Symbol("n", positive=True)
    for operateur in (CIRCUIT_RL, MASSE):
        _, apres = reponse_au_creneau(operateur, n)
        limite = sp.limit(apres, n, sp.oo)
        assert sp.simplify(limite - fonction_de_green(operateur)) == 0


def test_les_deux_branches_du_creneau_se_raccordent() -> None:
    """A `t = 1/n`, la reponse est continue."""
    pendant, apres = reponse_au_creneau(MASSE, 4)
    assert sp.simplify(pendant.subs(T, R(1, 4)) - apres.subs(T, R(1, 4))) == 0


def test_la_fonction_de_green_sans_impulsion_de_dirac() -> None:
    """3.9 b : `g` resout `K(D) y = 0` avec `y^(n-1)(0) = 1/a_n`, le reste nul."""
    assert conditions_initiales_equivalentes(CIRCUIT_RL) == (1,)
    assert conditions_initiales_equivalentes(MASSE) == (0, 1)
    double = OperateurDifferentiel(2 * D**2 + 6 * D + 4)
    assert conditions_initiales_equivalentes(double) == (0, R(1, 2))
    for operateur, coefficients in ((MASSE, (1, 3, 2)), (double, (2, 6, 4))):
        libre = ProblemeLineaire(
            coefficients, sp.Integer(0), conditions_initiales_equivalentes(operateur)
        )
        assert libre.est_solution(fonction_de_green(operateur))


def test_la_fonction_de_green_n_est_pas_derivable_en_zero() -> None:
    """Page 69 : pour `g = e^(-t)`, `g'(+0) = -1` alors que `g'(-0) = 0`."""
    g = fonction_de_green(CIRCUIT_RL)
    assert sp.limit((g - g.subs(T, 0)) / T, T, 0, "+") == -1


def test_la_reponse_indicielle_du_systeme_masse_amortisseur_ressort() -> None:
    """3.6.3 a, `m = 1`, `C = 3`, `K = 2` : `1/2 + e^(-2t)/2 - e^(-t)`, regime 1/2."""
    obtenue = reponse(MASSE, sp.Integer(1))
    assert sp.simplify(obtenue - (R(1, 2) + exp(-2 * T) / 2 - exp(-T))) == 0
    assert sp.limit(obtenue, T, sp.oo) == R(1, 2)


def test_la_reponse_du_systeme_a_un_creneau_d_une_seconde() -> None:
    """3.6.3 b : apres `t = 1`, `-e^(-t) + e^(-2t)/2 + e^(-(t-1)) - e^(-2(t-1))/2`."""
    pendant, apres = reponse_au_creneau(MASSE, 1)
    indicielle = R(1, 2) - exp(-T) + exp(-2 * T) / 2
    attendue = -exp(-T) + exp(-2 * T) / 2 + exp(-(T - 1)) - exp(-2 * (T - 1)) / 2
    assert sp.simplify(pendant - indicielle) == 0
    assert sp.simplify(apres - attendue) == 0
    assert sp.limit(apres, T, sp.oo) == 0


def test_on_allume_le_circuit_rl_en_continu() -> None:
    """3.6.4 a : `u = L i' + R i`, `u = 1` : `i = (1 - e^(-Rt/L)) / R`, regime `1/R`."""
    resistance, inductance = sp.symbols("R L", positive=True)
    circuit = OperateurDifferentiel(inductance * D + resistance)
    courant = reponse(circuit, sp.Integer(1))
    attendu = (1 - exp(-resistance * T / inductance)) / resistance
    assert sp.simplify(courant - attendu) == 0
    assert sp.limit(courant, T, sp.oo) == 1 / resistance


def test_l_enclenchement_des_circuits_donne_le_meme_courant() -> None:
    """Lien : regime `1/R`, courant initial nul, constante de temps `L/R`."""
    resistance, inductance = sp.symbols("R L", positive=True)
    cote_circuits = enclenchement(
        1 / resistance, sp.Integer(0), inductance / resistance
    )
    cote_maths = reponse(
        OperateurDifferentiel(inductance * D + resistance), sp.Integer(1)
    )
    assert sp.simplify(cote_circuits.subs(TEMPS_DES_CIRCUITS, T) - cote_maths) == 0


def test_on_allume_le_circuit_rl_en_alternatif() -> None:
    """3.6.4 b, avec `R = 3`, `L = 2`, `w = 5` dans la formule de la page 62."""
    resistance, inductance, w = 3, 2, 5
    rapport = R(resistance, inductance)
    courant = reponse(OperateurDifferentiel(inductance * D + resistance), sin(w * T))
    crochet = (exp(-rapport * T) - cos(w * T) + rapport / w * sin(w * T)) / (
        w**2 + rapport**2
    )
    assert sp.simplify(courant - R(w, inductance) * crochet) == 0


def test_la_fonction_de_transfert_du_circuit_rlc() -> None:
    """3.6.5 et exemple 3.2 : `Cp / (LCp^2 + RCp + 1)`, soit `p / (p^2 + 3p + 2)`."""
    inductance, resistance, capacite = 1, 3, R(1, 2)
    transfert = (
        capacite * P / (inductance * capacite * P**2 + resistance * capacite * P + 1)
    )
    assert sp.simplify(transfert - RLC) == 0


def test_les_reponses_impulsionnelle_et_indicielle_de_l_exemple_3_2() -> None:
    """`-e^(-t) + 2e^(-2t)` et `e^(-t) - e^(-2t)` ; la premiere derive la seconde."""
    impulsionnelle = par_fractions_simples(RLC)
    indicielle = par_fractions_simples(RLC / P)
    assert sp.simplify(impulsionnelle - (-exp(-T) + 2 * exp(-2 * T))) == 0
    assert sp.simplify(indicielle - (exp(-T) - exp(-2 * T))) == 0
    assert sp.simplify(sp.diff(indicielle, T) - impulsionnelle) == 0


def test_les_reponses_de_l_electronique_sont_les_memes() -> None:
    """Lien : les reponses impulsionnelle et indicielle du cours d'electronique."""
    cote_electronique = reponse_impulsionnelle(RLC, P, T)
    assert sp.simplify(cote_electronique - par_fractions_simples(RLC)) == 0
    echelon = reponse_a_un_echelon(RLC, sp.Integer(1), P, T)
    assert sp.simplify(echelon - par_fractions_simples(RLC / P)) == 0


def test_la_reponse_du_circuit_rlc_a_un_sinus() -> None:
    """Exemple 3.2.4 : `-e^(-t)/2 + 2e^(-2t)/5`, plus `0,3162 cos(t - 1,249)`."""
    courant = par_fractions_simples(RLC / (P**2 + 1))
    transitoire = -exp(-T) / 2 + R(2, 5) * exp(-2 * T)
    regime = sp.simplify(courant - transitoire)
    assert not regime.has(sp.exp)
    for instant in (0, 1, R(5, 2)):
        attendu = 0.3162 * sp.cos(instant - 1.249)
        assert float(regime.subs(T, instant)) == pytest.approx(float(attendu), abs=1e-4)


def test_le_regime_se_lit_sur_la_transmittance_des_signaux() -> None:
    """Lien : `|G(j)| = racine(10)/10 = 0,3162`, et `arg G(j) = pi/2 - 1,249`."""
    transmittance = Transmittance(expression=RLC.subs(P, P_DES_SIGNAUX))
    assert transmittance.poles() == (-2, -1)
    en_j = transmittance.reponse_frequentielle().subs(
        sp.Symbol("omega", positive=True), 1
    )
    assert sp.Abs(en_j) == sp.sqrt(10) / 10
    assert float(sp.pi / 2 - sp.arg(en_j)) == pytest.approx(1.249, abs=5e-4)
