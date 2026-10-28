"""Syllabus de mathematiques appliquees, chapitre 10 : la transformee de Fourier.

Source : `Mathematique/SYLLABUS/Syllabus_Mathematiques_appliquees_2016_2017.pdf`,
pages 170 a 181 et 185 a 186. La definition et son inverse, les spectres,
la porte et la gaussienne (10.5), les proprietes (10.6), l'identite de
Parseval-Plancherel, la resolution d'une equation differentielle (10.8).

Coquilles du syllabus etablies : le signe de l'exponentielle dans les
inverses (10.8) et (10.9) ; `lim 2a sinc(wa) = delta` page 178, auquel il
manque `2 pi` ; « variance » pour l'ecart-type de la gaussienne.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.differentielles.operateurs import D, OperateurDifferentiel
from polytech.maths.domain.fourier.dispersion import ecart_type
from polytech.maths.domain.fourier.proprietes import (
    de_la_derivee,
    de_la_modulation,
    de_la_primitive,
    du_changement_d_echelle,
    du_glissement,
    energie_frequentielle,
    energie_temporelle,
    image_de_la_solution,
)
from polytech.maths.domain.fourier.series import SerieDeFourier
from polytech.maths.domain.fourier.transformee import (
    PULSATION as W,
)
from polytech.maths.domain.fourier.transformee import (
    spectres,
    transformee_de_fourier,
    transformee_inverse,
)
from polytech.maths.domain.laplace.transformee import Transformee
from polytech.maths.domain.symboles import IMAGE as P
from polytech.maths.domain.symboles import TEMPS
from polytech.maths.domain.symboles import VARIABLE as T
from polytech.signaux.domain.symboles import IMAGE as P_DES_SIGNAUX
from polytech.signaux.domain.symboles import PULSATION as W_DES_SIGNAUX
from polytech.signaux.domain.transformees import fourier_depuis_laplace

R = sp.Rational
J = sp.I
A = sp.Symbol("a", positive=True)
GAUSSIENNE = sp.exp(-(A**2) * T**2)
PORTE = sp.Piecewise((1, sp.Abs(T) <= A), (0, True))
CAUSALE = sp.Piecewise((sp.exp(-T), T >= 0), (0, True))


def _egales(obtenue: sp.Expr, attendue: sp.Expr) -> bool:
    """Compare en deux pulsations : sympy laisse des `|w|` qu'il ne simplifie pas."""
    ecart = sp.simplify(obtenue - attendue)
    return all(sp.simplify(ecart.subs(W, valeur)) == 0 for valeur in (1, R(-7, 3)))


def test_la_transformee_de_la_gaussienne() -> None:
    """10.5.1 : `A e^(-a^2 t^2)` devient `(A racine(pi) / a) e^(-w^2 / (4 a^2))`."""
    attendue = 3 * sp.sqrt(sp.pi) / A * sp.exp(-(W**2) / (4 * A**2))
    assert sp.simplify(transformee_de_fourier(3 * GAUSSIENNE) - attendue) == 0


def test_la_gaussienne_etroite_devient_large() -> None:
    """Ecart-type `1/(a racine(2))` en temps, `a racine(2)` en pulsation.

    Le syllabus dit « variance » ; ce sont des ecarts-types. La variance de
    `e^(-a^2 t^2)` est `1/(2 a^2)`.
    """
    image = transformee_de_fourier(GAUSSIENNE)

    assert ecart_type(GAUSSIENNE, T) == 1 / (A * sp.sqrt(2))
    assert sp.simplify(ecart_type(image, W) - A * sp.sqrt(2)) == 0
    assert ecart_type(GAUSSIENNE, T) ** 2 == 1 / (2 * A**2)


def test_la_transformee_de_la_porte() -> None:
    """10.5.2 : `Pi_a` devient `2a sinc(wa) = 2 sin(wa) / w`."""
    assert _egales(transformee_de_fourier(PORTE), 2 * sp.sin(W * A) / W)


def test_la_valeur_en_zero_est_l_integrale_de_la_fonction() -> None:
    """(10.7) : `F(0) = int f(t) dt`."""
    assert sp.limit(transformee_de_fourier(PORTE), W, 0) == 2 * A
    aire = sp.integrate(GAUSSIENNE, (T, -sp.oo, sp.oo))
    assert sp.simplify(transformee_de_fourier(GAUSSIENNE).subs(W, 0) - aire) == 0


def test_la_transformee_inverse_referme_la_boucle() -> None:
    """(10.2) puis (10.3) : `(1/2pi) int F(w) e^(+jwt) dw = f(t)`."""
    decalee = sp.exp(-((T - 1) ** 2))
    image = transformee_de_fourier(decalee)
    assert sp.simplify(transformee_inverse(image) - decalee) == 0


def test_l_inverse_imprimee_en_10_8_a_le_mauvais_signe() -> None:
    """Le syllabus ecrit `e^(-jwt)` dans l'inverse de (10.8) et de (10.9).

    Avec ce signe on retrouve `f(-t)` et non `f(t)` : la gaussienne centree en
    1 revient centree en -1.
    """
    decalee = sp.exp(-((T - 1) ** 2))
    image = transformee_de_fourier(decalee)
    noyau = image * sp.exp(-J * W * T)
    imprimee = sp.simplify(sp.integrate(noyau, (W, -sp.oo, sp.oo)) / (2 * sp.pi))
    assert sp.simplify(imprimee - decalee.subs(T, -T)) == 0
    assert sp.simplify(imprimee - decalee) != 0


def test_les_spectres_d_amplitude_de_phase_et_d_energie() -> None:
    """(10.5) et (10.6) : `F = R + jX = A e^(jb)`, energie `A^2`."""
    image = 1 / (1 + J * W)
    amplitude, phase, energie = spectres(image)
    assert sp.simplify(amplitude - 1 / sp.sqrt(1 + W**2)) == 0
    assert sp.simplify(phase + sp.atan(W)) == 0
    assert sp.simplify(energie - 1 / (1 + W**2)) == 0
    assert sp.simplify(sp.expand_complex(amplitude * sp.exp(J * phase) - image)) == 0


def test_la_serie_de_fourier_de_la_porte_repetee_echantillonne_sa_transformee() -> None:
    """10.2.2 : `c_n = sin(pi n / T) / (pi n) = F(2 pi n / T) / T`."""
    porte_unite = sp.Piecewise((1, sp.Abs(T) <= R(1, 2)), (0, True))
    image = transformee_de_fourier(porte_unite)
    for periode in (2, 4, 8):
        serie = SerieDeFourier(porte_unite, -R(periode, 2), periode)
        for rang in (1, 2, 3):
            coefficient = serie.coefficient(rang)
            assert (
                sp.simplify(
                    coefficient - sp.sin(sp.pi * rang / periode) / (sp.pi * rang)
                )
                == 0
            )
            echantillon = image.subs(W, 2 * sp.pi * rang / periode) / periode
            assert sp.simplify(coefficient - echantillon) == 0


def test_la_limite_de_la_porte_n_est_pas_le_dirac_mais_deux_pi_fois_lui() -> None:
    """Page 178 : `int 2a sinc(wa) dw = 2 pi` pour tout `a`, et non 1.

    Le syllabus ecrit `lim 2a sinc(wa) = delta` ; sa section 10.7.4 donne bien
    `F(w) = 2 pi delta(w)` pour la constante 1.
    """
    masse = sp.integrate(2 * sp.sin(W * A) / W, (W, -sp.oo, sp.oo))
    assert masse == 2 * sp.pi


def test_le_changement_d_echelle() -> None:
    """10.6.2 : `f(at)` devient `F(w/a) / |a|`."""
    image = transformee_de_fourier(sp.exp(-(T**2)))
    for facteur in (2, -3, R(1, 2)):
        directe = transformee_de_fourier(sp.exp(-((facteur * T) ** 2)))
        assert sp.simplify(du_changement_d_echelle(image, facteur) - directe) == 0


def test_le_glissement_en_temps() -> None:
    """10.6.3 : `f(t - t0)` devient `e^(-j w t0) F(w)`."""
    image = transformee_de_fourier(sp.exp(-(T**2)))
    directe = transformee_de_fourier(sp.exp(-((T - 2) ** 2)))
    assert sp.simplify(du_glissement(image, 2) - directe) == 0


def test_la_modulation() -> None:
    """10.6.3 : `f(t) e^(j w0 t)` devient `F(w - w0)`, d'ou les formules en cos et sin.

    Le syllabus ecrit `cos(j w0 t)` et `sin(j w0 t)` ; le `j` y est de trop.
    """
    fonction = sp.exp(-(T**2))
    image = transformee_de_fourier(fonction)
    directe = transformee_de_fourier(fonction * sp.exp(3 * J * T))
    assert sp.simplify(de_la_modulation(image, 3) - directe) == 0
    en_cosinus = transformee_de_fourier(fonction * sp.cos(3 * T))
    somme = (de_la_modulation(image, 3) + de_la_modulation(image, -3)) / 2
    assert _egales(en_cosinus, somme)
    en_sinus = transformee_de_fourier(fonction * sp.sin(3 * T))
    difference = (de_la_modulation(image, 3) - de_la_modulation(image, -3)) / (2 * J)
    assert _egales(en_sinus, difference)


def test_la_derivation_multiplie_par_j_omega() -> None:
    """10.6.4 : `f^(n)` devient `(jw)^n F(w)`."""
    fonction = sp.exp(-(T**2))
    image = transformee_de_fourier(fonction)
    for ordre in (1, 2):
        directe = transformee_de_fourier(sp.diff(fonction, T, ordre))
        assert sp.simplify(de_la_derivee(image, ordre) - directe) == 0


def test_l_integration_divise_par_j_omega() -> None:
    """(10.16) : `int_(-oo)^t f` devient `F / (jw)`, pourvu que `F(0) = 0`."""
    fonction = -2 * T * sp.exp(-(T**2))
    image = transformee_de_fourier(fonction)
    primitive = sp.exp(-(T**2))
    assert sp.simplify(de_la_primitive(image) - transformee_de_fourier(primitive)) == 0


def test_une_fonction_d_integrale_non_nulle_n_a_pas_de_primitive_sommable() -> None:
    """« Ce qui equivaut a `F(0) = 0` » : la gaussienne, d'aire non nulle, y echappe."""
    with pytest.raises(ValueError, match="valeur en zero non nulle"):
        de_la_primitive(transformee_de_fourier(sp.exp(-(T**2))))


def test_la_convolution_devient_un_produit() -> None:
    """10.6.5 : deux gaussiennes convoluees ; le produit des images est l'image."""
    f, g = sp.exp(-(T**2)), sp.exp(-(T**2) / 2)
    muette = sp.Symbol("u", real=True)
    convolution = sp.integrate(
        f.subs(T, muette) * g.subs(T, T - muette), (muette, -sp.oo, sp.oo)
    )
    produit = transformee_de_fourier(f) * transformee_de_fourier(g)
    assert sp.simplify(transformee_de_fourier(sp.simplify(convolution)) - produit) == 0


def test_la_transformee_conserve_la_parite() -> None:
    """10.6.6 : paire donne paire, impaire donne impaire."""
    paire = transformee_de_fourier(sp.exp(-(T**2)))
    impaire = transformee_de_fourier(T * sp.exp(-(T**2)))
    assert sp.simplify(paire.subs(W, -W) - paire) == 0
    assert sp.simplify(impaire.subs(W, -W) + impaire) == 0


def test_le_lemme_de_riemann_lebesgue() -> None:
    """10.6.7 : la transformee d'une fonction sommable tend vers 0 a l'infini."""
    for fonction in (GAUSSIENNE.subs(A, 1), PORTE.subs(A, 1), CAUSALE):
        image = transformee_de_fourier(fonction)
        assert sp.limit(sp.Abs(image), W, sp.oo) == 0


def test_l_identite_de_parseval_plancherel() -> None:
    """10.6.8 : `int |f|^2 dt = (1/2pi) int |F|^2 dw`, la conservation de l'energie."""
    for fonction in (GAUSSIENNE, CAUSALE, PORTE.subs(A, 2)):
        temporelle = energie_temporelle(fonction)
        frequentielle = energie_frequentielle(transformee_de_fourier(fonction))
        assert sp.simplify(temporelle - frequentielle) == 0
    assert energie_temporelle(CAUSALE) == R(1, 2)


def test_la_transformee_de_l_exponentielle_causale() -> None:
    """`e^(-t) H(t)` devient `1 / (1 + jw)` : la transformee de Laplace en `p = jw`."""
    image = transformee_de_fourier(CAUSALE)
    assert sp.simplify(image - 1 / (1 + J * W)) == 0
    laplace = Transformee.de(sp.exp(-TEMPS)).image
    assert sp.simplify(laplace.subs(P, J * W) - image) == 0


def test_le_lien_avec_la_transformee_des_signaux() -> None:
    """Lien : `fourier_depuis_laplace` du cours de signaux rend la meme image."""
    cote_signaux = fourier_depuis_laplace(1 / (P_DES_SIGNAUX + 1))
    image = transformee_de_fourier(CAUSALE)
    assert sp.simplify(cote_signaux.subs(W_DES_SIGNAUX, W) - image) == 0


def test_une_equation_differentielle_devient_algebrique() -> None:
    """10.8 : `P(jw) F(w) = H(w)`. Ici `f' + 2f = e^(-t) H(t)`."""
    operateur = OperateurDifferentiel(D + 2)
    image = image_de_la_solution(operateur, transformee_de_fourier(CAUSALE))
    assert sp.simplify(image - 1 / ((1 + J * W) * (2 + J * W))) == 0
    solution = sp.Piecewise((sp.exp(-T) - sp.exp(-2 * T), T >= 0), (0, True))
    assert sp.simplify(transformee_de_fourier(solution) - image) == 0


def test_l_oscillateur_amorti_du_10_8_1() -> None:
    """`x'' + gamma x' + w0^2 x = f` : `X = F / (w0^2 - w^2 + j gamma w)`."""
    gamma, w0 = sp.symbols("gamma omega_0", positive=True)
    operateur = OperateurDifferentiel(D**2 + gamma * D + w0**2)
    force = sp.Function("F")(W)
    image = image_de_la_solution(operateur, force)
    assert sp.simplify(image - force / (w0**2 - W**2 + J * gamma * W)) == 0


def test_un_oscillateur_sans_amortissement_n_a_pas_d_image_a_la_resonance() -> None:
    """`P(jw)` s'annule en `w = w0` quand `gamma = 0` : l'image n'est plus fonction."""
    with pytest.raises(ValueError, match="le polynome caracteristique s'annule"):
        image_de_la_solution(OperateurDifferentiel(D**2 + 4), sp.Integer(1))


def test_l_inverse_au_signe_moins_rend_f_de_moins_t_et_un_autre_signe_leve() -> None:
    decalee = sp.exp(-((T - 1) ** 2))
    image = transformee_de_fourier(decalee)
    assert sp.simplify(transformee_inverse(image) - decalee) == 0
    assert sp.simplify(transformee_inverse(image, signe=-1) - decalee.subs(T, -T)) == 0
    with pytest.raises(ValueError, match="signe"):
        transformee_inverse(image, signe=2)


def test_l_ecart_type_de_la_gaussienne_et_de_son_image() -> None:
    image = transformee_de_fourier(GAUSSIENNE)
    assert ecart_type(GAUSSIENNE, T) == 1 / (A * sp.sqrt(2))
    assert sp.simplify(ecart_type(image, W) - A * sp.sqrt(2)) == 0
    with pytest.raises(ValueError, match="masse nulle"):
        ecart_type(T * GAUSSIENNE, T)
