"""Les planches d'electronique : capteurs et processus aleatoires (chapitre 4, I).

Source : `ElectroniqueInstrumentation/SLIDES/Slides_4_part1.pdf`, 61
planches lues sur leur couche de texte.

Les oracles : l'anemometre a fil chaud `I^2 = A' + B' U^n` (planche 10), la
meilleure droite `beta_1' = Cov[X, Y] / Var[X]` (planche 23), `K_YY = A
K_XX A^H` (planche 24), la loi conditionnelle d'un processus gaussien
stationnaire (planche 33), le theoreme de Wiener-Khintchine (planche 47)
et l'intervalle a 99,9 % `[-3,2905 sigma, 3,2905 sigma]` (planche 53).

Les notions de probabilite vivent dans `polytech.maths`, celles de
processus aleatoire dans `polytech.signaux` : l'electronique les emploie.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electronique.domain.acquisition import (
    autocorrelation_d_un_bruit_blanc,
    autocovariance,
    densite_spectrale_unilaterale_depuis_bilaterale,
    variance_d_un_bruit_borne,
)
from polytech.electronique.domain.bruit_gaussien import (
    excursion_crete_a_crete,
    valeur_efficace_de_la_somme,
)
from polytech.electronique.domain.capteur import (
    FilChaud,
    erreur_d_estimation,
    sensibilite,
)
from polytech.maths.domain.fourier.transformee import PULSATION, transformee_de_fourier
from polytech.maths.domain.probabilites.covariance import (
    LoiConjointe,
    covariance_apres_transformation,
)
from polytech.maths.domain.probabilites.loi_normale import (
    demi_largeur_de_l_intervalle,
    densite_normale,
    densite_normale_multivariee,
    loi_conditionnelle,
)
from polytech.maths.domain.symboles import VARIABLE
from polytech.signaux.domain.processus_aleatoires import (
    autocorrelation_depuis_la_densite,
    densite_de_la_somme,
    densite_depuis_l_autocorrelation,
    densite_en_sortie,
    puissance_dans_la_bande,
)

X = sp.Symbol("X", positive=True)
F = sp.Symbol("f", real=True)
TAU = sp.Symbol("tau", real=True)


def test_la_sensibilite_est_la_pente_de_la_caracteristique() -> None:
    """Planche 19 : `S = df/dX` en `X_0` ; `a` partout pour `Y = a X + N`."""
    assert sensibilite(7 * X, X, 3) == 7
    assert sensibilite(7 * X, X, 11) == 7
    assert sensibilite(X**2, X, 3) == 6
    assert sensibilite(sp.sqrt(X), X, 4) == sp.Rational(1, 4)


def test_une_forte_sensibilite_ecrase_l_effet_du_bruit() -> None:
    """Planche 20 : `(f^-1)'(a) = 1 / f'(f^-1(a))`, donc `dX ~ N / S`."""
    caracteristique = X**2
    for point, bruit in ((1.0, 0.01), (5.0, 0.01)):
        pente = float(sensibilite(caracteristique, X, point))
        estimee = math.sqrt(point**2 + bruit)
        attendue = erreur_d_estimation(bruit, pente)
        assert estimee - point == pytest.approx(attendue, rel=1e-2)
    assert erreur_d_estimation(0.01, 10.0) < erreur_d_estimation(0.01, 2.0)
    with pytest.raises(ValueError, match="sensibilite nulle"):
        erreur_d_estimation(0.01, 0.0)


def test_l_anemometre_a_fil_chaud_n_est_pas_lineaire() -> None:
    """Planche 10 : `I^2 = A' + B' U^n`, que l'unite numerique inverse."""
    fil = FilChaud(ordonnee=0.04, pente=0.02, exposant=0.5)
    courant = fil.courant(9.0)
    assert courant**2 == pytest.approx(0.04 + 0.02 * 3.0)
    assert fil.vitesse(courant) == pytest.approx(9.0)
    assert fil.sensibilite(1.0) > 2 * fil.sensibilite(16.0)
    with pytest.raises(
        ValueError, match="courant inferieur au courant a vitesse nulle"
    ):
        fil.vitesse(0.1)


def test_la_densite_normale() -> None:
    """Planche 17 : `exp(-(x - mu)^2 / (2 sigma^2)) / (racine(2 pi) sigma)`."""
    x = sp.Symbol("x", real=True)
    densite = densite_normale(x, 2, 9)
    assert sp.integrate(densite, (x, -sp.oo, sp.oo)) == 1
    assert sp.integrate(x * densite, (x, -sp.oo, sp.oo)) == 2
    assert sp.simplify(sp.integrate((x - 2) ** 2 * densite, (x, -sp.oo, sp.oo))) == 9
    assert densite.subs(x, 2) == 1 / (3 * sp.sqrt(2 * sp.pi))
    with pytest.raises(ValueError, match="variance nulle ou negative"):
        densite_normale(x, 0, 0)


LOI = LoiConjointe(
    couples=((0, 1), (1, 3), (2, 2), (3, 6)),
    probabilites=(
        sp.Rational(1, 4),
        sp.Rational(1, 4),
        sp.Rational(1, 8),
        sp.Rational(3, 8),
    ),
)


def test_la_covariance_donne_la_meilleure_droite() -> None:
    """Planche 23 : `beta_1' = Cov[X, Y] / Var[X]`, `beta_0' = E[Y] - beta_1' E[X]`."""
    b0, b1 = sp.symbols("beta_0 beta_1")
    ecart = sum(
        p * (y - (b1 * x + b0)) ** 2
        for (x, y), p in zip(LOI.couples, LOI.probabilites, strict=True)
    )
    optimum = sp.solve([sp.diff(ecart, b0), sp.diff(ecart, b1)], [b0, b1])
    ordonnee, pente = LOI.meilleure_droite()
    assert (ordonnee, pente) == (optimum[b0], optimum[b1])
    assert pente == LOI.covariance() / LOI.variances()[0]
    esperance_x, esperance_y = LOI.esperances()
    assert ordonnee == esperance_y - pente * esperance_x
    assert (esperance_x, esperance_y) == (sp.Rational(13, 8), sp.Rational(7, 2))
    assert LOI.covariance() == 8 - sp.Rational(13, 8) * sp.Rational(7, 2)
    with pytest.raises(ValueError, match="les probabilites ne somment pas a un"):
        LoiConjointe(couples=((0, 0), (1, 1)), probabilites=(0.5, 0.4))


def test_la_matrice_de_covariance_se_transforme_par_congruence() -> None:
    """Planche 24 : si `Y = A X + b`, alors `K_YY = A K_XX A^H`."""
    matrice = LOI.matrice_de_covariance()
    assert matrice == matrice.T
    assert matrice[0, 1] == LOI.covariance()
    assert (matrice[0, 0], matrice[1, 1]) == LOI.variances()
    transformation = sp.Matrix([[1, 1], [2, -1]])
    transformes = tuple((x + y + 5, 2 * x - y - 3) for x, y in LOI.couples)
    image = LoiConjointe(couples=transformes, probabilites=LOI.probabilites)
    attendue = covariance_apres_transformation(transformation, matrice)
    assert image.matrice_de_covariance() == attendue
    complexe = sp.Matrix([[sp.I, 0], [0, 1]])
    assert covariance_apres_transformation(complexe, matrice)[0, 0] == matrice[0, 0]


def test_la_gaussienne_a_deux_dimensions() -> None:
    """Planche 25 : `exp(-(x - mu)^T S^-1 (x - mu) / 2) / racine((2 pi)^2 det S)`."""
    x1, x2 = sp.symbols("x_1 x_2", real=True)
    diagonale = sp.Matrix([[4, 0], [0, 9]])
    densite = densite_normale_multivariee((x1, x2), (1, -2), diagonale)
    produit = densite_normale(x1, 1, 4) * densite_normale(x2, -2, 9)
    assert sp.simplify(densite - produit) == 0
    correlee = sp.Matrix([[1, sp.Rational(1, 2)], [sp.Rational(1, 2), 1]])
    jointe = densite_normale_multivariee((x1, x2), (0, 0), correlee)
    marginale = sp.integrate(jointe, (x2, -sp.oo, sp.oo))
    assert sp.simplify(marginale - densite_normale(x1, 0, 1)) == 0
    with pytest.raises(ValueError, match="matrice de covariance degeneree"):
        densite_normale_multivariee((x1, x2), (0, 0), sp.Matrix([[1, 1], [1, 1]]))


def test_l_autocovariance_et_l_autocorrelation() -> None:
    """Planche 30 : `K_XX = R_XX - E[X(t_1)] E[X(t_2)]` ; egales a moyenne nulle."""
    assert autocovariance(5.0, 2.0) == pytest.approx(1.0)
    assert autocovariance(5.0, 0.0) == 5.0


def test_la_loi_conditionnelle_d_un_processus_gaussien_stationnaire() -> None:
    """Planche 33 : `X(t + tau) | X(t) = a ~ N(rho a, (1 - rho^2) sigma^2)`."""
    x1, x2 = sp.symbols("x_1 x_2", real=True)
    rho, variance = sp.Rational(3, 5), sp.Integer(4)
    moyenne, conditionnelle = loi_conditionnelle(x1, rho, variance)
    assert (moyenne, conditionnelle) == (rho * x1, (1 - rho**2) * variance)
    covariance = sp.Matrix([[variance, rho * variance], [rho * variance, variance]])
    jointe = densite_normale_multivariee((x1, x2), (0, 0), covariance)
    quotient = jointe / densite_normale(x1, 0, variance)
    attendue = densite_normale(x2, moyenne, conditionnelle)
    assert sp.simplify(quotient - attendue) == 0
    assert loi_conditionnelle(x1, 0, variance) == (0, variance)
    assert loi_conditionnelle(x1, 1, variance) == (x1, 0)
    with pytest.raises(ValueError, match="correlation hors de"):
        loi_conditionnelle(x1, sp.Rational(3, 2), variance)


def test_les_proprietes_de_la_densite_spectrale() -> None:
    """Planche 41 : paire, additive pour des processus independants, `|H|^2 S`."""
    premiere = 3 / (1 + F**2)
    seconde = sp.Integer(2)
    assert densite_de_la_somme(premiere, seconde) == premiere + seconde
    assert premiere.subs(F, -F) == premiere
    reponse = 1 / (1 + sp.I * F / 10)
    sortie = densite_en_sortie(reponse, seconde)
    assert sp.simplify(sortie - 2 / (1 + F**2 / 100)) == 0


def test_la_densite_unilaterale_donne_la_puissance_d_une_bande() -> None:
    """Planches 44 a 46 : `P = 2 int S df = int S_os df` sur `(f_l, f_h)`."""
    bilaterale = 3 / (1 + F**2)
    puissance = puissance_dans_la_bande(bilaterale, F, (1, sp.sqrt(3)))
    assert sp.simplify(puissance - 6 * (sp.pi / 3 - sp.pi / 4)) == 0
    blanche = puissance_dans_la_bande(sp.Integer(5), F, (100, 300))
    unilaterale = densite_spectrale_unilaterale_depuis_bilaterale(5.0)
    assert blanche == pytest.approx(variance_d_un_bruit_borne(unilaterale, 200.0))
    with pytest.raises(ValueError, match="bande vide ou inversee"):
        puissance_dans_la_bande(bilaterale, F, (3, 1))


def test_le_theoreme_de_wiener_khintchine() -> None:
    """Planche 47 : `S(f) = int R(tau) e^(-j 2 pi f tau) dtau`, et l'inverse."""
    autocorrelation = 4 * sp.exp(-sp.Abs(TAU) / 2)
    densite = densite_depuis_l_autocorrelation(autocorrelation, TAU, F)
    attendue = 16 / (1 + (4 * sp.pi * F) ** 2)
    assert sp.simplify(densite - attendue) == 0
    retour = autocorrelation_depuis_la_densite(densite, F, TAU)
    for retard in (0, 1, -3):
        ecart = retour.subs(TAU, retard) - autocorrelation.subs(TAU, retard)
        assert sp.simplify(ecart) == 0
    variance = sp.integrate(densite, (F, -sp.oo, sp.oo))
    assert variance == autocorrelation.subs(TAU, 0)
    par_les_maths = transformee_de_fourier(autocorrelation.subs(TAU, VARIABLE))
    assert sp.simplify(par_les_maths.subs(PULSATION, 2 * sp.pi * F) - densite) == 0


def test_le_bruit_blanc_a_une_densite_plate() -> None:
    """Planches 55 et 56 : `R(tau) = R_0 delta(tau)`, donc `S(f) = R_0`."""
    assert densite_depuis_l_autocorrelation(7 * sp.DiracDelta(TAU), TAU, F) == 7
    assert autocorrelation_d_un_bruit_blanc(7.0, 0.0) == 7.0
    assert autocorrelation_d_un_bruit_blanc(7.0, 1e-9) == 0.0


def test_les_variances_de_bruits_independants_s_ajoutent() -> None:
    """Planches 50 et 51 : les variances s'ajoutent, et `V_RMS = sigma_n`."""
    assert valeur_efficace_de_la_somme(3e-6, 4e-6) == pytest.approx(5e-6)
    x = sp.Symbol("x", real=True)
    sigma = sp.Symbol("sigma", positive=True)
    densite = densite_normale(x, 0, sigma**2)
    carre_moyen = sp.integrate(x**2 * densite, (x, -sp.oo, sp.oo))
    assert sp.simplify(carre_moyen - sigma**2) == 0
    with pytest.raises(ValueError, match="valeur efficace negative"):
        valeur_efficace_de_la_somme(3e-6, -4e-6)


def test_l_excursion_crete_a_crete_d_un_bruit_gaussien() -> None:
    """Planches 52 et 53 : 99,9 % dans `+-3,2905 sigma`, soit 6,6 `V_RMS`."""
    assert demi_largeur_de_l_intervalle(0.999) == pytest.approx(3.2905, abs=5e-5)
    assert demi_largeur_de_l_intervalle(0.6827) == pytest.approx(1.0, abs=1e-3)
    assert demi_largeur_de_l_intervalle(0.9545) == pytest.approx(2.0, abs=1e-3)
    assert excursion_crete_a_crete(1e-3) == pytest.approx(6.6e-3, rel=3e-3)
    assert excursion_crete_a_crete(1e-3, probabilite=0.9973) == pytest.approx(
        6e-3, rel=1e-3
    )
    with pytest.raises(ValueError, match="probabilite hors de"):
        demi_largeur_de_l_intervalle(1.0)


def test_les_gardes_des_lois_et_des_densites() -> None:
    """Une loi mal formee, une droite sans pente, une bande negative sont refusees."""
    with pytest.raises(ValueError, match="autant de probabilites que de couples"):
        LoiConjointe(couples=((0, 0), (1, 1)), probabilites=(1,))
    with pytest.raises(ValueError, match="probabilite negative"):
        LoiConjointe(
            couples=((0, 0), (1, 1)),
            probabilites=(sp.Rational(3, 2), -sp.Rational(1, 2)),
        )
    constante = LoiConjointe(
        couples=((1, 0), (1, 4)), probabilites=(sp.Rational(1, 2),) * 2
    )
    with pytest.raises(ValueError, match="variance de X nulle"):
        constante.meilleure_droite()
    with pytest.raises(ValueError, match="dimensions incompatibles"):
        covariance_apres_transformation(sp.eye(3), sp.eye(2))
    with pytest.raises(ValueError, match="aucune densite"):
        densite_de_la_somme()
    with pytest.raises(ValueError, match="bande vide ou inversee"):
        puissance_dans_la_bande(sp.Integer(1), F, (-2, 1))


def test_la_sensibilite_du_fil_chaud_est_la_pente_de_sa_caracteristique() -> None:
    """`dI/dU = n B' U^(n - 1) / (2 I)`, recoupee par une difference finie."""
    fil = FilChaud(ordonnee=0.04, pente=0.02, exposant=0.5)
    pas = 1e-6
    finie = (fil.courant(4.0 + pas) - fil.courant(4.0 - pas)) / (2 * pas)
    assert fil.sensibilite(4.0) == pytest.approx(finie, rel=1e-6)
    with pytest.raises(ValueError, match="coefficients invalides"):
        FilChaud(ordonnee=0.04, pente=0.0, exposant=0.5)
    with pytest.raises(ValueError, match="vitesse negative"):
        fil.courant(-1.0)
    with pytest.raises(ValueError, match="valeur efficace negative"):
        excursion_crete_a_crete(-1e-3)
