"""Planches de mecanique des fluides, chapitres VII et VIII.

Source : `MecaniqueFluides/SLIDES/Slides_07_Lubrification_Ecoulement_turbulent.pdf`
(35 planches), lues sur les pages rendues.

Les oracles du cours : `0,0375` pour l'hypothese de lubrification
(planche 13), `10^7 Pa` de pression maximale et la courbe `h_L / h_0 = 0,5`
qui culmine a 1 (planche 14), `0,01 mm` pour l'echelle de Kolmogorov
(planche 23), et la loi de paroi `u+ = 2,5 ln y+ + 5,0` (planche 34).
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.fluides.domain.lubrification import (
    Patin,
    critere_de_lubrification,
    residu_de_reynolds,
    vitesse_dans_le_film,
)
from polytech.fluides.domain.paroi_turbulente import (
    contrainte_reduite_en_conduite,
    deficit_de_vitesse,
    echelle_de_kolmogorov,
    energie_cinetique_turbulente,
    tenseur_de_reynolds,
)
from polytech.fluides.domain.regime_de_reynolds import nombre_de_reynolds
from polytech.fluides.domain.similitude import (
    nombre_de_froude,
    rapport_des_viscosites_en_similitude_complete,
)
from polytech.fluides.domain.turbulence import (
    CONSTANTE_LOG,
    KARMAN,
    Decomposition,
    covariance,
    vitesse_adimensionnee,
)


def test_la_similitude_complete_lie_viscosites_et_echelles() -> None:
    """Planche 3 : `Re` et `Fr` egaux, donc `(nu_1/nu_2)^2 (L_2/L_1)^3 g_2/g_1 = 1`."""
    rapport = rapport_des_viscosites_en_similitude_complete(echelle=0.1)
    longueur, vitesse, visqueux = 2.0, 3.0, 1e-6
    maquette = (0.1 * longueur, vitesse * math.sqrt(0.1), rapport * visqueux)
    assert nombre_de_froude(vitesse, longueur) == pytest.approx(
        nombre_de_froude(maquette[1], maquette[0])
    )
    assert nombre_de_reynolds(vitesse, longueur, visqueux) == pytest.approx(
        nombre_de_reynolds(maquette[1], maquette[0], maquette[2])
    )
    assert (1 / rapport) ** 2 * 0.1**3 == pytest.approx(1.0)


def test_les_gardes_de_la_similitude() -> None:
    with pytest.raises(ValueError, match="echelle nulle ou negative"):
        rapport_des_viscosites_en_similitude_complete(echelle=0.0)
    with pytest.raises(ValueError, match="longueur nulle ou negative"):
        nombre_de_froude(3.0, 0.0)


def test_l_hypothese_de_lubrification_de_la_planche_13() -> None:
    """`Re_L h^2 / L^2 = 0,0375` : huile SAE50 a 60 degres, 10 cm, 0,5 mm, 1 m/s."""
    critere = critere_de_lubrification(
        masse_volumique=900.0,
        viscosite=60e-3,
        vitesse=1.0,
        longueur=0.1,
        epaisseur=0.5e-3,
    )
    assert critere == pytest.approx(0.0375)


Y, H, U, MU, PENTE = sp.symbols("y h U mu G", real=True)


def test_le_profil_du_film_resout_l_equation_de_reynolds() -> None:
    """Planche 12 : `dp/dx = mu d2u/dy2`, `u(0) = U`, `u(h) = 0`."""
    profil = vitesse_dans_le_film(Y, H, U, MU, PENTE)
    assert residu_de_reynolds(profil, Y, MU, PENTE) == 0
    assert profil.subs(Y, 0) == U
    assert sp.simplify(profil.subs(Y, H)) == 0


PATIN = Patin(longueur=0.1, entree=0.1e-3, sortie=0.05e-3, vitesse=10.0, viscosite=0.1)
"""Planche 14 : huile SAE50 a 50 degres."""


def test_la_pression_maximale_de_la_planche_14_vaut_dix_megapascals() -> None:
    """`p_max - p_0 = (3 mu U L / (2 h_0 h_L)) (h_0 - h_L) / (h_0 + h_L) = 10^7 Pa`."""
    assert PATIN.pression_maximale() == pytest.approx(1e7)
    sommet = PATIN.epaisseur_du_maximum()
    assert sommet == pytest.approx(2 * 0.1e-3 * 0.05e-3 / 0.15e-3)
    assert PATIN.surpression(sommet) == pytest.approx(1e7)
    assert PATIN.surpression(1.01 * sommet) < 1e7 > PATIN.surpression(0.99 * sommet)


def test_la_surpression_s_annule_aux_deux_bouts_du_patin() -> None:
    assert PATIN.surpression(0.1e-3) == 0
    assert PATIN.surpression(0.05e-3) == 0


def test_la_formule_de_la_planche_12_oublie_h_carre() -> None:
    """Elle imprime `6 mu U L (h_0 - h)(h - h_L) / (h_0^2 - h_L^2)`, sans `h^2`.

    Telle quelle, elle n'est pas une pression, et ne rend pas les `10^7 Pa`
    de la planche 14 : il manque `h^2` au denominateur.
    """
    sommet = PATIN.epaisseur_du_maximum()
    imprimee = 6 * 0.1 * 10.0 * 0.1 * (0.1e-3 - sommet) * (sommet - 0.05e-3)
    imprimee /= 0.1e-3**2 - 0.05e-3**2
    assert imprimee == pytest.approx(PATIN.surpression(sommet) * sommet**2)
    assert imprimee < 1.0


def test_la_surpression_integre_le_debit_constant_du_film() -> None:
    """Un second chemin : `dp/dx = 6 mu (U h - 2 q) / h^3`, a debit `q` constant."""
    debit = PATIN.debit()
    assert debit == pytest.approx(10.0 * 0.1e-3 * 0.05e-3 / 0.15e-3)
    pas, pression, epaisseur = 20_000, 0.0, 0.1e-3
    for k in range(pas // 2):
        milieu = 0.1e-3 - 0.05e-3 * (k + 0.5) / pas
        pente = 6 * 0.1 * (10.0 * milieu - 2 * debit) / milieu**3
        pression += pente * 0.1 / pas
        epaisseur = 0.1e-3 - 0.05e-3 * (k + 1) / pas
    assert pression == pytest.approx(PATIN.surpression(epaisseur), rel=1e-6)


def test_les_courbes_reduites_de_la_planche_14() -> None:
    """`(p - p_0) h_0^2 / (mu U L)` culmine a 1 pour `h_L / h_0 = 0,5`, 2,7 pour 0,3."""
    for rapport, sommet in ((0.5, 1.0), (0.3, 2.69)):
        patin = Patin(1.0, 1.0, rapport, 1.0, 1.0)
        assert patin.pression_maximale() == pytest.approx(sommet, abs=0.01)


def test_un_patin_qui_ne_converge_pas_est_refuse() -> None:
    with pytest.raises(ValueError, match="film non convergent"):
        Patin(0.1, 0.05e-3, 0.1e-3, 10.0, 0.1)
    with pytest.raises(ValueError, match="hors du film"):
        PATIN.surpression(0.2e-3)


def test_l_echelle_de_kolmogorov_du_melangeur_de_la_planche_23() -> None:
    """`eta = (nu^3 / epsilon)^(1/4) = 0,01 mm` pour 100 W/kg dans l'eau."""
    assert echelle_de_kolmogorov(1e-6, 100.0) == pytest.approx(1e-5)
    with pytest.raises(ValueError, match="dissipation nulle ou negative"):
        echelle_de_kolmogorov(1e-6, 0.0)


def test_les_regles_sur_les_moyennes_de_la_planche_26() -> None:
    """La moyenne d'une moyenne est la moyenne ; celle d'une fluctuation est nulle."""
    signal = Decomposition((1.0, 4.0, 2.0, 5.0))
    constante = Decomposition((signal.moyenne(),) * 4)
    assert constante.moyenne() == signal.moyenne()
    assert Decomposition(signal.fluctuations()).moyenne() == pytest.approx(0.0)
    assert signal.variance() == pytest.approx(signal.ecart_type() ** 2)


def test_le_tenseur_de_reynolds_et_l_energie_cinetique_turbulente() -> None:
    """Planche 28 : `R_ij = -rho u_i' u_j'`, `k = -R_kk / (2 rho)`."""
    composantes = (
        Decomposition((1.0, 3.0, 1.0, 3.0)),
        Decomposition((0.0, 2.0, 2.0, 0.0)),
        Decomposition((5.0, 5.0, 5.0, 5.0)),
    )
    tenseur = tenseur_de_reynolds(1000.0, composantes)
    assert tenseur == tenseur.T
    assert tenseur[0, 0] == pytest.approx(-1000.0)
    assert tenseur[0, 1] == pytest.approx(-1000.0 * covariance(*composantes[:2]))
    assert float(tenseur[2, 2]) == pytest.approx(0.0)
    energie = energie_cinetique_turbulente(composantes)
    assert energie == pytest.approx(1.0)
    assert energie == pytest.approx(-float(tenseur.trace()) / (2 * 1000.0))


def test_la_contrainte_totale_d_une_conduite_est_lineaire() -> None:
    """Planche 31 : `tau / tau_w = 1 - (R - r) / R`, nulle sur l'axe."""
    assert contrainte_reduite_en_conduite(0.0, 0.05) == 0
    assert contrainte_reduite_en_conduite(0.05, 0.05) == 1
    assert contrainte_reduite_en_conduite(0.02, 0.05) == pytest.approx(0.4)
    with pytest.raises(ValueError, match="hors de la conduite"):
        contrainte_reduite_en_conduite(0.06, 0.05)


def test_la_loi_de_paroi_de_la_planche_34() -> None:
    """`u+ = 2,5 ln y+ + 5,0` : la constante de von Karman vaut 0,4, `B` 5,0."""
    assert KARMAN == 0.4
    assert CONSTANTE_LOG == 5.0
    assert vitesse_adimensionnee(100.0) == pytest.approx(2.5 * math.log(100.0) + 5.0)
    assert vitesse_adimensionnee(3.0) == 3.0


def test_la_loi_deficitaire_de_la_couche_externe() -> None:
    """Planche 35 : `(u_max - u) / u* = 2,5 ln(R / (R - r))`, nulle sur l'axe.

    C'est la loi logarithmique dont on a evalue `B` en `y = R`.
    """
    assert deficit_de_vitesse(0.0, 0.05) == 0
    rayon, position = 0.05, 0.03
    par_la_loi = 2.5 * math.log(rayon / (rayon - position))
    assert deficit_de_vitesse(position, rayon) == pytest.approx(par_la_loi)
    frottement = 2000.0
    sur_l_axe = vitesse_adimensionnee(frottement)
    ailleurs = vitesse_adimensionnee(frottement * (rayon - position) / rayon)
    assert deficit_de_vitesse(position, rayon) == pytest.approx(sur_l_axe - ailleurs)
    with pytest.raises(ValueError, match="hors de la conduite"):
        deficit_de_vitesse(0.05, 0.05)
