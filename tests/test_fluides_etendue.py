"""Les cinq notions de fluides qui restaient : chapitres IV, VII et VIII.

Les ancrages sont les resultats que les planches affichent et qui sont
exacts, donc verifiables sans tolerance choisie :

* Torricelli, `V = racine(2 g h)`, et Pitot, `V = racine(2 Delta p / rho)`
  -- tous deux **derives** de la charge de Bernoulli et non recopies ;
* l'ecoulement de Couette totalement developpe : profil lineaire,
  cisaillement uniforme, obtenus en integrant l'equation reduite ;
* les regles sur les moyennes de Reynolds, dont la troisieme est celle qui
  fait exister le tenseur de Reynolds.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.fluides.domain.aerodynamique import Ecoulement, SurfacePortante
from polytech.fluides.domain.cinematique.symboles import T, X, Y
from polytech.fluides.domain.fondamentales import bernoulli, continuite, navier_stokes
from polytech.fluides.domain.grandeurs import MasseVolumique
from polytech.fluides.domain.turbulence import (
    Decomposition,
    covariance,
    vitesse_adimensionnee,
)

EAU = MasseVolumique(kg_par_m3=1000.0)

# --- Chapitre IV : conservation de la masse --------------------------------


def test_un_champ_a_divergence_nulle_est_incompressible() -> None:
    """`u = 2x`, `v = -2y` : le champ C de l'activite d'integration."""
    assert continuite.est_incompressible((2 * X, -2 * Y))


def test_un_champ_a_divergence_non_nulle_ne_l_est_pas() -> None:
    assert not continuite.est_incompressible((X**2, Y))


def test_la_forme_locale_se_reduit_a_la_forme_incompressible() -> None:
    """A masse volumique constante, le terme en `d rho/dt` disparait."""
    locale = continuite.forme_locale((X * T, X - Y))
    constante = sp.Symbol("rho_0", positive=True)
    sans_variation = locale.lhs.subs(continuite.MASSE_VOLUMIQUE(X, Y, T), constante)
    rapport = sp.simplify(sans_variation.doit() / constante)
    assert sp.simplify(rapport - continuite.divergence((X * T, X - Y))) == 0


def test_un_resserrement_accelere_l_ecoulement() -> None:
    """`S_1 V_1 = S_2 V_2` : halver la section double la vitesse."""
    aval = continuite.vitesse_apres_resserrement(2.0, 0.1, 0.05)
    assert aval == pytest.approx(4.0)


def test_une_section_nulle_est_refusee() -> None:
    """Elle donnerait une vitesse infinie que le document composerait."""
    with pytest.raises(ValueError, match="section nulle"):
        continuite.vitesse_dans_la_section(1.0, 0.0)


# --- Chapitre IV : conservation de l'energie -------------------------------


def test_torricelli_se_derive_de_bernoulli() -> None:
    """`V = racine(2 g h)` -- obtenu, et non recopie."""
    hauteur = sp.Symbol("h", positive=True)
    attendu = sp.sqrt(2 * bernoulli.GRAVITE_EXACTE * hauteur)
    assert sp.simplify(bernoulli.vitesse_de_torricelli(hauteur) - attendu) == 0


def test_pitot_se_derive_du_meme_principe() -> None:
    """`V = racine(2 Delta p / rho)`."""
    ecart = sp.Symbol("Delta_p", positive=True)
    attendu = sp.sqrt(2 * ecart / bernoulli.MASSE_VOLUMIQUE)
    assert sp.simplify(bernoulli.vitesse_de_pitot(ecart) - attendu) == 0


def test_la_charge_se_conserve_le_long_d_une_ligne_de_courant() -> None:
    """Deux points, meme charge : c'est l'enonce du theoreme."""
    hauteur = sp.Symbol("h", positive=True)
    vitesse = bernoulli.vitesse_de_torricelli(hauteur)
    en_haut = bernoulli.charge(altitude=hauteur, vitesse=sp.Integer(0))
    en_bas = bernoulli.charge(altitude=sp.Integer(0), vitesse=vitesse)
    assert sp.simplify(en_haut - en_bas) == 0


# --- Chapitre IV : quantite de mouvement -----------------------------------


def test_l_equation_reduite_de_couette_est_du_second_ordre() -> None:
    """Il ne reste que le laplacien : deux integrations, deux constantes."""
    assert navier_stokes.equation_reduite().lhs.args[1] == (Y, 2)


def test_le_profil_de_couette_est_lineaire() -> None:
    """Resultat obtenu, non suppose : le degre est verifie."""
    profil = navier_stokes.profil_de_couette()
    assert sp.degree(sp.Poly(profil, Y), Y) == 1


def test_le_profil_de_couette_satisfait_ses_deux_conditions() -> None:
    """Nul a la paroi fixe, `U` a la paroi mobile."""
    profil = navier_stokes.profil_de_couette()
    assert sp.simplify(profil.subs(Y, 0)) == 0
    assert (
        sp.simplify(
            profil.subs(Y, navier_stokes.HAUTEUR) - navier_stokes.VITESSE_DE_PAROI,
        )
        == 0
    )


def test_le_cisaillement_de_couette_est_uniforme() -> None:
    """C'est ce qui caracterise Couette : `tau = mu U / h`, partout le meme."""
    cisaillement = navier_stokes.cisaillement()
    assert Y not in cisaillement.free_symbols
    attendu = (
        navier_stokes.VISCOSITE
        * navier_stokes.VITESSE_DE_PAROI
        / (navier_stokes.HAUTEUR)
    )
    assert sp.simplify(cisaillement - attendu) == 0


# --- Chapitre VII : turbulence ---------------------------------------------

SIGNAL = Decomposition(echantillons=(1.0, 3.0, 2.0, 4.0, 0.0))
AUTRE = Decomposition(echantillons=(2.0, 4.0, 1.0, 5.0, 3.0))


def test_la_moyenne_des_fluctuations_est_nulle() -> None:
    """Premiere regle des planches, et elle se verifie exactement."""
    assert sum(SIGNAL.fluctuations()) == pytest.approx(0.0)


def test_la_moyenne_d_une_moyenne_est_cette_moyenne() -> None:
    """Deuxieme regle : moyenner deux fois ne change rien."""
    moyenne = SIGNAL.moyenne()
    constante = Decomposition(echantillons=(moyenne,) * len(SIGNAL.echantillons))
    assert constante.moyenne() == pytest.approx(moyenne)


def test_la_moyenne_d_un_produit_n_est_pas_le_produit_des_moyennes() -> None:
    """Troisieme regle, et c'est elle qui fait exister le tenseur de Reynolds.

    L'ecart entre les deux est exactement la covariance des fluctuations.
    """
    produits = tuple(
        un * deux
        for un, deux in zip(SIGNAL.echantillons, AUTRE.echantillons, strict=True)
    )
    moyenne_du_produit = Decomposition(echantillons=produits).moyenne()
    produit_des_moyennes = SIGNAL.moyenne() * AUTRE.moyenne()
    ecart = moyenne_du_produit - produit_des_moyennes
    assert ecart == pytest.approx(covariance(SIGNAL, AUTRE))
    assert ecart != pytest.approx(0.0)


def test_l_ecart_type_est_la_racine_de_la_variance() -> None:
    assert SIGNAL.ecart_type() == pytest.approx(math.sqrt(SIGNAL.variance()))


def test_un_echantillonnage_vide_est_refuse() -> None:
    with pytest.raises(ValueError, match="aucun echantillon"):
        Decomposition(echantillons=())


def test_deux_echantillonnages_de_tailles_differentes_sont_refuses() -> None:
    with pytest.raises(ValueError, match="tailles differentes"):
        covariance(SIGNAL, Decomposition(echantillons=(1.0,)))


def test_la_sous_couche_visqueuse_suit_u_plus_egale_y_plus() -> None:
    """`y+ < 5` : le profil y est lineaire."""
    assert vitesse_adimensionnee(2.0) == pytest.approx(2.0)


def test_la_zone_tampon_sans_loi_est_refusee() -> None:
    """Les planches la nomment sans la correler : y choisir serait taire un choix."""
    with pytest.raises(ValueError, match="zone tampon"):
        vitesse_adimensionnee(15.0)


def test_la_loi_logarithmique_vaut_au_dela_de_trente() -> None:
    assert vitesse_adimensionnee(100.0) == pytest.approx(
        math.log(100.0) / 0.41 + 5.0,
    )


# --- Chapitre VIII : aerodynamique -----------------------------------------

AILE = SurfacePortante(
    ecoulement=Ecoulement(fluide=EAU, vitesse=10.0),
    surface_de_reference=2.0,
)


def test_la_pression_dynamique_vaut_un_demi_rho_v_carre() -> None:
    assert AILE.ecoulement.pression_dynamique() == pytest.approx(0.5 * 1000.0 * 100.0)


def test_le_coefficient_et_la_force_sont_reciproques() -> None:
    """Deux definitions inversibles : l'aller-retour doit etre exact."""
    force = 1234.0
    coefficient = AILE.coefficient(force)
    assert AILE.force(coefficient) == pytest.approx(force)


def test_la_finesse_est_le_rapport_des_deux_coefficients() -> None:
    assert AILE.finesse(1.2, 0.08) == pytest.approx(15.0)


def test_une_trainee_nulle_rendrait_la_finesse_infinie() -> None:
    with pytest.raises(ValueError, match="coefficient de trainee"):
        AILE.finesse(1.0, 0.0)


def test_une_vitesse_amont_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="vitesse amont"):
        Ecoulement(fluide=EAU, vitesse=0.0)


def test_une_surface_de_reference_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="surface de reference"):
        SurfacePortante(
            ecoulement=Ecoulement(fluide=EAU, vitesse=1.0),
            surface_de_reference=0.0,
        )
