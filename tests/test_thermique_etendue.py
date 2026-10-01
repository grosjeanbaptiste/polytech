"""Les huit notions de thermique qui restaient, et leurs ancrages.

Trois sources distinctes servent de verite de reference, et aucune n'est
une invention de ce fichier.

Le **formulaire du cours** donne en clair -- ce qui est rare, le reste
etant des abaques -- la temperature en bout d'ailette droite, la
correction d'ailette circulaire avec son domaine `m(r_e - r_b) < 1,6`, les
correlations de plaque plane et de cylindre en travers avec leur table de
cinq plages, et les trois regimes de la lame de fluide avec le trou entre
1400 et 6000 qu'il ne correle pas.

Le **corrige du TP 4** publie `K_gl S = 775,61 W/degC`,
`Delta T_log = 15,582 degC`, `phi = 12,084 kW`, `S_i = 1,1556 m^2` et
`K_gli = 671,17 W/(m^2 K)`. Ces cinq valeurs se recoupent entre elles, et
c'est ce recoupement qui est verifie.

Pour le **corps noir**, rien n'est lisible : le formulaire n'en donne que
des courbes. L'ancrage est donc intrinseque, et il est plus fort -- la loi
de Planck integree sur tout le spectre doit redonner Stefan-Boltzmann, et
son maximum doit tomber sur Wien.
"""

from __future__ import annotations

import math

import mpmath
import pytest

from polytech.thermique.domain.conduction.ailettes import (
    AiletteCirculaire,
    AiletteDroite,
)
from polytech.thermique.domain.convection.forcee import (
    TABLE_DU_CYLINDRE,
    coefficient_d_echange,
    nusselt_cylindre,
    nusselt_plaque,
    prandtl,
    reynolds,
)
from polytech.thermique.domain.convection.naturelle import (
    grashof,
    nusselt_lame_de_fluide,
    rayleigh,
)
from polytech.thermique.domain.echangeurs import (
    Echangeur,
    efficacite_courants_croises_sans_melange,
    moyenne_logarithmique,
)
from polytech.thermique.domain.rayonnement.corps_noir import (
    BOLTZMANN,
    LUMIERE,
    PLANCK,
    STEFAN_BOLTZMANN,
    WIEN,
    emittance,
    longueur_d_onde_du_maximum,
    luminance_spectrale,
)
from polytech.thermique.domain.rayonnement.facteurs_de_forme import Enceinte
from polytech.thermique.domain.rayonnement.surfaces_grises import (
    SurfaceGrise,
    flux_entre_deux_surfaces,
)
from polytech.thermique.domain.transitoire import CorpsTransitoire

# --- Ailettes --------------------------------------------------------------


def test_l_efficacite_d_une_ailette_tres_courte_tend_vers_un() -> None:
    """Une ailette qui ne depasse pas est a la temperature du pied."""
    courte = AiletteDroite(parametre=1.0, longueur=1e-6)
    assert courte.efficacite() == pytest.approx(1.0, abs=1e-9)


def test_l_ecart_en_bout_suit_la_formule_du_formulaire() -> None:
    """`t_l - t_f = (t_p - t_f) / cosh(m l)`."""
    ailette = AiletteDroite(parametre=2.0, longueur=0.5)
    assert ailette.ecart_en_bout(100.0) == pytest.approx(100.0 / math.cosh(1.0))


def test_la_correction_circulaire_abaisse_l_efficacite() -> None:
    """Une ailette circulaire est moins efficace que la droite equivalente."""
    droite = AiletteDroite(parametre=4.0, longueur=0.2)
    circulaire = AiletteCirculaire(
        droite=droite,
        rayon_de_base=0.05,
        rayon_externe=0.25,
    )
    assert circulaire.efficacite() < droite.efficacite()


def test_la_correction_circulaire_refuse_de_sortir_de_son_domaine() -> None:
    """Le formulaire la donne pour `m (r_e - r_b) < 1,6`, et pas au-dela."""
    droite = AiletteDroite(parametre=20.0, longueur=0.2)
    circulaire = AiletteCirculaire(
        droite=droite,
        rayon_de_base=0.05,
        rayon_externe=0.25,
    )
    with pytest.raises(ValueError, match="hors de son domaine"):
        circulaire.efficacite()


def test_une_ailette_circulaire_inversee_est_refusee() -> None:
    with pytest.raises(ValueError, match="rayon externe"):
        AiletteCirculaire(
            droite=AiletteDroite(parametre=1.0, longueur=1.0),
            rayon_de_base=0.3,
            rayon_externe=0.1,
        )


# --- Regime transitoire ----------------------------------------------------

MINCE = CorpsTransitoire(
    echange=10.0,
    longueur_caracteristique=0.01,
    conductivite=200.0,
    capacite_volumique=2.4e6,
)


def test_un_corps_tres_conducteur_est_thermiquement_mince() -> None:
    """`Bi = 10 x 0,01 / 200 = 0,0005`, bien en dessous de 0,1."""
    assert MINCE.biot() == pytest.approx(0.0005)
    assert MINCE.est_thermiquement_mince()


def test_un_corps_epais_refuse_la_constante_de_temps() -> None:
    """Au-dela de `Bi = 0,1` la temperature uniforme est une fiction."""
    epais = CorpsTransitoire(
        echange=500.0,
        longueur_caracteristique=0.1,
        conductivite=0.2,
        capacite_volumique=2.4e6,
    )
    assert not epais.est_thermiquement_mince()
    with pytest.raises(ValueError, match="non thermiquement mince"):
        epais.constante_de_temps()


def test_la_decroissance_vaut_un_sur_e_apres_une_constante_de_temps() -> None:
    """Definition meme de la constante de temps."""
    tau = MINCE.constante_de_temps()
    assert MINCE.ecart_relatif(tau) == pytest.approx(1 / math.e)


def test_le_nombre_de_fourier_rend_le_temps_sans_dimension() -> None:
    """`Fo = alpha t / L^2` : a `Fo = 1`, la chaleur a traverse `L`."""
    duree = MINCE.longueur_caracteristique**2 / MINCE.diffusivite()
    assert MINCE.fourier(duree) == pytest.approx(1.0)


# --- Corps noir ------------------------------------------------------------


def test_la_constante_de_stefan_boltzmann_decoule_de_planck() -> None:
    """Le recoupement : sigma = 2 pi^5 k^4 / (15 h^3 c^2).

    C'est l'integrale de la loi de Planck sur tout le spectre, faite une
    fois pour toutes a la main -- le changement de variable
    `x = h c / (lambda k T)` ramene l'integrale a `pi^4/15`. Verifier
    l'identite plutot que la quadrature est plus fort : les deux
    constantes sont declarees independamment dans le module, et une faute
    de frappe sur l'une se verrait ici.
    """
    sigma = 2 * math.pi**5 * BOLTZMANN**4 / (15 * PLANCK**3 * LUMIERE**2)
    assert sigma == pytest.approx(STEFAN_BOLTZMANN, rel=1e-9)


def test_la_constante_de_wien_decoule_de_planck() -> None:
    """Second recoupement : `lambda_max T = h c / (k x*)`.

    ou `x*` est l'unique racine positive de `x = 5 (1 - exp(-x))`, qui
    annule la derivee de la loi de Planck. La racine est trouvee
    numeriquement, sur une equation bien conditionnee -- et non sur la
    luminance elle-meme, dont les valeurs sont de l'ordre de `10^13`.
    """
    racine = mpmath.findroot(lambda x: x - 5 * (1 - mpmath.e ** (-x)), 5)
    constante = PLANCK * LUMIERE / (BOLTZMANN * float(racine))
    assert constante == pytest.approx(WIEN, rel=1e-9)


def test_le_maximum_numerique_de_planck_tombe_bien_sur_wien() -> None:
    """Et pour finir, la luminance elle-meme, echantillonnee.

    Pas de derivee numerique a l'echelle du micrometre : on compare
    simplement la luminance au maximum de Wien a celles de ses voisins.
    Si Wien ne donnait pas le maximum, l'un des deux serait plus haut.
    """
    temperature = 1000.0
    maximum = longueur_d_onde_du_maximum(temperature)
    au_sommet = luminance_spectrale(maximum, temperature)
    for facteur in (0.8, 0.9, 1.1, 1.25):
        assert luminance_spectrale(maximum * facteur, temperature) < au_sommet


def test_une_temperature_negative_n_emet_rien() -> None:
    with pytest.raises(ValueError, match="temperature absolue negative"):
        emittance(-1.0)


def test_une_temperature_nulle_n_a_pas_de_maximum() -> None:
    with pytest.raises(ValueError, match="nulle ou negative"):
        longueur_d_onde_du_maximum(0.0)


def test_une_longueur_d_onde_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="longueur d'onde nulle"):
        luminance_spectrale(0.0, 1000.0)


# --- Facteurs de forme -----------------------------------------------------

DEUX_SURFACES = Enceinte(
    surfaces=(2.0, 1.0),
    facteurs={(0, 1): 0.5, (1, 0): 1.0, (0, 0): 0.5, (1, 1): 0.0},
)


def test_la_reciprocite_contraint_les_facteurs_lus_sur_abaque() -> None:
    """`S_1 F_12 = S_2 F_21` : 2 x 0,5 = 1 x 1."""
    assert DEUX_SURFACES.reciproque(0, 1)


def test_un_facteur_se_deduit_de_l_autre_sans_seconde_lecture() -> None:
    """Lire un abaque deux fois, c'est se tromper deux fois."""
    assert DEUX_SURFACES.facteur_deduit(0, 1) == pytest.approx(0.5)


def test_la_sommation_vaut_un_sur_une_enceinte_fermee() -> None:
    assert DEUX_SURFACES.est_fermee_depuis(0)
    assert DEUX_SURFACES.est_fermee_depuis(1)


def test_un_facteur_hors_de_zero_un_est_refuse() -> None:
    """Un facteur d'angle est une fraction : au-dela de un il n'a aucun sens."""
    with pytest.raises(ValueError, match="hors de"):
        Enceinte(surfaces=(1.0,), facteurs={(0, 0): 1.5})


def test_une_surface_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="surface nulle"):
        Enceinte(surfaces=(0.0,), facteurs={})


# --- Surfaces grises -------------------------------------------------------


def test_une_surface_noire_n_oppose_aucune_resistance_de_surface() -> None:
    """`(1 - eps) / (eps S)` s'annule pour `eps = 1`."""
    noire = SurfaceGrise(aire=1.0, emissivite=1.0, temperature=400.0)
    assert noire.resistance_de_surface() == pytest.approx(0.0)


def test_le_flux_va_du_chaud_vers_le_froid() -> None:
    """Signe du resultat : une evidence physique qu'un signe inverse trahirait."""
    chaude = SurfaceGrise(aire=1.0, emissivite=0.8, temperature=500.0)
    froide = SurfaceGrise(aire=1.0, emissivite=0.8, temperature=300.0)
    assert flux_entre_deux_surfaces(chaude, froide, facteur=1.0) > 0


def test_deux_surfaces_noires_echangent_la_difference_des_emittances() -> None:
    """Sans resistance de surface, il ne reste que la resistance geometrique."""
    chaude = SurfaceGrise(aire=1.0, emissivite=1.0, temperature=500.0)
    froide = SurfaceGrise(aire=1.0, emissivite=1.0, temperature=300.0)
    attendu = emittance(500.0) - emittance(300.0)
    assert flux_entre_deux_surfaces(chaude, froide, facteur=1.0) == pytest.approx(
        attendu
    )


def test_une_emissivite_nulle_est_refusee() -> None:
    """Elle opposerait une resistance infinie que le document composerait."""
    with pytest.raises(ValueError, match="emissivite hors de"):
        SurfaceGrise(aire=1.0, emissivite=0.0, temperature=300.0)


# --- Convection forcee -----------------------------------------------------


def test_les_nombres_sans_dimension_se_composent() -> None:
    """`Re`, `Pr`, puis `Nu`, puis `K` : la chaine du chapitre."""
    nombre = reynolds(vitesse=2.0, longueur=0.5, viscosite_cinematique=1.5e-5)
    assert nombre == pytest.approx(2.0 * 0.5 / 1.5e-5)
    nombre_de_prandtl = prandtl(viscosite_cinematique=1.5e-5, diffusivite=2.1e-5)
    assert nombre_de_prandtl == pytest.approx(0.714, abs=0.001)


def test_la_correlation_de_plaque_change_de_regime_au_reynolds_critique() -> None:
    """Le formulaire donne deux formules : il faut choisir la bonne."""
    laminaire = nusselt_plaque(1e4, 0.7)
    turbulent = nusselt_plaque(1e6, 0.7)
    assert laminaire == pytest.approx(0.664 * 1e4**0.5 * 0.7 ** (1 / 3))
    assert turbulent == pytest.approx(0.037 * 1e6**0.8 * 0.7 ** (1 / 3))


@pytest.mark.parametrize(
    ("borne_basse", "borne_haute", "coefficient", "exposant"),
    TABLE_DU_CYLINDRE,
)
def test_chaque_plage_du_cylindre_emploie_ses_propres_coefficients(
    borne_basse: float,
    borne_haute: float,
    coefficient: float,
    exposant: float,
) -> None:
    """Les cinq plages du formulaire, verifiees une par une."""
    milieu = (borne_basse + borne_haute) / 2
    attendu = coefficient * milieu**exposant * 0.7 ** (1 / 3)
    assert nusselt_cylindre(milieu, 0.7) == pytest.approx(attendu)


def test_un_reynolds_hors_des_plages_tabulees_est_refuse() -> None:
    """Extrapoler une correlation empirique est l'erreur du chapitre."""
    with pytest.raises(ValueError, match="hors des plages"):
        nusselt_cylindre(1e7, 0.7)


def test_le_coefficient_d_echange_revient_du_nombre_de_nusselt() -> None:
    assert coefficient_d_echange(nusselt=100.0, conductivite=0.026, longueur=0.5) == (
        pytest.approx(100.0 * 0.026 / 0.5)
    )


# --- Convection naturelle --------------------------------------------------


def test_une_lame_de_fluide_au_repos_ne_fait_que_conduire() -> None:
    """`Gr Pr < 1400` : le formulaire donne `Nu = 1`."""
    assert nusselt_lame_de_fluide(1000.0) == pytest.approx(1.0)


def test_la_zone_de_transition_du_formulaire_n_est_pas_comblee() -> None:
    """Entre 1400 et 6000 le formulaire ne correle rien.

    Y interpoler serait inventer une correlation.
    """
    with pytest.raises(ValueError, match="zone de transition"):
        nusselt_lame_de_fluide(3000.0)


@pytest.mark.parametrize(
    ("produit", "coefficient", "exposant"),
    [(1e5, 0.11, 0.29), (1e7, 0.40, 0.20)],
)
def test_les_deux_regimes_correles_de_la_lame(
    produit: float,
    coefficient: float,
    exposant: float,
) -> None:
    attendu = coefficient * produit**exposant
    assert nusselt_lame_de_fluide(produit) == pytest.approx(attendu)


def test_au_dela_de_la_derniere_plage_le_formulaire_se_tait() -> None:
    with pytest.raises(ValueError, match="depasse la derniere plage"):
        nusselt_lame_de_fluide(1e9)


def test_grashof_et_rayleigh_se_composent() -> None:
    nombre = grashof(
        dilatation=1 / 300,
        ecart=20.0,
        longueur=0.1,
        viscosite_cinematique=1.5e-5,
    )
    assert rayleigh(nombre, 0.7) == pytest.approx(nombre * 0.7)


# --- Echangeurs ------------------------------------------------------------

CONDUCTANCE_DU_TP = 775.61
ECART_LOG_DU_TP = 15.582
SURFACE_DU_TP = 1.1556


def test_le_flux_du_tp_4_se_retrouve_a_partir_des_valeurs_publiees() -> None:
    """`775,61 x 15,582 = 12 084 W`, soit les `12,084 kW` du corrige."""
    echangeur = Echangeur(conductance=CONDUCTANCE_DU_TP)
    flux = echangeur.conductance * ECART_LOG_DU_TP
    assert flux == pytest.approx(12_084.0, abs=2.0)


def test_le_coefficient_rapporte_du_tp_4_est_celui_du_corrige() -> None:
    """`775,61 / 1,1556 = 671,2`, soit le `K_gli = 671,17` publie."""
    echangeur = Echangeur(conductance=CONDUCTANCE_DU_TP)
    assert echangeur.coefficient_rapporte(SURFACE_DU_TP) == pytest.approx(
        671.17, abs=0.1
    )


def test_la_moyenne_logarithmique_degenere_en_ecart_quand_les_deux_sont_egaux() -> None:
    """La formule est indeterminee ; sa limite est l'ecart lui-meme."""
    assert moyenne_logarithmique(10.0, 10.0) == pytest.approx(10.0)


def test_la_moyenne_logarithmique_est_entre_les_deux_ecarts() -> None:
    """Et plus proche du petit : c'est ce qui la distingue de l'arithmetique."""
    moyenne = moyenne_logarithmique(30.0, 10.0)
    assert 10.0 < moyenne < 30.0
    assert moyenne < 20.0


def test_des_ecarts_de_signes_opposes_sont_refuses() -> None:
    """Les courbes de temperature se croiseraient : impossible."""
    with pytest.raises(ValueError, match="signes opposes"):
        moyenne_logarithmique(10.0, -5.0)


def test_le_nut_se_tire_de_la_conductance_et_de_la_capacite() -> None:
    echangeur = Echangeur(conductance=CONDUCTANCE_DU_TP)
    assert echangeur.nombre_d_unites_de_transfert(1000.0) == pytest.approx(0.77561)


def test_l_efficacite_croit_avec_le_nut_et_reste_sous_un() -> None:
    """Deux proprietes qu'aucune efficacite ne peut violer."""
    faible = efficacite_courants_croises_sans_melange(0.5, 0.5)
    forte = efficacite_courants_croises_sans_melange(5.0, 0.5)
    assert 0 < faible < forte < 1


def test_un_rapport_de_capacites_hors_de_l_intervalle_est_refuse() -> None:
    with pytest.raises(ValueError, match="rapport de capacites"):
        efficacite_courants_croises_sans_melange(1.0, 1.5)


def test_une_conductance_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="conductance globale"):
        Echangeur(conductance=0.0)
