"""Le syllabus de thermique, tome 2 : la convection naturelle.

Source : `ThermiqueFondements/SYLLABUS/Syllabus_Tome_2_2013.pdf`, pages 51
a 63 (premiere partie, chapitre 4), lues sur les pages rendues.

Les oracles : les relations (4.1) a (4.9), la relation d'Elenbaas et sa
figure 4.10 -- lue au pixel --, le tableur de la planche du jeu 10 pour
le cylindre, et les exercices III.4.1 a III.4.4.
"""

from __future__ import annotations

import math

import pytest
from scipy.optimize import minimize_scalar

from polytech.fluides.domain.proprietes import (
    coefficient_de_dilatation_du_gaz_parfait,
)
from polytech.thermique.domain.conduction.ailette_droite import (
    parametre_d_ailette_mince,
)
from polytech.thermique.domain.convection.chenal import (
    ReseauDAilettes,
    ecartement_optimal,
    groupe_du_chenal,
    nusselt_d_elenbaas,
)
from polytech.thermique.domain.convection.forcee import coefficient_d_echange
from polytech.thermique.domain.convection.naturelle import (
    grashof,
    nusselt_du_cylindre_horizontal,
    nusselt_lame_de_fluide,
    rayleigh,
)
from polytech.thermique.domain.convection.naturelle_externe import (
    GRASHOF_CRITIQUE,
    coefficient_du_cylindre_dans_l_air,
    coefficient_local_laminaire,
    conductance_de_la_lame,
    epaisseur_d_eckert,
    hauteur_critique,
    nusselt_de_la_paroi_verticale,
    nusselt_de_la_plaque_horizontale,
    nusselt_du_cylindre,
)
from polytech.thermique.domain.tables.air_sec import air_sec


def test_la_transition_d_une_paroi_verticale() -> None:
    """(4.1) : `Gr_crit = 1e9` ; pour l'air et quelques dizaines de degres, un metre."""
    assert GRASHOF_CRITIQUE == 1e9
    air = air_sec(300.0)
    dilatation = coefficient_de_dilatation_du_gaz_parfait(300.0)
    critique = hauteur_critique(dilatation, 30.0, air.viscosite_cinematique())
    assert critique == pytest.approx(0.63, abs=0.01)
    assert grashof(dilatation, 30.0, critique, air.viscosite_cinematique()) == (
        pytest.approx(1e9)
    )


def test_le_nusselt_moyen_d_une_paroi_verticale() -> None:
    """(4.2) et (4.3) : `0,56 Ra^1/4` en laminaire, `0,12 Ra^1/3` en turbulent."""
    assert nusselt_de_la_paroi_verticale(1e6) == pytest.approx(0.56 * 1e6**0.25)
    assert nusselt_de_la_paroi_verticale(1e12) == pytest.approx(0.12 * 1e4)
    with pytest.raises(ValueError, match="hors du domaine"):
        nusselt_de_la_paroi_verticale(100.0)


def test_en_laminaire_le_flux_croit_comme_l_ecart_a_la_puissance_cinq_quarts() -> None:
    """Page 53 : `phi = K theta ~ theta^5/4` ; page 54 : `K ~ h^-1/4`."""
    simple = nusselt_de_la_paroi_verticale(1e6)
    assert nusselt_de_la_paroi_verticale(2e6) / simple == pytest.approx(2**0.25)
    basse = nusselt_de_la_paroi_verticale(1e5) / 1.0
    seize_fois_plus_haute = nusselt_de_la_paroi_verticale(1e5 * 16**3) / 16.0
    assert seize_fois_plus_haute / basse == pytest.approx(16**-0.25)


def test_le_cylindre_horizontal_du_tableur_des_planches() -> None:
    """(4.4) : `0,47 (Gr Pr)^1/4`. Jeu 10 : `C = 0,47`, `n = 0,25`, et trois `Nu`.

    La bibliotheque portait 0,53, que ni le syllabus ni les planches ne
    donnent.
    """
    for produit, imprime in ((5.11e8, 70.66), (4.10e8, 66.89), (4.41e8, 68.12)):
        assert nusselt_du_cylindre_horizontal(produit) == pytest.approx(
            imprime, abs=0.02
        )
    assert nusselt_du_cylindre(2e5, 0.7) == pytest.approx(0.47 * (1.4e5) ** 0.25)
    assert nusselt_du_cylindre(2e9, 0.7) == pytest.approx(0.10 * (1.4e9) ** (1 / 3))
    juste_au_dela = nusselt_du_cylindre(1.2e9, 0.7)
    assert juste_au_dela == pytest.approx(0.10 * (8.4e8) ** (1 / 3))
    with pytest.raises(ValueError, match="hors du domaine"):
        nusselt_du_cylindre(100.0, 0.7)


def test_la_plaque_horizontale_ne_depend_pas_de_sa_dimension() -> None:
    """(4.5) : `Nu = 0,137 (Gr Pr)^1/3` ; l'exposant un tiers efface la longueur."""
    assert nusselt_de_la_plaque_horizontale(8e6) == pytest.approx(0.137 * 200.0)
    petite = nusselt_de_la_plaque_horizontale(1e6) / 1.0
    grande = nusselt_de_la_plaque_horizontale(1e6 * 3**3) / 3.0
    assert grande == pytest.approx(petite)


def test_la_lame_de_fluide_et_sa_conductivite_equivalente() -> None:
    """(4.6) a (4.9) : `lambda_eq / lambda = F(Gr_b Pr)`, un sous 1400."""
    assert nusselt_lame_de_fluide(1000.0) == 1.0
    assert nusselt_lame_de_fluide(1e5) == pytest.approx(0.11 * 1e5**0.29)
    assert nusselt_lame_de_fluide(1e7) == pytest.approx(0.40 * 1e7**0.20)
    au_repos = conductance_de_la_lame(1000.0, 0.025, 0.01)
    assert au_repos == pytest.approx(2.5)
    en_mouvement = conductance_de_la_lame(1e5, 0.025, 0.05)
    assert en_mouvement == pytest.approx(0.11 * 1e5**0.29 * 0.025 / 0.05)


def test_l_exercice_iii_4_1_un_tuyau_dans_l_air_calme() -> None:
    """25 mm, 120 et 20 degres : `Gr = 1,12e5`, puis « la formule 4.4 donne 8,51 ».

    Elle donne 7,93, d'ou 9,35 W/m^2 K et 73 W/m pour 10,04 et 79. Et le
    `Pr = 0,721` de l'enonce vaut 0,698 dans la table 3.2 du syllabus.
    """
    air = air_sec(273.15 + (120.0 + 20.0) / 2)
    assert air.conductivite == pytest.approx(0.0295, abs=5e-5)
    assert air.viscosite == pytest.approx(2.043e-5, abs=2e-8)
    assert air.prandtl == pytest.approx(0.698, abs=5e-4)
    cinematique = 2.043e-5 / (1.287 * 273.1 / 343.1)
    nombre = grashof(1 / 343.1, 100.0, 0.025, cinematique)
    assert nombre == pytest.approx(1.12e5, rel=5e-3)
    assert rayleigh(nombre, 0.721) == pytest.approx(8.089e4, rel=2e-3)
    nusselt = nusselt_du_cylindre(nombre, 0.721)
    assert nusselt == pytest.approx(7.93, abs=0.01)
    echange = coefficient_d_echange(nusselt, 0.0295, 0.025)
    assert echange == pytest.approx(9.35, abs=0.02)
    assert math.pi * 0.025 * echange * 100.0 == pytest.approx(73.5, abs=0.2)
    imprime = coefficient_d_echange(8.51, 0.0295, 0.025)
    assert imprime == pytest.approx(10.04, abs=0.005)
    assert math.pi * 0.025 * imprime * 100.0 == pytest.approx(79.0, abs=0.2)


def test_la_formule_rapide_du_cylindre_dans_l_air() -> None:
    """Page 58 : `K = 1,3 (theta_o / d)^0,25`, pour l'air aux conditions ambiantes."""
    rapide = coefficient_du_cylindre_dans_l_air(100.0, 0.025)
    assert rapide == pytest.approx(1.3 * 4000.0**0.25)
    assert rapide == pytest.approx(10.3, abs=0.05)
    with pytest.raises(ValueError, match="diametre nul ou negatif"):
        coefficient_du_cylindre_dans_l_air(100.0, 0.0)


def test_l_exercice_iii_4_2_la_couche_limite_d_une_plaque_verticale() -> None:
    """90 et 20 degres, a 0,20 m : `Gr = 4,84e7`, 1,28 cm, 6,07 puis 4,55 W/m^2 K."""
    nombre = grashof(3.048e-3, 70.0, 0.2, 1.86e-5)
    assert nombre == pytest.approx(4.84e7, rel=1e-3)
    assert nombre < GRASHOF_CRITIQUE
    epaisseur = epaisseur_d_eckert(0.2, nombre, 0.701)
    assert epaisseur == pytest.approx(0.0128, abs=5e-5)
    moyen = coefficient_d_echange(
        nusselt_de_la_paroi_verticale(nombre * 0.701), 0.0284, 0.2
    )
    assert moyen == pytest.approx(6.07, abs=0.005)
    assert coefficient_local_laminaire(moyen) == pytest.approx(4.55, abs=0.005)


def test_le_coefficient_local_vaut_les_trois_quarts_du_moyen() -> None:
    """« Comme il varie comme z^-1/4 » : la moyenne de `z^-1/4` vaut `4/3` de sa fin."""
    hauteur = 0.2
    pas = hauteur / 100000
    somme = sum(((k + 0.5) * pas) ** -0.25 for k in range(100000)) * pas
    assert somme / hauteur == pytest.approx(4 / 3 * hauteur**-0.25, rel=1e-4)


def test_l_exercice_iii_4_3_la_lame_d_air_d_un_mur() -> None:
    """5 cm entre 0 et 5 degres : `Gr Pr = 91 200`, `lambda_eq / lambda = 3,05`.

    Le calcul rend 3,02, et 1,47 W/m^2 K pour 1,49.
    """
    produit = rayleigh(grashof(3.628e-3, 5.0, 0.05, 1.32e-5), 0.714)
    assert produit == pytest.approx(91200.0, rel=1e-3)
    assert nusselt_lame_de_fluide(produit) == pytest.approx(3.02, abs=0.005)
    conductance = conductance_de_la_lame(produit, 0.0244, 0.05)
    assert conductance == pytest.approx(1.49, rel=0.015)


FIGURE_4_10 = (
    (15.0, 0.537), (30.0, 0.835), (50.0, 1.105), (70.0, 1.286), (150.0, 1.793),
    (300.0, 2.300), (500.0, 2.683), (700.0, 2.954), (1500.0, 3.615),
)  # fmt: skip
"""La courbe de la figure 4.10, relevee au pixel sur la page rendue a 300 dpi."""


@pytest.mark.parametrize(("groupe", "lu"), FIGURE_4_10)
def test_la_relation_d_elenbaas_suit_la_figure_4_10(groupe: float, lu: float) -> None:
    """`Nu_b = (q / 24) {1 - exp[-24 (0,5 / q)^0,75]}`, a 3 % de la courbe."""
    assert nusselt_d_elenbaas(groupe) == pytest.approx(lu, rel=0.03)


def test_les_deux_limites_de_la_relation_d_elenbaas() -> None:
    """Chenal etroit : `q / 24`. Chenal large : `q^1/4`, les plaques s'ignorent."""
    assert nusselt_d_elenbaas(0.1) == pytest.approx(0.1 / 24, rel=1e-6)
    large = nusselt_d_elenbaas(1e12) / nusselt_d_elenbaas(1e8)
    assert large == pytest.approx(10.0, rel=2e-3)
    assert groupe_du_chenal(2000.0, 0.7, ecartement=0.005, hauteur=0.1) == (
        pytest.approx(70.0)
    )
    with pytest.raises(ValueError, match="groupe nul ou negatif"):
        nusselt_d_elenbaas(0.0)


def test_l_ecartement_optimal_des_ailettes() -> None:
    """Page 62 : `Phi ~ F(q) q^-0,5`, maximum pour `q ~ 50` ; `b_opt = 6,25 mm`."""

    def oppose(groupe: float) -> float:
        return -nusselt_d_elenbaas(groupe) / math.sqrt(groupe)

    optimum = minimize_scalar(oppose, bounds=(5.0, 500.0), method="bounded")
    assert optimum.x == pytest.approx(50.0, abs=1.0)
    ecartement = ecartement_optimal(
        1.807e-5, hauteur=0.08, dilatation=0.003095, ecart=40.0, prandtl=0.701
    )
    assert ecartement == pytest.approx(0.00625, abs=1e-5)
    produit = rayleigh(grashof(0.003095, 40.0, ecartement, 1.807e-5), 0.701)
    assert produit * ecartement / 0.08 == pytest.approx(50.0)


def test_la_figure_4_10_ne_donne_pas_1_25() -> None:
    """« Pour q = 50, le graphique precedent fournit Nu_opt = 1,25 ».

    La courbe passe a 1,105 en `q = 50`, et la relation imprimee y vaut
    1,108 : `K = 4,99 W/m^2 K`, et non 5,63.
    """
    optimal = nusselt_d_elenbaas(50.0)
    assert optimal == pytest.approx(1.108, abs=0.001)
    assert coefficient_d_echange(1.25, 0.02816, 0.00625) == pytest.approx(
        5.63, abs=0.005
    )
    assert coefficient_d_echange(optimal, 0.02816, 0.00625) == pytest.approx(
        4.99, abs=0.005
    )


@pytest.mark.parametrize(
    ("epaisseur", "ligne"),
    [
        (0.001, (28, 2.005, 0.48, 121.0, 0.56)),
        (0.002, (25, 1.418, 0.63, 142.0, 1.0)),
        (0.003, (22, 1.158, 0.71, 140.8, 1.32)),
    ],
)
def test_le_tableau_de_l_exercice_iii_4_4(
    epaisseur: float, ligne: tuple[int, float, float, float, float]
) -> None:
    """Page 62, avec le `K = 5,63` du syllabus ; ses `mL` supposent 175 W/m K."""
    nombre, reduit, efficacite, flux, volume = ligne
    reseau = ReseauDAilettes(
        largeur=0.21,
        hauteur=0.08,
        longueur=0.25,
        epaisseur=epaisseur,
        ecartement=0.00625,
    )
    assert reseau.nombre() == nombre
    parametre = parametre_d_ailette_mince(5.63, 175.0, epaisseur)
    assert parametre * 0.25 == pytest.approx(reduit, abs=0.001)
    omega = reseau.efficacite(5.63, 175.0)
    assert omega == pytest.approx(efficacite, abs=0.005)
    assert reseau.flux_des_ailettes(5.63, 40.0, 175.0) == pytest.approx(flux, rel=0.01)
    assert reseau.volume_de_metal() * 1e3 == pytest.approx(volume)


def test_les_surfaces_d_un_reseau_d_ailettes() -> None:
    """Page 60 : `S_a = 2 L H A / (b + e)`, `S_o = b A H / (b + e)`."""
    reseau = ReseauDAilettes(0.21, 0.08, 0.25, epaisseur=0.002, ecartement=0.00625)
    assert reseau.surface_des_ailettes() == pytest.approx(
        2 * 0.25 * 0.08 * 0.21 / 0.00825
    )
    assert reseau.surface_primaire() == pytest.approx(0.00625 * 0.21 * 0.08 / 0.00825)
    apparent = 142.0 / (0.08 * 0.21 * 40.0)
    assert apparent == pytest.approx(211.0, abs=0.5)
    with pytest.raises(ValueError, match="ecartement nul ou negatif"):
        ReseauDAilettes(0.21, 0.08, 0.25, epaisseur=0.002, ecartement=0.0)


def test_avec_la_valeur_de_la_figure_le_reseau_dissipe_130_watts() -> None:
    """25 ailettes de 2 mm : 142 W avec `Nu = 1,25`, 130 W avec 1,108."""
    reseau = ReseauDAilettes(0.21, 0.08, 0.25, epaisseur=0.002, ecartement=0.00625)
    echange = coefficient_d_echange(nusselt_d_elenbaas(50.0), 0.02816, 0.00625)
    assert reseau.flux_des_ailettes(echange, 40.0, 175.0) == pytest.approx(
        130.2, abs=0.3
    )
