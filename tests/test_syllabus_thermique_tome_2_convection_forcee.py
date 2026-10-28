"""Le syllabus de thermique, tome 2 : quelques cas de convection forcee.

Source : `ThermiqueFondements/SYLLABUS/Syllabus_Tome_2_2013.pdf`, pages 33
a 48 (premiere partie, chapitre 3), lues sur les pages rendues.

Les oracles : les relations (3.1) a (3.13), le tableau de la figure 3.8,
celui de `f_n`, et les exercices III.3.1 a III.3.3 -- 7930 W/m^2 K,
133,1 W/m^2 K et 575 W/m, 160,1 W/m^2 K.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.thermique.domain.convection.canal_plan import diametre_hydraulique
from polytech.thermique.domain.convection.definitions import coefficient_moyen
from polytech.thermique.domain.convection.faisceau_de_tubes import (
    FACTEUR_DES_AILETTES_LONGITUDINALES,
    Disposition,
    facteur_d_etablissement,
    facteur_d_interaction,
    nusselt_du_faisceau,
    vitesse_maximale,
)
from polytech.thermique.domain.convection.forcee import (
    coefficient_d_echange,
    nusselt_cylindre,
    nusselt_plaque,
    reynolds,
)
from polytech.thermique.domain.convection.plaque_plane import (
    constante_de_raccord,
    epaisseur_turbulente,
    nusselt_local_laminaire,
    nusselt_moyen_a_constante,
    nusselt_moyen_mixte,
    rapport_de_la_sous_couche,
    rapport_des_couches_limites,
)
from polytech.thermique.domain.convection.tube_laminaire import (
    NUSSELT_ETABLI,
    groupe_d_entree,
    longueur_d_entree_thermique,
    nusselt_de_hausen,
    nusselt_de_sieder_et_tate,
)
from polytech.thermique.domain.convection.tube_turbulent import (
    diametre_hydraulique_de_la_section,
    facteur_j,
    nusselt_de_colburn,
    nusselt_de_transition,
    nusselt_turbulent_de_sieder_et_tate,
    sous_couche_en_tube,
)
from polytech.thermique.domain.tables.air_sec import air_sec
from polytech.thermique.domain.tables.eau_saturee import eau_saturee


def test_la_couche_limite_thermique_suit_le_prandtl() -> None:
    """Page 35 : `delta_t / delta = Pr^-1/3` ; un pour un gaz, 0,1 pour une huile."""
    assert rapport_des_couches_limites(1.0) == 1.0
    assert rapport_des_couches_limites(0.7) == pytest.approx(1.13, abs=0.01)
    assert rapport_des_couches_limites(1000.0) == pytest.approx(0.1)


def test_le_nusselt_local_et_le_nusselt_moyen_d_une_plaque() -> None:
    """(3.1) et (3.2) : `Nu_x = 0,332 Re_x^0,5 Pr^1/3`, et `K_moyen = 2 K_l`."""
    assert nusselt_local_laminaire(1e4, 0.7) == pytest.approx(
        0.332 * 100 * 0.7 ** (1 / 3)
    )
    assert nusselt_plaque(1e4, 0.7) == pytest.approx(
        2 * nusselt_local_laminaire(1e4, 0.7)
    )
    abscisse, longueur = sp.symbols("x L", positive=True)
    local = sp.Rational(332, 1000) * sp.sqrt(abscisse) / abscisse
    moyen = coefficient_moyen(local, abscisse, longueur)
    assert sp.simplify(moyen - 2 * local.subs(abscisse, longueur)) == 0
    with pytest.raises(ValueError, match="couche limite turbulente"):
        nusselt_local_laminaire(1e6, 0.7)


def test_la_couche_limite_turbulente_et_sa_sous_couche() -> None:
    """Page 35 : `delta / x = 0,376 Re_x^-0,2`, `delta_l / delta = 194 Re_x^-0,7`.

    Leur produit est le `72,94 Re_x^-0,9` du chapitre 1.
    """
    assert epaisseur_turbulente(2.0, 1e6) == pytest.approx(0.376 * 2.0 * 1e6**-0.2)
    assert rapport_de_la_sous_couche(1e6) == pytest.approx(194 * 1e6**-0.7)
    assert pytest.approx(72.94, abs=0.005) == 0.376 * 194
    double = epaisseur_turbulente(4.0, 2e6) / epaisseur_turbulente(2.0, 1e6)
    assert double == pytest.approx(2**0.8)


def test_le_nusselt_moyen_d_une_plaque_en_partie_turbulente() -> None:
    """Page 36 : `Nu_L = 0,037 Pr^1/3 (Re_L^0,8 - 23 100)` si `Re_crit = 5e5`.

    Raccorder (3.2) a la loi turbulente en `Re_crit = 5e5` donne
    23 550 ; le syllabus imprime 23 100.
    """
    constante = constante_de_raccord(5e5)
    assert constante == pytest.approx(23549.0, abs=0.5)
    mixte = nusselt_moyen_mixte(2e6, 0.7)
    assert mixte == pytest.approx(nusselt_moyen_a_constante(2e6, 0.7, constante))
    imprime = nusselt_moyen_a_constante(2e6, 0.7, 23100.0)
    assert mixte / imprime - 1 == pytest.approx(-0.0052, abs=1e-4)
    # Au raccord, les deux lois se rejoignent : la constante C le garantit.
    assert nusselt_moyen_mixte(5e5, 0.7) == pytest.approx(
        nusselt_moyen_a_constante(5e5, 0.7, constante), rel=1e-12
    )
    # En laminaire, la moyenne sur la plaque vaut le double du local au bord de fuite.
    court = nusselt_moyen_mixte(1e5, 0.7)
    assert court == pytest.approx(2 * nusselt_local_laminaire(1e5, 0.7))
    tout_turbulent = nusselt_moyen_a_constante(1e8, 0.7, 0.0)
    assert nusselt_moyen_mixte(1e8, 0.7) == pytest.approx(tout_turbulent, rel=0.01)


def test_l_entree_thermique_d_un_tube() -> None:
    """(3.5) : `L_e,th / d = 0,05 Re_d Pr`, soit 1,7 Pr fois l'entree hydrodynamique."""
    thermique = longueur_d_entree_thermique(0.014, 450.0, 177.7)
    assert thermique == pytest.approx(0.05 * 450.0 * 177.7 * 0.014)
    hydrodynamique = 0.03 * 450.0 * 0.014
    assert thermique / hydrodynamique == pytest.approx(1.7 * 177.7, rel=0.02)


def test_le_nusselt_etabli_et_la_theorie_de_hausen() -> None:
    """(3.6) : 3,65 loin de l'entree, davantage sur un tube court."""
    assert NUSSELT_ETABLI == 3.65
    assert nusselt_de_hausen(1e-9) == pytest.approx(3.65)
    entree = groupe_d_entree(450.0, 177.7, 0.014, 2.0)
    with pytest.raises(ValueError, match="longueur nulle ou negative"):
        groupe_d_entree(450.0, 177.7, 0.014, 0.0)
    assert entree == pytest.approx(559.8, abs=0.05)
    attendu = 3.65 + 0.0668 * entree / (1 + 0.04 * entree ** (2 / 3))
    assert nusselt_de_hausen(entree) == pytest.approx(attendu)
    assert nusselt_de_hausen(entree, viscosites=2.0) == pytest.approx(
        attendu * 2.0**0.14
    )


def test_la_relation_de_sieder_et_tate_vaut_pour_les_tubes_courts() -> None:
    """(3.7) : `Nu = 1,86 (Re Pr d/L)^1/3 (mu_m/mu_p)^0,14`.

    Le syllabus imprime la condition `(1 / Re Pr)(d / L) < 0,1`. C'est
    `L / d` : hors de `Re Pr d / L > 10`, la relation passe sous le
    Nusselt etabli, ce qui n'a pas de sens.
    """
    assert nusselt_de_sieder_et_tate(10.0) == pytest.approx(
        nusselt_de_hausen(10.0), rel=0.05
    )
    assert nusselt_de_sieder_et_tate(10.0) == pytest.approx(4.0, abs=0.05)
    hors_domaine = nusselt_de_sieder_et_tate(1.0, hors_validite=True)
    assert hors_domaine == pytest.approx(1.86)
    assert hors_domaine < NUSSELT_ETABLI
    with pytest.raises(ValueError, match="tube trop long"):
        nusselt_de_sieder_et_tate(1.0)
    assert 1 / groupe_d_entree(450.0, 177.7, 0.014, 2.0) < 0.1


def test_la_sous_couche_visqueuse_d_un_tube() -> None:
    """Page 38 : `delta_l / d = 63,5 Re_d^-0,8` ; quelques dixiemes de millimetre."""
    epaisseur = sous_couche_en_tube(0.014, 46128.0)
    assert epaisseur == pytest.approx(63.5 * 0.014 * 46128.0**-0.8)
    assert 1e-4 < epaisseur < 3e-4


def test_les_relations_du_tube_turbulent() -> None:
    """(3.8) et (3.9) : Colburn, `0,0234 Re^0,8 Pr^1/3` ; Sieder et Tate, `0,027`."""
    assert nusselt_de_colburn(1e5, 0.7) == pytest.approx(0.0234 * 1e4 * 0.7 ** (1 / 3))
    visqueux = nusselt_turbulent_de_sieder_et_tate(1e5, 50.0, viscosites=2.0)
    assert visqueux == pytest.approx(0.027 * 1e4 * 50.0 ** (1 / 3) * 2.0**0.14)
    with pytest.raises(ValueError, match="ecoulement non turbulent"):
        nusselt_de_colburn(2000.0, 0.7)


def test_la_zone_de_transition() -> None:
    """Page 39 : `0,116 (Re^2/3 - 125) Pr^1/3 [1 + (d/L)^2/3] (mu_m/mu_p)^0,14`.

    Elle reste sous la loi turbulente, a 5 % pres vers `Re = 2e4`.
    """
    transition = nusselt_de_transition(5000.0, 4.0, elancement=0.01)
    coeur = 0.116 * (5000.0 ** (2 / 3) - 125) * 4.0 ** (1 / 3)
    assert transition == pytest.approx(coeur * (1 + 0.01 ** (2 / 3)))
    visqueux = nusselt_de_transition(5000.0, 4.0, elancement=0.01, viscosites=2.0)
    assert visqueux == pytest.approx(transition * 2.0**0.14)
    long = nusselt_de_transition(2e4, 4.0, elancement=1e-9)
    turbulent = nusselt_turbulent_de_sieder_et_tate(2e4, 4.0)
    assert long / turbulent == pytest.approx(0.95, abs=0.01)


def test_le_facteur_j_efface_la_viscosite() -> None:
    """Page 40 : `j_H = Nu Pr^-1/3 (mu_m/mu_p)^-0,14 / Re`, soit `0,027 Re^-0,2`.

    Le syllabus imprime l'exposant `+0,14` ; l'axe de sa figure 3.6
    porte `-0,14`, et c'est avec lui que `j_H` ne depend plus que de `Re`.
    """
    for viscosites in (0.5, 1.0, 3.0):
        nusselt = nusselt_turbulent_de_sieder_et_tate(1e5, 30.0, viscosites=viscosites)
        facteur = facteur_j(nusselt, 1e5, 30.0, viscosites=viscosites)
        assert facteur == pytest.approx(0.027 * 1e5**-0.2)
        avec_le_signe_plus = facteur_j(
            nusselt, 1e5, 30.0, viscosites=viscosites, exposant=0.14
        )
        if viscosites != 1.0:
            assert avec_le_signe_plus != pytest.approx(facteur, rel=1e-3)
    assert facteur == pytest.approx(0.0027, abs=5e-6)


def test_le_diametre_hydraulique() -> None:
    """(3.10) : `d_hyd = 4 A / p` ; le diametre d'un cercle, le cote d'un carre."""
    cercle = diametre_hydraulique_de_la_section(math.pi * 0.05**2 / 4, math.pi * 0.05)
    assert cercle == pytest.approx(0.05)
    assert diametre_hydraulique_de_la_section(0.04**2, 4 * 0.04) == pytest.approx(0.04)
    large = diametre_hydraulique_de_la_section(0.01 * 1e6, 2 * 1e6)
    assert large == pytest.approx(diametre_hydraulique(0.01))
    with pytest.raises(ValueError, match="perimetre nul ou negatif"):
        diametre_hydraulique_de_la_section(1.0, 0.0)


@pytest.mark.parametrize(
    ("reynolds_local", "coefficient", "exposant"),
    [(2.0, 0.989, 0.330), (20.0, 0.911, 0.385), (1000.0, 0.683, 0.466),
     (1e4, 0.193, 0.618), (1e5, 0.0266, 0.805)],
)  # fmt: skip
def test_le_tableau_de_la_figure_3_8(
    reynolds_local: float, coefficient: float, exposant: float
) -> None:
    """(3.11) : `Nu_d = C Re_d^n Pr^1/3`, autour d'un cylindre isole."""
    attendu = coefficient * reynolds_local**exposant * 0.7 ** (1 / 3)
    assert nusselt_cylindre(reynolds_local, 0.7) == pytest.approx(attendu)


def test_la_vitesse_dans_la_section_la_plus_retrecie() -> None:
    """Page 42 : `u_max = u_o p_T / (p_T - d)`, entre deux tubes d'une rangee."""
    assert vitesse_maximale(6.0, pas=0.038, diametre=0.025) == pytest.approx(
        17.54, abs=0.005
    )
    with pytest.raises(ValueError, match="tubes jointifs"):
        vitesse_maximale(6.0, pas=0.025, diametre=0.025)


def test_le_facteur_d_etablissement_du_tableau() -> None:
    """Page 43 : `f_n` croit jusqu'a un, atteint a la dixieme rangee."""
    assert facteur_d_etablissement(1, Disposition.EN_FILES) == 0.64
    assert facteur_d_etablissement(5, Disposition.EN_FILES) == 0.92
    assert facteur_d_etablissement(1, Disposition.EN_QUINCONCE) == 0.68
    assert facteur_d_etablissement(6, Disposition.EN_QUINCONCE) == 0.95
    for disposition in Disposition:
        facteurs = [facteur_d_etablissement(n, disposition) for n in range(1, 11)]
        assert facteurs == sorted(facteurs)
        assert facteurs[-1] == 1.0
        assert facteur_d_etablissement(25, disposition) == 1.0
    with pytest.raises(ValueError, match="nombre de rangees nul ou negatif"):
        facteur_d_etablissement(0, Disposition.EN_FILES)


def test_les_facteurs_d_interaction_de_grimison() -> None:
    """(3.12) et (3.13) : `f_a` selon la maille."""
    files = facteur_d_interaction(Disposition.EN_FILES, 1.52, 1.52, reynolds=26183.0)
    assert files == pytest.approx(0.978, abs=5e-4)
    quinconce = facteur_d_interaction(Disposition.EN_QUINCONCE, 1.52, 1.52)
    assert quinconce == pytest.approx(1 + 0.1 * 1.52 + 0.34 / 1.52)
    assert quinconce > files
    with pytest.raises(ValueError, match="Reynolds requis"):
        facteur_d_interaction(Disposition.EN_FILES, 1.52, 1.52)
    assert FACTEUR_DES_AILETTES_LONGITUDINALES == 0.83


def test_l_exercice_iii_3_1_l_eau_dans_un_tuyau() -> None:
    """2 m/s, 14 mm, film a 45 degres : `Re = 46 128`, `Nu = 199,2`.

    Le syllabus conclut « K = 199,2 x 0,637 / 0,016 = 7930 » : il divise
    par 16 mm un Nusselt forme sur les 14 mm du diametre interieur.
    Avec 14 mm, 9063 W/m^2 K.
    """
    eau = eau_saturee((30.0 + 60.0) / 2)
    nombre = reynolds(2.0, 0.014, 0.607e-6)
    assert nombre == pytest.approx(46128.0, abs=1.0)
    assert reynolds(2.0, 0.014, eau.viscosite_cinematique) == pytest.approx(
        nombre, rel=2e-3
    )
    nusselt = nusselt_de_colburn(nombre, 3.95)
    assert nusselt == pytest.approx(199.2, abs=0.05)
    assert coefficient_d_echange(nusselt, 0.637, 0.016) == pytest.approx(
        7930.0, abs=1.0
    )
    assert coefficient_d_echange(nusselt, 0.637, 0.014) == pytest.approx(
        9063.0, abs=1.0
    )


def test_l_exercice_iii_3_2_l_air_a_travers_un_faisceau() -> None:
    """Cinq rangs en ligne, maille de 38 mm, tubes de 25 mm : 133,1 W/m^2 K, 575 W/m."""
    air = air_sec(273.15 + (10.0 + 65.0) / 2)
    masse_volumique = 1.287 * 273.1 / 310.6
    cinematique = 1.895e-5 / masse_volumique
    assert cinematique == pytest.approx(1.6746e-5, rel=1e-4)
    assert air.viscosite_cinematique() == pytest.approx(cinematique, rel=0.01)
    maximale = vitesse_maximale(6.0, pas=0.038, diametre=0.025)
    nombre = reynolds(maximale, 0.025, cinematique)
    assert nombre == pytest.approx(26183.0, abs=1.0)
    pas_reduit = 38 / 25
    interaction = facteur_d_interaction(
        Disposition.EN_FILES, pas_reduit, pas_reduit, reynolds=nombre
    )
    etablissement = facteur_d_etablissement(5, Disposition.EN_FILES)
    nusselt = nusselt_du_faisceau(
        Disposition.EN_FILES, nombre, 0.705, interaction * etablissement
    )
    assert nusselt == pytest.approx(122.8, abs=0.1)
    echange = coefficient_d_echange(nusselt, 0.0271, 0.025)
    assert echange == pytest.approx(133.1, abs=0.15)
    assert echange * math.pi * 0.025 * (65.0 - 10.0) == pytest.approx(575.0, abs=0.5)


def test_l_exercice_iii_3_3_l_huile_dans_un_tube_chauffe() -> None:
    """0,0574 kg/s dans un tube de 14 mm, 2 m : `Re = 450`, 160,1 W/m^2 K."""
    diffusivite = 0.13 / (829.0 * 1990.0)
    cinematique = 0.0116 / 829.0
    prandtl = cinematique / diffusivite
    assert prandtl == pytest.approx(177.7, abs=0.2)
    vitesse = 0.0574 / (829.0 * math.pi / 4 * 0.014**2)
    assert vitesse == pytest.approx(0.45, abs=0.001)
    nombre = reynolds(vitesse, 0.014, cinematique)
    assert nombre == pytest.approx(450.0, abs=0.1)
    nusselt = nusselt_de_sieder_et_tate(559.8, viscosites=2.32)
    assert nusselt == pytest.approx(17.25, abs=0.01)
    assert coefficient_d_echange(nusselt, 0.13, 0.014) == pytest.approx(160.1, abs=0.1)


def test_la_viscosite_a_la_paroi_de_l_exercice_iii_3_3() -> None:
    """« ln mu = f(1/T) est pratiquement lineaire » ; « mu_p = 50 cP ».

    Le rapport 2,32 de la ligne suivante suppose 5,0 cP. La loi que le
    syllabus enonce, entre 11,6 cP a 60 degres et 3,0 cP a 120, donne
    4,04 cP a 105 degres, soit un rapport de 2,87 et 3 % de plus sur `K`.
    """
    assert pytest.approx(5.0) == 11.6 / 2.32
    avancement = (1 / 378.15 - 1 / 333.15) / (1 / 393.15 - 1 / 333.15)
    logarithme = math.log(11.6) + avancement * (math.log(3.0) - math.log(11.6))
    a_la_paroi = math.exp(logarithme)
    assert a_la_paroi == pytest.approx(4.04, abs=0.005)
    imprime = nusselt_de_sieder_et_tate(559.8, viscosites=2.32)
    par_la_loi = nusselt_de_sieder_et_tate(559.8, viscosites=11.6 / a_la_paroi)
    assert par_la_loi / imprime == pytest.approx(1.03, abs=0.002)
