"""Le syllabus de thermique, tome 2 : le coefficient global de transmission.

Source : `ThermiqueFondements/SYLLABUS/Syllabus_Tome_2_2013.pdf`, pages 97
a 111 (deuxieme partie, chapitre 1, sections 1.1 a 1.6).

Les oracles : les relations (1.2) a (1.13), le tableau de `K_sigma`, les
deux exemples du rayonnement (5,1 et 21 a 84 W/m^2 K), et les trois
exemples de la section 1.4 -- 98,5 %, 39 450 W/m^2 et 214 degres, puis
28 300 W/m^2 et 483 degres avec le depot.
"""

from __future__ import annotations

import math

import pytest
from scipy.optimize import brentq

from polytech.maths.domain.fausse_position import racine_par_fausse_position
from polytech.thermique.domain.coefficient_global import (
    CONSTANTE_DE_STEFAN,
    coefficient_de_rayonnement_exact,
    coefficient_de_rayonnement_linearise,
    coefficient_global_plan,
)
from polytech.thermique.domain.conduction.ailettes import AiletteDroite
from polytech.thermique.domain.continuite_du_flux import temperature_superficielle
from polytech.thermique.domain.convection.forcee import coefficient_d_echange
from polytech.thermique.domain.convection.naturelle import (
    grashof,
    nusselt_du_cylindre_horizontal,
)
from polytech.thermique.domain.ordres_de_grandeur import (
    ORDRES_DE_GRANDEUR,
    Phenomene,
    phenomene_limitatif,
)
from polytech.thermique.domain.paroi_de_separation import (
    Reference,
    chutes_partielles,
    coefficient_global_de_tube,
    coefficient_global_rapporte,
)
from polytech.thermique.domain.rayonnement.corps_noir import STEFAN_BOLTZMANN
from polytech.thermique.domain.rayonnement_equivalent import (
    coefficient_du_gaz,
    coefficient_du_gaz_linearise,
    coefficient_entre_surfaces,
    coefficient_sigma,
)
from polytech.thermique.domain.surface_ailetee import (
    SurfaceAiletee,
    coefficient_global_de_surface_ailetee,
    coefficient_global_de_tube_ailete,
)
from polytech.thermique.domain.tables.air_sec import air_sec


def test_la_paroi_plane_additionne_trois_resistances() -> None:
    """(1.1) et (1.2) : `1 / K_gl = 1 / K_1 + e / lambda + 1 / K_2`."""
    resistances = (1 / 50.0, 0.004 / 45.0, 1 / 5000.0)
    assert coefficient_global_plan(50.0, 0.004, 45.0, 5000.0) == pytest.approx(
        1 / sum(resistances)
    )


def test_le_tube_se_rapporte_a_l_une_ou_l_autre_surface() -> None:
    """(1.3) a (1.5) : `K_gl,e S_e = K_gl,i S_i`, le flux ne depend pas du choix."""
    rayons = (0.010, 0.014)
    externe = coefficient_global_de_tube(
        5000.0, 50.0, rayons=rayons, conductivite=45.0, reference=Reference.EXTERIEURE
    )
    interne = coefficient_global_de_tube(
        5000.0, 50.0, rayons=rayons, conductivite=45.0, reference=Reference.INTERIEURE
    )
    inverse = 0.014 / (5000 * 0.010) + 0.014 / 45 * math.log(1.4) + 1 / 50
    assert externe == pytest.approx(1 / inverse)
    assert externe * rayons[1] == pytest.approx(interne * rayons[0])
    assert interne > externe
    with pytest.raises(ValueError, match="rayon exterieur non superieur"):
        coefficient_global_de_tube(
            5000.0,
            50.0,
            rayons=(0.014, 0.010),
            conductivite=45.0,
            reference=Reference.EXTERIEURE,
        )


def test_un_tube_mince_se_ramene_a_la_paroi_plane() -> None:
    """Page 99 : « pour les tubes minces, on peut l'assimiler a e / lambda »."""
    mince = coefficient_global_de_tube(
        150.0,
        80.0,
        rayons=(1.0, 1.001),
        conductivite=45.0,
        reference=Reference.EXTERIEURE,
    )
    plan = coefficient_global_plan(150.0, 0.001, 45.0, 80.0)
    assert mince == pytest.approx(plan, rel=1e-3)


def test_la_paroi_quelconque_contient_les_deux_autres() -> None:
    """(1.6) : `1 / K_gl = S_ref / (K_1 S_1) + R_p S_ref + S_ref / (K_2 S_2)`."""
    plan = coefficient_global_rapporte(
        2.0, cotes=((50.0, 2.0), (5000.0, 2.0)), resistance_de_paroi=0.004 / 45 / 2
    )
    assert plan == pytest.approx(coefficient_global_plan(50.0, 0.004, 45.0, 5000.0))
    interieure, exterieure = 2 * math.pi * 0.010, 2 * math.pi * 0.014
    paroi = math.log(1.4) / (2 * math.pi * 45.0)
    tube = coefficient_global_rapporte(
        exterieure,
        cotes=((5000.0, interieure), (50.0, exterieure)),
        resistance_de_paroi=paroi,
    )
    assert tube == pytest.approx(
        coefficient_global_de_tube(
            5000.0,
            50.0,
            rayons=(0.010, 0.014),
            conductivite=45.0,
            reference=Reference.EXTERIEURE,
        )
    )
    with pytest.raises(ValueError, match="surface de reference nulle ou negative"):
        coefficient_global_rapporte(0.0, cotes=((1.0, 1.0),), resistance_de_paroi=0.0)


def test_les_ordres_de_grandeur_de_la_section_1_2() -> None:
    """Page 100 et 101 : de 2 a 5 dans l'air calme, jusqu'a 50 000 en ebullition."""
    assert ORDRES_DE_GRANDEUR[Phenomene.CONVECTION_NATURELLE_DANS_L_AIR] == (2.0, 5.0)
    assert ORDRES_DE_GRANDEUR[Phenomene.EAU_TURBULENTE_EN_TUBE] == (4000.0, 8000.0)
    assert ORDRES_DE_GRANDEUR[Phenomene.EBULLITION_DE_L_EAU] == (15000.0, 50000.0)
    assert ORDRES_DE_GRANDEUR[Phenomene.CONDENSATION_DE_LA_VAPEUR] == (8000.0, 8000.0)
    assert len(ORDRES_DE_GRANDEUR) == len(Phenomene)
    plus_faible = min(ORDRES_DE_GRANDEUR, key=lambda p: ORDRES_DE_GRANDEUR[p][1])
    plus_fort = max(ORDRES_DE_GRANDEUR, key=lambda p: ORDRES_DE_GRANDEUR[p][0])
    assert plus_faible is Phenomene.CONVECTION_NATURELLE_DANS_L_AIR
    assert plus_fort is Phenomene.EBULLITION_DE_L_EAU


def test_les_gaz_echangent_moins_que_les_liquides() -> None:
    """Page 101 : « toujours des coefficients beaucoup plus faibles »."""
    gaz = ORDRES_DE_GRANDEUR[Phenomene.GAZ_TURBULENT_EN_TUBE]
    eau = ORDRES_DE_GRANDEUR[Phenomene.EAU_TURBULENTE_EN_TUBE]
    travers = ORDRES_DE_GRANDEUR[Phenomene.GAZ_EN_TRAVERS_D_UN_TUBE]
    assert gaz[1] < eau[0]
    assert travers[0] > gaz[1]


def test_le_tableau_de_k_sigma() -> None:
    """Page 104 : 5,7 a 20 degres, 104,8 a 500, 468 a 1000 -- `4 sigma T^3`."""
    for kelvins, attendu in ((293.1, 5.7), (773.1, 104.8), (1273.1, 468.0)):
        assert coefficient_sigma(kelvins, kelvins) == pytest.approx(attendu, rel=2e-3)
    exact = coefficient_sigma(320.0, 280.0)
    assert exact == pytest.approx(STEFAN_BOLTZMANN * (320.0**4 - 280.0**4) / 40.0)
    assert exact == pytest.approx(coefficient_sigma(280.0, 320.0))
    assert exact == pytest.approx(4 * STEFAN_BOLTZMANN * 300.0**3, rel=5e-3)
    with pytest.raises(ValueError, match="temperature absolue nulle ou negative"):
        coefficient_sigma(0.0, 300.0)


def test_la_constante_de_stefan_n_est_ecrite_qu_une_fois() -> None:
    """Le chapitre emploie le `sigma` du rayonnement : une seule valeur."""
    assert CONSTANTE_DE_STEFAN is STEFAN_BOLTZMANN


def test_le_rayonnement_entre_surfaces_grises() -> None:
    """(1.7), (1.8) et (1.10) : `K_r = F sigma (T_p^4 - T_s^4) / (T_p - T_s)`."""
    gris = coefficient_entre_surfaces(0.9, 320.0, 280.0)
    assert gris == pytest.approx(0.9 * coefficient_sigma(320.0, 280.0))
    assert gris == pytest.approx(coefficient_de_rayonnement_exact(0.9, 320.0, 280.0))
    lineaire = coefficient_de_rayonnement_linearise(0.9, 320.0, 280.0)
    assert lineaire == pytest.approx(0.9 * coefficient_sigma(300.0, 300.0))
    with pytest.raises(ValueError, match="facteur de rayonnement hors de"):
        coefficient_entre_surfaces(1.2, 320.0, 280.0)


def test_l_exemple_des_objets_a_la_temperature_ambiante() -> None:
    """Page 104 : `epsilon = 0,9` donne 5,1 W/m^2 K, l'ordre de la convection."""
    rayonnement = coefficient_entre_surfaces(0.9, 293.1, 293.1)
    assert rayonnement == pytest.approx(5.1, abs=0.05)
    bas, haut = ORDRES_DE_GRANDEUR[Phenomene.CONVECTION_NATURELLE_DANS_L_AIR]
    assert bas < rayonnement < haut + 0.5


def test_le_rayonnement_d_un_gaz_emetteur() -> None:
    """(1.9) et (1.11) : `K_r = 2 (eps_p + 1) eps_g sigma T_gp^3` a petit ecart."""
    exact = coefficient_du_gaz(
        0.8, emission=0.1, absorption=0.12, gaz=1300.0, paroi=700.0
    )
    numerateur = 0.1 * 1300.0**4 - 0.12 * 700.0**4
    assert exact == pytest.approx(0.9 * STEFAN_BOLTZMANN * numerateur / 600.0)
    proche = coefficient_du_gaz(
        0.8, emission=0.1, absorption=0.1, gaz=1001.0, paroi=999.0
    )
    lineaire = coefficient_du_gaz_linearise(0.8, 0.1, 1000.0)
    assert lineaire == pytest.approx(2 * 1.8 * 0.1 * STEFAN_BOLTZMANN * 1e9)
    assert proche == pytest.approx(lineaire, rel=1e-5)
    with pytest.raises(ValueError, match="le gaz et la paroi sont a la meme"):
        coefficient_du_gaz(0.8, emission=0.1, absorption=0.1, gaz=900.0, paroi=900.0)


def test_l_exemple_des_fumees_a_mille_degres() -> None:
    """Page 104 : `eps_g` de 0,05 a 0,2 et `eps_p = 0,8` donnent 21 a 84 W/m^2 K."""
    assert coefficient_du_gaz_linearise(0.8, 0.05, 1273.1) == pytest.approx(
        21.0, abs=0.1
    )
    assert coefficient_du_gaz_linearise(0.8, 0.2, 1273.1) == pytest.approx(
        84.0, abs=0.3
    )
    bas, haut = ORDRES_DE_GRANDEUR[Phenomene.GAZ_TURBULENT_EN_TUBE]
    assert bas <= coefficient_du_gaz_linearise(0.8, 0.05, 1273.1) <= haut


def test_les_chutes_partielles_sont_comme_les_resistances() -> None:
    """Section 1.4, exemple 1 : 98,5 % de la resistance est cote gaz."""
    resistances = (1 / 50.0, 0.004 / 45.0, 1 / 5000.0)
    assert resistances[0] == pytest.approx(2e-2)
    assert resistances[1] == pytest.approx(0.01e-2, abs=0.002e-2)
    assert resistances[2] == pytest.approx(0.02e-2)
    chutes = chutes_partielles(100.0, resistances)
    assert sum(chutes) == pytest.approx(100.0)
    assert chutes[0] == pytest.approx(98.5, abs=0.1)
    coefficients = {"gaz": 50.0, "eau": 5000.0}
    assert phenomene_limitatif(coefficients) == "gaz"
    with pytest.raises(ValueError, match="resistance negative"):
        chutes_partielles(100.0, (1.0, -1.0))
    with pytest.raises(ValueError, match="aucun coefficient"):
        phenomene_limitatif({})


FUMEES, EBULLITION = 1200.0, 200.0
ACIER = 40.0
"""La conductivite n'est pas donnee ; 40 W/m K rend les 39 450 et les 214."""


def test_l_exemple_2_le_tube_de_fumees() -> None:
    """Page 106 : `K_gl ~ K_i = 40`, 39 450 W/m^2, paroi a 214 degres au plus."""
    resistances = (1 / 40.0, 0.010 / ACIER, 1 / 10000.0)
    flux = (FUMEES - EBULLITION) / sum(resistances)
    assert flux == pytest.approx(39450.0, abs=5.0)
    assert 1 / sum(resistances) == pytest.approx(40.0, rel=0.015)
    chutes = chutes_partielles(FUMEES - EBULLITION, resistances)
    assert EBULLITION + chutes[1] + chutes[2] == pytest.approx(214.0, abs=0.3)
    avec_45 = (FUMEES - EBULLITION) / (1 / 40.0 + 0.010 / 45.0 + 1 / 10000.0)
    assert abs(avec_45 - 39450.0) > 30.0


def test_l_exemple_2_avec_un_depot_cote_eau() -> None:
    """Page 107 : 2 mm a 0,2 W/m K font tomber le flux a 28 300 W/m^2.

    Les « 483 degres » sont `200 + phi e' / lambda'` : la chute dans le
    seul depot. La face de la paroi cote eau est a 486 degres avec le
    film d'ebullition, et sa face cote fumees a 493.
    """
    resistances = (1 / 40.0, 0.010 / ACIER, 0.002 / 0.2, 1 / 10000.0)
    flux = (FUMEES - EBULLITION) / sum(resistances)
    assert flux == pytest.approx(28300.0, abs=15.0)
    chutes = chutes_partielles(FUMEES - EBULLITION, resistances)
    assert EBULLITION + chutes[2] == pytest.approx(483.0, abs=0.5)
    assert EBULLITION + chutes[2] + chutes[3] == pytest.approx(485.7, abs=0.2)
    assert FUMEES - chutes[0] == pytest.approx(492.8, abs=0.2)


def test_l_exemple_3_le_prechauffeur_d_huile() -> None:
    """Page 107 : `K_gl,i ~ K_i`, et la paroi est a la saturation de la vapeur."""
    assert phenomene_limitatif({"huile": 150.0, "vapeur": 5000.0}) == "huile"
    chutes = chutes_partielles(100.0, (1 / 150.0, 1 / 5000.0))
    assert chutes[1] == pytest.approx(2.9, abs=0.05)
    bas, haut = ORDRES_DE_GRANDEUR[Phenomene.HUILE_LAMINAIRE_EN_TUBE]
    assert bas == haut == pytest.approx(150.0)


def test_la_fausse_position_trouve_la_racine() -> None:
    """Page 110 : « par la methode de Mueller ou regula falsi »."""

    def fonction(x: float) -> float:
        return x**3 - 2 * x - 5

    racine = racine_par_fausse_position(fonction, 2.0, 3.0)
    assert racine == pytest.approx(brentq(fonction, 2.0, 3.0), abs=1e-9)
    assert racine_par_fausse_position(fonction, 3.0, 2.0) == pytest.approx(racine)
    assert racine_par_fausse_position(lambda x: x - 2.0, 2.0, 3.0) == 2.0
    assert racine_par_fausse_position(lambda x: x - 3.0, 2.0, 3.0) == 3.0
    with pytest.raises(ValueError, match="pas de changement de signe"):
        racine_par_fausse_position(fonction, 3.0, 4.0)
    with pytest.raises(ValueError, match="pas de convergence"):
        racine_par_fausse_position(fonction, 2.0, 3.0, iterations=2)


def test_la_fausse_position_ne_s_enlise_pas_sur_une_fonction_convexe() -> None:
    """Sans alleger la borne immobile, il faut 120 tours ; avec, 42.

    `x^10 - 1` laisse la borne haute immobile, son image retournee la
    borne basse : chacune des deux corrections est eprouvee.
    """
    convexe = racine_par_fausse_position(lambda x: x**10 - 1, 0.0, 1.3, iterations=60)
    assert convexe == pytest.approx(1.0, abs=1e-12)
    concave = racine_par_fausse_position(
        lambda x: 1 - (1.3 - x) ** 10, 0.0, 1.3, iterations=60
    )
    assert concave == pytest.approx(0.3, abs=1e-12)


COQUILLE = (0.029, 0.079)
EAU, AMBIANCE = 90.0, 20.0


def _conductivite(moyenne: float) -> float:
    return 0.0326 + (0.0558 - 0.0326) * moyenne / 100.0


def _entrant(paroi: float) -> float:
    interne, externe = COQUILLE
    isolant = _conductivite((EAU + paroi) / 2)
    return (EAU - paroi) / (externe / isolant * math.log(externe / interne))


def _sortant(paroi: float) -> float:
    film = air_sec(273.15 + (paroi + AMBIANCE) / 2)
    nombre = grashof(
        1 / (273.15 + (paroi + AMBIANCE) / 2),
        paroi - AMBIANCE,
        2 * COQUILLE[1],
        film.viscosite_cinematique(),
    )
    nusselt = nusselt_du_cylindre_horizontal(nombre * film.prandtl)
    convection = coefficient_d_echange(nusselt, film.conductivite, 2 * COQUILLE[1])
    rayonnement = coefficient_entre_surfaces(0.9, 273.15 + paroi, 273.15 + AMBIANCE)
    return (convection + rayonnement) * (paroi - AMBIANCE)


def test_la_tuyauterie_calorifugee_par_continuite_du_flux() -> None:
    """(1.13) et figure 1.11 : `F_1` decroit, `F_2` croit, un seul croisement."""
    paroi = temperature_superficielle(_entrant, _sortant, (AMBIANCE + 0.01, EAU))
    assert AMBIANCE < paroi < EAU
    assert _entrant(paroi) == pytest.approx(_sortant(paroi), rel=1e-8)
    assert _entrant(paroi - 1) > _entrant(paroi) > _entrant(paroi + 1)
    assert _sortant(paroi - 1) < _sortant(paroi) < _sortant(paroi + 1)
    lineique = 2 * math.pi * COQUILLE[1] * _sortant(paroi)
    assert lineique == pytest.approx(
        2
        * math.pi
        * _conductivite((EAU + paroi) / 2)
        * (EAU - paroi)
        / math.log(COQUILLE[1] / COQUILLE[0])
    )


def test_a_coefficients_constants_la_continuite_rend_le_diviseur() -> None:
    """Section 1.4 : sans dependance en `t_p`, c'est le partage des resistances."""
    paroi = temperature_superficielle(
        lambda t: 2.0 * (EAU - t), lambda t: 8.0 * (t - AMBIANCE), (AMBIANCE, EAU)
    )
    chutes = chutes_partielles(EAU - AMBIANCE, (1 / 2.0, 1 / 8.0))
    assert paroi == pytest.approx(EAU - chutes[0])


SURFACES = SurfaceAiletee(primaire=1.0, libre=0.8, ailettes=6.0)


def test_la_surface_ailetee_compense_le_gaz() -> None:
    """Section 1.6 : la resistance du gaz devient `S_p / ((S_1o + Omega S_1a) K_1)`."""
    ailette = AiletteDroite(
        parametre=math.sqrt(2 * 50.0 / (45.0 * 0.002)), longueur=0.03
    )
    efficacite = ailette.efficacite()
    assert efficacite == pytest.approx(math.tanh(1.0) / 1.0)
    ailete = coefficient_global_de_surface_ailetee(
        5000.0, 50.0, paroi=(0.004, 45.0), surfaces=SURFACES, efficacite=efficacite
    )
    inverse = 1 / 5000 + 0.004 / 45 + 1 / ((0.8 + efficacite * 6.0) * 50.0)
    assert ailete == pytest.approx(1 / inverse)
    lisse = coefficient_global_de_surface_ailetee(
        5000.0,
        50.0,
        paroi=(0.004, 45.0),
        surfaces=SurfaceAiletee(primaire=1.0, libre=1.0, ailettes=0.0),
        efficacite=1.0,
    )
    assert lisse == pytest.approx(coefficient_global_plan(5000.0, 0.004, 45.0, 50.0))
    assert ailete > 5 * lisse
    assert SURFACES.totale == pytest.approx(6.8)
    with pytest.raises(ValueError, match="efficacite hors de"):
        coefficient_global_de_surface_ailetee(
            5000.0, 50.0, paroi=(0.004, 45.0), surfaces=SURFACES, efficacite=1.2
        )
    with pytest.raises(ValueError, match="surface primaire nulle ou negative"):
        SurfaceAiletee(primaire=0.0, libre=0.8, ailettes=6.0)
    with pytest.raises(ValueError, match="surface negative"):
        SurfaceAiletee(primaire=1.0, libre=0.8, ailettes=-6.0)


def test_le_tube_a_ailettes_circulaires() -> None:
    """Page 111, exercice complementaire : rapporte a la surface interieure."""
    interieure = 2 * math.pi * 0.010
    surfaces = SurfaceAiletee(primaire=interieure, libre=0.07, ailettes=0.9)
    ailete = coefficient_global_de_tube_ailete(
        5000.0,
        50.0,
        rayons=(0.010, 0.014),
        conductivite=45.0,
        surfaces=surfaces,
        efficacite=0.8,
    )
    inverse = (
        1 / 5000 + 0.010 / 45 * math.log(1.4) + interieure / ((0.07 + 0.8 * 0.9) * 50.0)
    )
    assert ailete == pytest.approx(1 / inverse)
    nu = coefficient_global_de_tube_ailete(
        5000.0,
        50.0,
        rayons=(0.010, 0.014),
        conductivite=45.0,
        surfaces=SurfaceAiletee(interieure, 2 * math.pi * 0.014, 0.0),
        efficacite=1.0,
    )
    assert nu == pytest.approx(
        coefficient_global_de_tube(
            5000.0,
            50.0,
            rayons=(0.010, 0.014),
            conductivite=45.0,
            reference=Reference.INTERIEURE,
        )
    )
