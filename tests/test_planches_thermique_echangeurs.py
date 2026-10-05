"""Les planches de thermique : les echangeurs de chaleur (jeu 11).

Source : `ThermiqueFondements/SLIDES/Slides_11_Echangeurs.pdf`, 33 planches
lues sur les pages rendues. Elles reprennent le chapitre 2 de la seconde
partie du tome 2, y ajoutent la classification des echangeurs et deux
applications dont elles ne donnent pas le resultat.

Les oracles : les compacites des planches 8 a 11, les remarques sur la
moyenne logarithmique (planches 17 et 18), les limites de l'efficacite
(planches 23 a 25). Les deux applications sont menees par deux chemins
independants, qui doivent s'accorder.
"""

from __future__ import annotations

import math

import pytest

from polytech.maths.domain.point_fixe.approximations_successives import (
    approximations_successives,
)
from polytech.thermique.domain.conduction.ailette_droite import parametre_d_ailette
from polytech.thermique.domain.conduction.ailettes import AiletteDroite
from polytech.thermique.domain.convection.forcee import coefficient_d_echange, reynolds
from polytech.thermique.domain.echangeurs import (
    Echangeur,
    efficacite_a_contre_courant,
    efficacite_a_courants_paralleles,
    flux_maximal_echangeable,
    moyenne_logarithmique,
)
from polytech.thermique.domain.echangeurs_a_courants_paralleles import (
    Extremites,
    Sens,
    debit_condense,
    flux_du_bilan,
    surface_d_echange,
)
from polytech.thermique.domain.paroi_de_separation import (
    Reference,
    coefficient_global_de_tube,
)
from polytech.thermique.domain.rendement_d_echangeur import (
    efficacite,
    rendement,
    sortie_du_fluide_chaud,
)
from polytech.thermique.domain.surface_ailetee import SurfaceAiletee
from polytech.thermique.domain.tables.air_sec import air_sec
from polytech.thermique.domain.tables.eau_saturee import (
    eau_saturee,
    temperature_de_saturation,
)
from polytech.thermique.domain.types_d_echangeurs import (
    COMPACITES,
    Contact,
    Ecoulement,
    Realisation,
    compacite,
    realisations_de_compacite,
)


def test_les_trois_modes_de_mise_en_contact() -> None:
    """Planches 4 a 6 : paroi de separation, contact direct, masse accumulatrice."""
    assert len(Contact) == 3
    assert Contact.INDIRECT.separe_les_fluides
    assert not Contact.DIRECT.separe_les_fluides
    assert Contact.PERIODIQUE.separe_les_fluides


def test_les_configurations_d_ecoulement() -> None:
    """Planche 12 : profils a une dimension en courants paralleles, deux en croises."""
    assert Ecoulement.COCOURANT.dimensions_des_profils == 1
    assert Ecoulement.CONTRE_COURANT.dimensions_des_profils == 1
    assert Ecoulement.COURANTS_CROISES.dimensions_des_profils == 2


def test_les_compacites_des_realisations() -> None:
    """Planches 8, 9 et 11 : 10 a 200, 200 a 400, 100 a 1000 m^2/m^3."""
    assert COMPACITES[Realisation.TUBES_ET_VIROLE] == (10.0, 200.0)
    assert COMPACITES[Realisation.PLAQUES] == (200.0, 400.0)
    assert COMPACITES[Realisation.PLAQUES_ET_AILETTES] == (100.0, 1000.0)
    assert compacite(1000.0, 5.0) == pytest.approx(200.0)
    assert realisations_de_compacite(600.0) == (Realisation.PLAQUES_ET_AILETTES,)
    assert realisations_de_compacite(300.0) == (
        Realisation.PLAQUES,
        Realisation.PLAQUES_ET_AILETTES,
    )
    assert realisations_de_compacite(5000.0) == ()
    assert realisations_de_compacite(200.0) == tuple(Realisation)
    with pytest.raises(ValueError, match="volume nul ou negatif"):
        compacite(1000.0, 0.0)


def test_les_bornes_de_la_moyenne_logarithmique() -> None:
    """Planche 18 : entre le plus petit et le plus grand ecart, sous leur moyenne."""
    for premier, second in ((80.0, 10.0), (50.0, 40.0), (12.0, 60.0)):
        moyenne = moyenne_logarithmique(premier, second)
        assert min(premier, second) <= moyenne <= max(premier, second)
        assert moyenne <= (premier + second) / 2
    assert moyenne_logarithmique(30.0, 30.0) == pytest.approx(30.0)


def test_le_contre_courant_est_le_plus_performant() -> None:
    """Planche 17 : `Dt_log,cc >= Dt_log,co`, donc `S_cc <= S_co`."""
    extremites = Extremites(110.0, 70.0, 30.0, 60.0)
    contre = extremites.moyenne_logarithmique(Sens.CONTRECOURANT)
    co = extremites.moyenne_logarithmique(Sens.COCOURANT)
    assert contre >= co
    assert surface_d_echange(1e4, 300.0, contre) <= surface_d_echange(1e4, 300.0, co)


def test_le_flux_maximal_et_la_definition_de_l_efficacite() -> None:
    """Planches 20 et 21 : `Phi_max = (q c)_min (t_1E - t_2E)`, et `epsilon`."""
    maximal = flux_maximal_echangeable(1000.0, 3000.0, 110.0, 30.0)
    assert maximal == pytest.approx(80000.0)
    epsilon = efficacite(Sens.CONTRECOURANT, 1500.0, (1000.0, 3000.0))
    sortie = sortie_du_fluide_chaud(110.0, 30.0, epsilon)
    assert flux_du_bilan(1000.0, 1.0, 110.0 - sortie) == pytest.approx(
        epsilon * maximal
    )


def test_les_relations_de_l_efficacite() -> None:
    """Planches 23 a 25 : `NUT / (1 + NUT)` a `Z = 1` ; a l'infini, `1 / (1 + Z)`."""
    assert efficacite_a_contre_courant(2.0, 1.0) == pytest.approx(2.0 / 3.0)
    assert efficacite_a_courants_paralleles(60.0, 1 / 3) == pytest.approx(0.75)
    assert efficacite_a_contre_courant(60.0, 1 / 3) == pytest.approx(1.0)
    for unites in (0.5, 1.0, 3.0):
        for rapport in (0.2, 1 / 3, 1.0):
            assert efficacite_a_contre_courant(
                unites, rapport
            ) >= efficacite_a_courants_paralleles(unites, rapport)
        assert rendement(Sens.COCOURANT, unites, 0.0) == pytest.approx(
            1 - math.exp(-unites)
        )


HUILE, EAU, CONDUCTANCE = 1.89 * 1883.5, 1.13 * 4180.0, 300.0 * 14.03
"""L'exercice 2 du tome 2 (page 133) : un echangeur existant, a contre-courant."""


def _nouvelle_sortie(sortie: float) -> float:
    """Planche 19, etapes 2 a 6 : bilans, `Dt_log`, transfert, nouvelle `t_1S`."""
    flux = flux_du_bilan(HUILE, 1.0, 110.0 - sortie)
    extremites = Extremites(110.0, sortie, 37.8, 37.8 + flux / EAU)
    transfere = Echangeur(CONDUCTANCE).flux(*extremites.ecarts(Sens.CONTRECOURANT))
    return 110.0 - transfere / HUILE


def test_les_approximations_successives_de_la_planche_19_divergent_ici() -> None:
    """Le schema, applique tel quel a l'exercice 2 du syllabus, s'ecarte a chaque tour.

    La planche ne le dit pas : chaque tour renvoie de l'autre cote de la
    solution, plus loin qu'il n'etait. Moyenner l'ancienne et la
    nouvelle estimation suffit a le faire converger, vers la sortie que
    donne l'efficacite.
    """
    attendue = sortie_du_fluide_chaud(
        110.0, 37.8, rendement(Sens.CONTRECOURANT, CONDUCTANCE / HUILE, HUILE / EAU)
    )
    assert _nouvelle_sortie(attendue) == pytest.approx(attendue)
    depart = attendue + 1.0
    premier = _nouvelle_sortie(depart)
    second = _nouvelle_sortie(premier)
    assert premier < attendue < second
    assert second - attendue > depart - attendue
    relaxee, _ = approximations_successives(
        lambda t: (t + _nouvelle_sortie(t)) / 2, 80.0, tolerance=1e-9
    )
    assert relaxee == pytest.approx(attendue, abs=1e-7)


def test_le_chauffage_d_eau_par_de_la_vapeur() -> None:
    """Planche 27 : 2,4 kg/s de 70 a 90 degres, vapeur a 1,2 bar, tubes de 14/16 mm.

    Les planches ne donnent pas le resultat. Le dimensionnement par la
    moyenne logarithmique et le calcul de l'echangeur obtenu par son
    efficacite doivent se rejoindre : 16 tubes de 3,2 m, 0,09 kg/s.
    """
    eau = eau_saturee(80.0)
    flux = flux_du_bilan(2.4, eau.chaleur_massique, 90.0 - 70.0)
    assert debit_condense(flux, 2251e3) == pytest.approx(0.09, abs=0.002)
    vapeur = temperature_de_saturation(1.2e5)
    ecart = Extremites(vapeur, vapeur, 70.0, 90.0).moyenne_logarithmique(
        Sens.CONTRECOURANT
    )
    echange = coefficient_global_de_tube(
        5000.0,
        15000.0,
        rayons=(0.007, 0.008),
        conductivite=380.0,
        reference=Reference.INTERIEURE,
    )
    assert echange == pytest.approx(3830.0, rel=5e-3)
    surface = surface_d_echange(flux, echange, ecart)
    masse_volumique = eau.viscosite / eau.viscosite_cinematique
    tubes = round(2.4 / (masse_volumique * 1.0 * math.pi / 4 * 0.014**2))
    assert tubes == 16
    assert surface / (tubes * math.pi * 0.014) == pytest.approx(3.2, abs=0.05)
    unites = echange * surface / (2.4 * eau.chaleur_massique)
    sortie = 70.0 + rendement(Sens.COCOURANT, unites, 0.0) * (vapeur - 70.0)
    assert sortie == pytest.approx(90.0, abs=1e-9)


AILETTES, DIAMETRE, HAUTEUR, LARGEUR = 13 * 13, 1.925e-3, 21.5e-3, 50e-3


def test_les_donnees_du_refroidisseur_se_recoupent() -> None:
    """Planche 31 : `A_f = H L = 1075 mm^2`, `q_a = 0,00215 m^3/s`, 13 pas."""
    frontale = HAUTEUR * LARGEUR
    assert frontale == pytest.approx(1075e-6)
    assert 2.0 * frontale == pytest.approx(0.00215)
    assert pytest.approx(LARGEUR, abs=1e-4) == 13 * 3.85e-3
    assert pytest.approx(2.0) == 3.85e-3 / DIAMETRE


def test_le_refroidisseur_de_microprocesseur() -> None:
    """Planches 30 a 32 : 40 W, air a 25 degres et 2 m/s, `Nu = 0,858 Re^0,5 Pr^1/3`.

    Les planches ne donnent pas le resultat. La base, isotherme, joue le
    fluide a temperature constante d'un echangeur : `epsilon = 1 -
    exp(-NUT)`. Le bilan sur l'air et la moyenne logarithmique doivent
    s'accorder : base a 47 degres, 0,55 K/W.
    """
    air = air_sec(273.15 + 30.0)
    nombre = reynolds(2.0, DIAMETRE, air.viscosite_cinematique())
    nusselt = 0.858 * nombre**0.5 * air.prandtl ** (1 / 3)
    echange = coefficient_d_echange(nusselt, air.conductivite, DIAMETRE)
    parametre = parametre_d_ailette(
        echange, math.pi * DIAMETRE, 210.0, math.pi * DIAMETRE**2 / 4
    )
    omega = AiletteDroite(parametre=parametre, longueur=HAUTEUR).efficacite()
    assert 0.75 < omega < 0.85
    surfaces = SurfaceAiletee(
        primaire=LARGEUR**2,
        libre=LARGEUR**2 - AILETTES * math.pi * DIAMETRE**2 / 4,
        ailettes=AILETTES * math.pi * DIAMETRE * HAUTEUR,
    )
    conductance = LARGEUR**2 / surfaces.resistance(echange, omega)
    capacite = air.masse_volumique * 0.00215 * air.chaleur_massique
    epsilon = rendement(Sens.COCOURANT, conductance / capacite, 0.0)
    echauffement = 40.0 / (epsilon * capacite)
    assert 25.0 + echauffement == pytest.approx(47.0, abs=1.0)
    assert echauffement / 40.0 == pytest.approx(0.55, abs=0.03)
    sortie = 25.0 + 40.0 / capacite
    base = 25.0 + echauffement
    moyenne = moyenne_logarithmique(base - 25.0, base - sortie)
    assert conductance * moyenne == pytest.approx(40.0)
