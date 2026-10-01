"""La couche plane avec generation interne : le tas de charbon du TP 1.

Le point de modelisation qui fait tout : **le sol adiabatique n'est pas un
autre probleme**. Le corrige officiel traite les deux cas sur des planches
separees, l'un avec `dt/dx = 0` en x = 0 et l'autre avec une perte vers le
sol. Ce sont les deux valeurs d'un meme parametre -- la condition de Robin
`lambda dt/dx = K (t - t_sol)` donne exactement `dt/dx = 0` quand K = 0.
Un seul modele, donc, et le cas (a) tombe comme cas limite du cas (b).

L'autre point est une convention de signe, et le cours du depot a deja
paye pour celle-la en mecanique des fluides : ici `x` monte du sol vers la
surface exposee, et la densite de flux dans le sens de `x` vaut
`-lambda dt/dx`. Les deux conditions aux limites s'ecrivent donc avec des
signes opposes, et s'y tromper deplace le maximum d'un bord a l'autre.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.thermique.domain.conduction.plan import CoucheAvecGeneration
from polytech.thermique.domain.conduction.symboles import ABSCISSE, CONSTANTES
from polytech.thermique.domain.echanges import CoefficientDEchange, DensiteDeFlux
from polytech.thermique.domain.grandeurs import Temperature
from polytech.thermique.domain.materiaux import (
    Conductivite,
    Epaisseur,
    GenerationVolumique,
)


def _couche(
    echange_bas: float,
    generation: float = 40.0,
) -> CoucheAvecGeneration:
    """Le tas de charbon de l'enonce, parametre par son echange au sol."""
    return CoucheAvecGeneration(
        conductivite=Conductivite(watts_par_m_k=0.2),
        generation=GenerationVolumique(watts_par_m3=generation),
        epaisseur=Epaisseur(metres=2.0),
        echange_bas=CoefficientDEchange(watts_par_m2_k=echange_bas),
        temperature_bas=Temperature.en_celsius(10.0),
        echange_haut=CoefficientDEchange(watts_par_m2_k=25.0),
        temperature_haut=Temperature.en_celsius(20.0),
        flux_absorbe=DensiteDeFlux(watts_par_m2=0.95 * 150.0),
    )


SOL_ADIABATIQUE = _couche(echange_bas=0.0)
SOL_PERMEABLE = _couche(echange_bas=1.0)


def test_la_loi_locale_est_celle_de_fourier_kirchhoff_stationnaire() -> None:
    """0 = lambda d2t/dx2 + q, une fois les hypotheses appliquees."""
    loi = SOL_ADIABATIQUE.loi_locale()
    profil = SOL_ADIABATIQUE.profil_general()
    assert sp.simplify(loi.lhs.subs(sp.Function("t")(ABSCISSE), profil).doit()) == 0


def test_le_profil_general_est_une_parabole_a_deux_constantes() -> None:
    """Deux integrations, deux constantes : il faut deux conditions aux limites."""
    profil = SOL_ADIABATIQUE.profil_general()
    assert sp.degree(sp.Poly(profil, ABSCISSE)) == 2
    assert set(CONSTANTES) <= profil.free_symbols


def test_un_sol_adiabatique_annule_la_premiere_constante() -> None:
    """C'est le cas (a) : aucun flux ne sort par le bas, donc dt/dx(0) = 0."""
    constantes = SOL_ADIABATIQUE.constantes()
    assert constantes[CONSTANTES[0]] == pytest.approx(0.0)


def test_le_cas_adiabatique_retrouve_les_valeurs_du_corrige_officiel() -> None:
    """Le corrige annonce t_max = 428,9 degres a x = 0 m."""
    assert SOL_ADIABATIQUE.abscisse_du_maximum() == pytest.approx(0.0)
    maximum = SOL_ADIABATIQUE.temperature_maximale()
    assert maximum.celsius == pytest.approx(428.9, abs=0.05)


def test_le_cas_avec_pertes_retrouve_les_quatre_valeurs_du_corrige() -> None:
    """Le corrige annonce C1 = 189,7, t_sol = 47,94, t_surf = 27,38, x = 0,9486."""
    constantes = SOL_PERMEABLE.constantes()
    assert float(constantes[CONSTANTES[0]]) == pytest.approx(189.7, abs=0.05)
    assert SOL_PERMEABLE.temperature_en(0.0).celsius == pytest.approx(47.94, abs=0.005)
    assert SOL_PERMEABLE.temperature_en(2.0).celsius == pytest.approx(27.38, abs=0.005)
    assert SOL_PERMEABLE.abscisse_du_maximum() == pytest.approx(0.9486, abs=0.0001)
    maximum = SOL_PERMEABLE.temperature_maximale()
    assert maximum.celsius == pytest.approx(137.93, abs=0.005)


def test_la_chaleur_generee_se_partage_entre_les_deux_faces() -> None:
    """Recoupement independant : rien ne s'accumule en regime stationnaire.

    La couche genere `q e` par metre carre. Tout doit sortir, par le haut
    ou par le bas -- et ce bilan n'a servi a resoudre aucune des deux
    conditions aux limites. S'il se ferme, c'est une verification, pas une
    tautologie.
    """
    genere = 40.0 * 2.0
    vers_le_sol = SOL_PERMEABLE.flux_vers_le_bas()
    vers_le_ciel = SOL_PERMEABLE.flux_conductif_au_sommet()
    assert vers_le_sol + vers_le_ciel == pytest.approx(genere)


def test_le_sol_adiabatique_ne_laisse_rien_passer() -> None:
    """Le meme bilan, au cas limite : tout sort par le haut."""
    assert SOL_ADIABATIQUE.flux_vers_le_bas() == pytest.approx(0.0)
    assert SOL_ADIABATIQUE.flux_conductif_au_sommet() == pytest.approx(80.0)


def test_le_bilan_de_la_surface_exposee_se_ferme() -> None:
    """Second recoupement : ce qui arrive en surface est ce qui en part."""
    arrive = SOL_PERMEABLE.flux_conductif_au_sommet() + 0.95 * 150.0
    part = (
        25.0
        * SOL_PERMEABLE.temperature_en(2.0)
        .ecart_a(
            Temperature.en_celsius(20.0),
        )
        .kelvins
    )
    assert arrive == pytest.approx(part)


def test_sans_generation_interne_il_n_y_a_pas_de_sommet() -> None:
    """Le profil devient affine : son maximum est sur un bord, pas dedans.

    Le dire en levant evite qu'une division par q rende silencieusement un
    `inf` que le document composerait tel quel.
    """
    couche = _couche(echange_bas=1.0, generation=0.0)
    with pytest.raises(ValueError, match="sans generation interne"):
        couche.abscisse_du_sommet()
