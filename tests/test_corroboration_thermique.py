"""Trois chemins pour les memes nombres, et ils doivent se recouper.

Le corrige officiel du TP 1 donne six valeurs pour l'exercice 1 et aucune
pour l'exercice 2 -- sa planche de valeurs numeriques est une image dont
rien ne s'extrait. Le recoupement ne peut donc pas se contenter de « le
corrige dit la meme chose » : il faut des chemins independants.

* la resolution **chiffree**, qui substitue les donnees puis resout ;
* la resolution **symbolique**, qui resout puis substitue -- l'ordre
  inverse, et donc une autre suite d'operations de sympy ;
* pour le reservoir, le **schema en resistances** d'un cote et l'equation
  de la chaleur avec ses conditions aux limites de l'autre.

Trois chemins qui tombent sur le meme chiffre ne prouvent pas la physique,
mais ils excluent la faute d'algebre -- qui est le risque reel ici.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.thermique.application import enonce
from polytech.thermique.domain.conduction import plan_symbolique
from polytech.thermique.domain.conduction.symboles import ABSCISSE, CONSTANTES

DONNEES = {
    plan_symbolique.CONDUCTIVITE: enonce.CONDUCTIVITE_CHARBON.watts_par_m_k,
    plan_symbolique.GENERATION: enonce.GENERATION_CHARBON.watts_par_m3,
    plan_symbolique.EPAISSEUR: enonce.EPAISSEUR_CHARBON.metres,
    plan_symbolique.ECHANGE_HAUT: enonce.ECHANGE_EXTERIEUR.watts_par_m2_k,
    plan_symbolique.TEMPERATURE_HAUT: enonce.TEMPERATURE_AIR.kelvins,
    plan_symbolique.TEMPERATURE_BAS: enonce.TEMPERATURE_SOL.kelvins,
    plan_symbolique.FLUX_ABSORBE: enonce.FLUX_ABSORBE.watts_par_m2,
}


def _constante_symbolique(echange_bas: float) -> float:
    """C_1 tiree de la formule generale, donnees substituees apres coup."""
    premiere, _ = CONSTANTES
    formule = plan_symbolique.constantes()[premiere]
    valeurs = {**DONNEES, plan_symbolique.ECHANGE_BAS: echange_bas}
    return float(formule.subs(valeurs))


def test_la_formule_generale_redonne_la_constante_du_cas_avec_pertes() -> None:
    """Resoudre puis substituer, ou substituer puis resoudre : meme nombre."""
    chiffre = float(enonce.SOL_PERMEABLE.constantes()[CONSTANTES[0]])
    symbole = _constante_symbolique(enonce.ECHANGE_SOL.watts_par_m2_k)
    assert symbole == pytest.approx(chiffre, rel=1e-12)


def test_la_formule_generale_tend_vers_zero_quand_le_sol_devient_adiabatique() -> None:
    """La formule divise par K_b : c'est elle qui est singuliere, pas la physique.

    Quand `K_b = 0` la condition basse s'ecrit directement `lambda C_1 = 0`,
    sans aucune division. La limite de la formule generale doit donc valoir
    zero -- et c'est bien ce que vaut la constante du cas adiabatique.
    """
    premiere, _ = CONSTANTES
    formule = plan_symbolique.constantes()[premiere]
    limite = sp.limit(formule, plan_symbolique.ECHANGE_BAS, 0)
    assert sp.simplify(limite) == 0
    assert float(enonce.SOL_ADIABATIQUE.constantes()[premiere]) == pytest.approx(0.0)


def test_le_profil_symbolique_et_le_profil_chiffre_coincident_partout() -> None:
    """Pas seulement en un point : sur toute l'epaisseur de la couche."""
    abscisse = ABSCISSE
    premiere, seconde = CONSTANTES
    constantes = enonce.SOL_PERMEABLE.constantes()
    symbolique = plan_symbolique.profil_general().subs(
        {**DONNEES, premiere: constantes[premiere], seconde: constantes[seconde]},
    )
    chiffre = enonce.SOL_PERMEABLE.profil()
    for position in (0.0, 0.5, 1.0, 1.5, 2.0):
        attendu = float(chiffre.subs(abscisse, position))
        obtenu = float(symbolique.subs(abscisse, position))
        assert obtenu == pytest.approx(attendu, rel=1e-12)


def test_les_deux_chemins_du_reservoir_donnent_le_meme_flux() -> None:
    """Schema en resistances d'un cote, equation de la chaleur de l'autre."""
    par_schema = enonce.RESERVOIR.flux()
    par_equation = enonce.RESERVOIR.flux_par_resolution_directe()
    assert par_equation == pytest.approx(par_schema, rel=1e-12)


def test_le_bilan_global_du_tas_de_charbon_se_ferme() -> None:
    """Ce que la couche genere sort par ses deux faces, et rien ne s'accumule.

    Ce bilan n'a servi a resoudre aucune des deux conditions aux limites :
    il se referme ou il ne se referme pas.
    """
    genere = enonce.GENERATION_CHARBON.watts_par_m3 * enonce.EPAISSEUR_CHARBON.metres
    for couche in (enonce.SOL_ADIABATIQUE, enonce.SOL_PERMEABLE):
        sortant = couche.flux_vers_le_bas() + couche.flux_conductif_au_sommet()
        assert sortant == pytest.approx(genere, rel=1e-9)


def test_le_bilan_de_la_surface_exposee_se_ferme_dans_les_deux_cas() -> None:
    """Conduction plus solaire absorbe egalent ce que l'air emporte."""
    for couche in (enonce.SOL_ADIABATIQUE, enonce.SOL_PERMEABLE):
        arrive = couche.flux_conductif_au_sommet() + enonce.FLUX_ABSORBE.watts_par_m2
        surface = couche.temperature_en(enonce.EPAISSEUR_CHARBON.metres)
        ecart = surface.ecart_a(enonce.TEMPERATURE_AIR).kelvins
        part = enonce.ECHANGE_EXTERIEUR.watts_par_m2_k * ecart
        assert arrive == pytest.approx(part, rel=1e-9)


def test_le_sol_emporte_ce_que_sa_condition_limite_annonce() -> None:
    """Troisieme bilan : la perte vers le sol, vue des deux cotes."""
    couche = enonce.SOL_PERMEABLE
    ecart = couche.temperature_en(0.0).ecart_a(enonce.TEMPERATURE_SOL).kelvins
    par_la_condition = enonce.ECHANGE_SOL.watts_par_m2_k * ecart
    assert couche.flux_vers_le_bas() == pytest.approx(par_la_condition, rel=1e-9)
