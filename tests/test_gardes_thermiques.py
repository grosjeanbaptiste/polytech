"""Chaque garde de la bibliotheque thermique, declenchee pour de vrai.

Une garde que nul test ne declenche est du code jamais execute : le jour
ou elle se declenche en production, rien ne garantit qu'elle fasse ce
qu'on croit. L'audit du depot l'exige, et une de ces gardes porte une
physique qui valait d'etre nommee -- une couche isolee sur ses deux faces
et qui produit de la chaleur n'a aucun regime permanent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import sympy as sp

from polytech.thermique.application import enonce
from polytech.thermique.application.formatage import nombre
from polytech.thermique.domain.conduction import plan_symbolique
from polytech.thermique.domain.conduction.plan import CoucheAvecGeneration
from polytech.thermique.domain.echanges import (
    CoefficientDEchange,
    ResistanceThermique,
)
from polytech.thermique.domain.geometrie import Rayon
from polytech.thermique.domain.grandeurs import Temperature
from polytech.thermique.domain.materiaux import (
    Conductivite,
    Epaisseur,
    GenerationVolumique,
)
from polytech.thermique.infrastructure.latex import FichierDeMacros

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def test_une_couche_isolee_des_deux_cotes_n_a_pas_de_regime_permanent() -> None:
    """Elle produit de la chaleur et n'a aucune sortie : elle chauffe sans fin.

    Les deux conditions aux limites se contredisent alors -- celle du bas
    impose `C_1 = 0`, celle du haut impose `lambda C_1 = q e + varphi > 0`.
    Le systeme n'a pas de solution, et c'est la bonne reponse : il n'existe
    pas d'etat stationnaire a trouver.
    """
    etouffee = CoucheAvecGeneration(
        conductivite=enonce.CONDUCTIVITE_CHARBON,
        generation=enonce.GENERATION_CHARBON,
        epaisseur=enonce.EPAISSEUR_CHARBON,
        echange_bas=CoefficientDEchange(watts_par_m2_k=0.0),
        temperature_bas=enonce.TEMPERATURE_SOL,
        echange_haut=CoefficientDEchange(watts_par_m2_k=0.0),
        temperature_haut=enonce.TEMPERATURE_AIR,
        flux_absorbe=enonce.FLUX_ABSORBE,
    )
    with pytest.raises(ValueError, match="les conditions aux limites"):
        etouffee.constantes()


def test_une_generation_volumique_negative_est_refusee() -> None:
    """Une source qui absorbe n'est pas une source : ce serait un puits."""
    with pytest.raises(ValueError, match="generation volumique negative"):
        GenerationVolumique(watts_par_m3=-1.0)


def test_un_nombre_de_decimales_negatif_est_refuse() -> None:
    """Le format s'en servirait sans broncher et rendrait n'importe quoi."""
    with pytest.raises(ValueError, match="nombre de decimales negatif"):
        nombre(1.0, -1)


def test_publier_zero_macro_est_refuse(tmp_path: Path) -> None:
    """Un fragment vide compilerait, et chaque valeur du corrige serait vide."""
    with pytest.raises(ValueError, match="aucune macro a publier"):
        FichierDeMacros(chemin=tmp_path / "vide.tex").publier({})


def test_un_nom_de_macro_non_alphabetique_est_refuse(tmp_path: Path) -> None:
    r"""`\newcommand{\Flux_1}` ne compile pas : la faute doit sortir ici."""
    with pytest.raises(ValueError, match="nom de macro LaTeX invalide"):
        FichierDeMacros(chemin=tmp_path / "x.tex").publier({"Flux_1": "1"})


def test_la_garde_du_systeme_symbolique_parle_si_sympy_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Celle-ci ne se declenche par aucune donnee : elle protege d'un amont.

    Le systeme symbolique est lineaire et non degenere, donc sympy lui
    trouve toujours une solution unique -- aujourd'hui. La garde existe
    pour le jour ou ce ne serait plus vrai, et ce test la declenche en
    faisant mentir `solve`, faute de pouvoir la declencher autrement.
    """
    monkeypatch.setattr(sp, "solve", lambda *_args, **_kwargs: [])
    with pytest.raises(ValueError, match="le systeme symbolique"):
        plan_symbolique.constantes()


def test_la_garde_du_reservoir_parle_si_sympy_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Meme nature, meme raison : un amont qui cesserait de tenir sa parole."""
    monkeypatch.setattr(sp, "solve", lambda *_args, **_kwargs: [])
    with pytest.raises(ValueError, match="les conditions aux limites"):
        enonce.RESERVOIR.flux_par_resolution_directe()


def test_une_temperature_de_sortie_reste_une_temperature_valide() -> None:
    """La paroi exterieure est construite par soustraction : elle peut mordre.

    Si la chute dans le film depassait l'ambiance absolue, l'objet
    leverait au lieu de rendre une temperature negative. Ici elle ne la
    depasse pas, et c'est ce que le test constate.
    """
    paroi = enonce.RESERVOIR.temperature_de_paroi_exterieure()
    assert paroi.kelvins > 0
    assert paroi.kelvins < Temperature.en_kelvins(300.0).kelvins


@pytest.mark.parametrize(
    "fabrique",
    [
        lambda: Conductivite(watts_par_m_k=0.0),
        lambda: Epaisseur(metres=-1.0),
        lambda: Rayon(metres=0.0),
        lambda: ResistanceThermique(kelvins_par_watt=0.0),
    ],
)
def test_une_grandeur_qui_divise_refuse_le_zero_et_le_negatif(
    fabrique: Callable[[], object],
) -> None:
    """Message commun : « nulle ou negative interdite ».

    Ces quatre grandeurs sont des denominateurs quelque part dans le TP.
    La garde est partagee, et c'est son message exact que ce test
    declenche -- les tests par grandeur, eux, nomment la grandeur.
    """
    with pytest.raises(ValueError, match="nulle ou negative interdite"):
        fabrique()
