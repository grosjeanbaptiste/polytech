"""Les macros du corrige enrichi de l'activite 2 : hypotheses et recoupements.

`macros_activite_2` publie les resultats des trois exercices ; ce module
publie ce qui les eprouve -- chaque hypothese chiffree sur les donnees du
cours, chaque resultat refait par un schema numerique.
"""

from __future__ import annotations

from polytech.fluides.application import activite_2 as a2
from polytech.fluides.application import activite_2_hypotheses as hyp
from polytech.fluides.application import activite_2_recoupements as rec
from polytech.fluides.application import activite_2_surface as surface
from polytech.fluides.application.formatage import nombre
from polytech.fluides.domain.statique.gaz import CONSTANTE_DES_GAZ

_POURCENT = 100.0
_MILLIMETRE = 1e3


def _scientifique(valeur: float, decimales: int) -> str:
    """`1.83e-5` plutot que `1.83e-05` : la forme que siunitx lit."""
    mantisse, exposant = f"{valeur:.{decimales}e}".split("e")
    return f"{mantisse}e{int(exposant)}"


def _film() -> dict[str, str]:
    return {
        "FilmMasseVolumiqueHuile": nombre(hyp.HUILE_MASSE_VOLUMIQUE, 0),
        "FilmViscositeHuileMilli": nombre(1e3 * hyp.HUILE_VISCOSITE, 0),
        "FilmViscositeAir": _scientifique(hyp.AIR_VISCOSITE, 2),
        "FilmRapportViscosites": _scientifique(hyp.rapport_des_viscosites(), 1),
        "FilmMasseVolumiqueAir": nombre(hyp.masse_volumique_de_l_air(), 2),
        "FilmCorrectionPousseePourcent": nombre(
            _POURCENT * hyp.rapport_des_masses_volumiques(), 2
        ),
        "FilmReynoldsMillimetre": nombre(hyp.reynolds_du_film(1e-3), 2),
        "FilmEpaisseurReynoldsUnite": nombre(
            _MILLIMETRE * hyp.epaisseur_a_reynolds_unite(), 2
        ),
    }


def _reservoir() -> dict[str, str]:
    def erreur(rapport: float) -> float:
        return _POURCENT * surface.borne_de_la_vitesse_de_surface(rapport)

    return {
        "ReservoirErreurDixiemePourcent": nombre(erreur(0.1), 2),
        "ReservoirErreurTiersPourcent": nombre(erreur(1 / 3), 2),
    }


def _bateau() -> dict[str, str]:
    elasticites = hyp.elasticites_du_bateau()
    etroite, large = hyp.ecarts_pour_un_millimetre_de_buse()
    return {
        "BateauElasticiteDiametre": nombre(elasticites["diametre"], 1),
        "BateauElasticiteDebit": nombre(elasticites["debit"], 1),
        "BateauEcartBuseEtroitePourcent": nombre(_POURCENT * etroite, 1),
        "BateauEcartBuseLargePourcent": nombre(_POURCENT * large, 1),
    }


def _recoupements() -> dict[str, str]:
    return {
        "RecoupementProfil": _scientifique(rec.ecart_du_profil(), 1),
        "RecoupementDebit": _scientifique(rec.ecart_du_debit(), 1),
        "RecoupementRemplissage": _scientifique(rec.ecart_du_remplissage(), 1),
        "RecoupementPuissance": _scientifique(rec.ecart_de_la_puissance(), 1),
    }


def _constantes() -> dict[str, str]:
    """Les donnees que le document cite : elles sortent de la bibliotheque."""
    return {
        "MasseVolumiqueEauDeMer": nombre(a2.EAU_DE_MER, 0),
        "PressionAtmospherique": nombre(hyp.AIR_PRESSION, 0),
        "ConstanteDesGaz": nombre(CONSTANTE_DES_GAZ, 3),
    }


def macros_complements() -> dict[str, str]:
    """Hypotheses chiffrees et recoupements, pour les trois exercices."""
    return {
        **_film(),
        **_reservoir(),
        **_bateau(),
        **_recoupements(),
        **_constantes(),
    }
