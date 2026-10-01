"""Macros de l'exercice 1 : le tas de charbon, dans ses deux cas.

Les formes **symboliques** et les valeurs **chiffrees** sont publiees
toutes les deux. Un corrige pas a pas doit montrer d'ou vient la formule
avant de l'appliquer : le lecteur qui ne verrait que `C_1 = 189,72`
n'apprend rien de transposable.

Les profils sont publies en degres Celsius, l'echelle de l'enonce pour cet
exercice, et arrondis -- afficher `189.719202898551` donnerait douze
chiffres que l'enonce ne fournit pas.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.thermique.application import enonce
from polytech.thermique.application.formatage import expression_latex, nombre
from polytech.thermique.domain.conduction import plan_symbolique
from polytech.thermique.domain.conduction.symboles import ABSCISSE, CONSTANTES

if TYPE_CHECKING:
    from polytech.thermique.domain.conduction.plan import CoucheAvecGeneration

SOMMET: float = enonce.EPAISSEUR_CHARBON.metres
DECIMALES_AFFICHEES = 2


def _coefficients_du_profil(couche: CoucheAvecGeneration, cas: str) -> dict[str, str]:
    r"""Les trois coefficients du profil, pour que la figure les trace.

    La figure ne peut pas consommer `\Profil...` : c'est du LaTeX
    mathematique, pas une expression que pgfplots sache evaluer. Publier
    les coefficients garde la figure sous la meme regle que le texte --
    aucun nombre tape a la main.
    """
    polynome = sp.Poly(couche.profil_en_celsius(), ABSCISSE)
    carre, lineaire, constant = (float(c) for c in polynome.all_coeffs())
    return {
        f"CoefCarre{cas}": nombre(carre, 4),
        f"CoefLineaire{cas}": nombre(lineaire, 4),
        f"CoefConstant{cas}": nombre(constant, 4),
    }


def _macros_du_cas(couche: CoucheAvecGeneration, cas: str) -> dict[str, str]:
    """Les resultats d'un des deux cas, en degres Celsius."""
    constantes = couche.constantes()
    return {
        **_coefficients_du_profil(couche, cas),
        f"ConstanteUn{cas}": nombre(float(constantes[CONSTANTES[0]]), 2),
        f"TemperatureBas{cas}": nombre(couche.temperature_en(0.0).celsius, 2),
        f"TemperatureSurface{cas}": nombre(couche.temperature_en(SOMMET).celsius, 2),
        f"AbscisseMax{cas}": nombre(couche.abscisse_du_maximum(), 4),
        f"TemperatureMax{cas}": nombre(couche.temperature_maximale().celsius, 2),
        f"FluxVersLeSol{cas}": nombre(couche.flux_vers_le_bas(), 2),
        f"FluxSommet{cas}": nombre(couche.flux_conductif_au_sommet(), 2),
        f"Profil{cas}": expression_latex(
            couche.profil_en_celsius(),
            DECIMALES_AFFICHEES,
        ),
    }


def _macros_symboliques() -> dict[str, str]:
    """Les cinq etapes de la derivation, avant toute substitution."""
    premiere, _ = CONSTANTES
    return {
        "LoiLocaleCharbon": expression_latex(plan_symbolique.loi_locale()),
        "ProfilGeneralCharbon": expression_latex(plan_symbolique.profil_general()),
        "GradientGeneralCharbon": expression_latex(
            plan_symbolique.gradient_general(),
        ),
        "ConditionBasCharbon": expression_latex(plan_symbolique.condition_au_bas()),
        "ConditionSommetCharbon": expression_latex(
            plan_symbolique.condition_au_sommet(),
        ),
        "ConstanteUnSymbolique": expression_latex(
            plan_symbolique.constantes()[premiere],
        ),
        "AbscisseSommetSymbolique": expression_latex(
            plan_symbolique.abscisse_du_sommet(),
        ),
    }


def macros_charbon() -> dict[str, str]:
    """Derivation symbolique, chaleur generee, et les deux cas chiffres."""
    generee = enonce.GENERATION_CHARBON.watts_par_m3 * enonce.EPAISSEUR_CHARBON.metres
    return {
        "ChaleurGeneree": nombre(generee, 0),
        **_macros_symboliques(),
        **_macros_du_cas(enonce.SOL_ADIABATIQUE, "Adiabatique"),
        **_macros_du_cas(enonce.SOL_PERMEABLE, "Permeable"),
    }
