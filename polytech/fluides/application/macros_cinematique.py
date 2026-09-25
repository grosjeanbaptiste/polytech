"""Macros de la partie II : cinematique des fluides.

Derivees partielles, loi de volume, tenseurs, lignes de courant et ecarts
materiels sont tous calcules par le domaine, jamais recopies.
"""

from __future__ import annotations

from polytech.fluides.application import enonce
from polytech.fluides.application.formatage import expression_latex

_AXE_X, _AXE_Y = 0, 1


def _macros_derivees() -> dict[str, str]:
    d = enonce.CHAMP_B.derivees_partielles()
    noms = {
        "DeriveeUDeT": d.du_dt,
        "DeriveeUDeX": d.du_dx,
        "DeriveeUDeY": d.du_dy,
        "DeriveeVDeT": d.dv_dt,
        "DeriveeVDeX": d.dv_dx,
        "DeriveeVDeY": d.dv_dy,
    }
    return {nom: expression_latex(expr) for nom, expr in noms.items()}


def _macros_deformation() -> dict[str, str]:
    tenseur = enonce.CHAMP_C.tenseur_taux_deformation()
    return {
        "TenseurB": expression_latex(enonce.CHAMP_B.tenseur_taux_deformation()),
        "TenseurC": expression_latex(tenseur),
        "EpsXXC": expression_latex(tenseur[0, 0]),
        "EpsYYC": expression_latex(tenseur[1, 1]),
        "EpsXYC": expression_latex(tenseur[0, 1]),
        "EcartMaterielCX": expression_latex(enonce.CHAMP_C.ecart_materiel(_AXE_X)),
        "EcartMaterielCY": expression_latex(enonce.CHAMP_C.ecart_materiel(_AXE_Y)),
    }


def macros_cinematique() -> dict[str, str]:
    """Rend les raisonnements litteraux de la cinematique."""
    gradient_x, _ = enonce.CHAMP_B.gradient_temporel()
    return {
        **_macros_derivees(),
        **_macros_deformation(),
        "GradientTemporelB": expression_latex(gradient_x),
        "LoiVolumeB": expression_latex(enonce.CHAMP_B.loi_de_volume()),
        "DivergenceC": expression_latex(enonce.CHAMP_C.divergence()),
        "TauxRotationC": expression_latex(enonce.CHAMP_C.taux_de_rotation()),
        "LignesCourantC": expression_latex(
            enonce.CHAMP_C.equation_lignes_de_courant(),
        ),
    }
