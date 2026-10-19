"""Activite 2, le bateau-pompe : les hypotheses (H12) a (H14) et `alpha`, chiffrees.

La puissance a l'arbre est lineaire en chaque terme de la hauteur :
`P = rho g Q (dz + h_L + alpha V_2^2 / 2g - V_1^2 / 2g) / eta`. Le poids
d'un terme est donc sa part dans la hauteur, et le recoupement refait la
hauteur par `hauteur_manometrique`, sans passer par cette part.

- `alpha` : le cours prend un pour un ecoulement turbulent, profil plat
  (`Slides_03_Equations_fondamentales.pdf`, Bernoulli generalise). Le
  terme cinetique faisant presque toute la hauteur, c'est l'hypothese qui
  pese le plus ;
- (H12) : une mer qui bouge a `V_1` apporte `V_1^2 / 2g` ;
- (H14) : les pertes entrent une fois, par leurs trois metres.

La pesanteur est celle du cours de mecanique rationnelle 1, que la
mecanique des fluides importe : le cours de fluides n'en ecrit pas la
valeur.
"""

from __future__ import annotations

import math

from polytech.fluides.application import activite_2 as a2
from polytech.fluides.application.formatage import nombre
from polytech.fluides.domain.fondamentales.bernoulli_generalise import (
    hauteur_manometrique,
)
from polytech.mecanique.domain.pesanteur import PESANTEUR

_POURCENT = 100.0
_KILO = 1e3
_ALPHA_ILLUSTRE = 1.05
_MER_ILLUSTREE = 1.0


def _hauteur(alpha: float = 1.0, mer: float = 0.0) -> float:
    ejection = a2.bateau_pompe().ejection
    return hauteur_manometrique(
        denivele=a2.HAUTEUR_DE_LA_BUSE,
        vitesse_finale=math.sqrt(alpha) * ejection,
        vitesse_initiale=mer,
        pertes=a2.PERTES,
    )


def elasticite_du_facteur_cinetique() -> float:
    """`d ln P / d ln alpha` : la part cinetique de la hauteur."""
    bateau = a2.bateau_pompe()
    return bateau.ejection**2 / (2 * PESANTEUR) / bateau.hauteur


def ecart_pour_un_facteur_cinetique(alpha: float) -> float:
    """L'ecart relatif de puissance si le jet porte `alpha` et non un."""
    if alpha < 1:
        msg = f"facteur de correction cinetique sous un : {alpha}"
        raise ValueError(msg)
    return _hauteur(alpha=alpha) / _hauteur() - 1


def ecart_pour_une_mer_agitee(vitesse: float) -> float:
    """L'ecart relatif de puissance si la surface de la mer va a `vitesse`."""
    if vitesse < 0:
        msg = f"vitesse de la mer negative : {vitesse}"
        raise ValueError(msg)
    return _hauteur(mer=vitesse) / _hauteur() - 1


def elasticite_des_pertes() -> float:
    """`d ln P / d ln h_L` : la part des pertes dans la hauteur."""
    return a2.PERTES / a2.bateau_pompe().hauteur


def deux_chiffres(valeur: float) -> float:
    """L'arrondi a deux chiffres significatifs que les examens demandent."""
    return float(f"{valeur:.2g}")


def macros_bateau() -> dict[str, str]:
    """Les hypotheses chiffrees du bateau, et ses resultats a deux chiffres."""
    bateau = a2.bateau_pompe()
    return {
        "Pesanteur": nombre(PESANTEUR, 2),
        "BateauPuissanceArrondie": nombre(deux_chiffres(bateau.arbre / _KILO), 0),
        "BateauVitesseArrondie": nombre(deux_chiffres(bateau.ejection), 0),
        "BateauElasticiteAlpha": nombre(elasticite_du_facteur_cinetique(), 2),
        "BateauEcartAlphaPourcent": nombre(
            _POURCENT * ecart_pour_un_facteur_cinetique(_ALPHA_ILLUSTRE), 1
        ),
        "BateauElasticitePertes": nombre(elasticite_des_pertes(), 3),
        "BateauEcartMerPourcent": nombre(
            _POURCENT * ecart_pour_une_mer_agitee(_MER_ILLUSTREE), 3
        ),
    }
