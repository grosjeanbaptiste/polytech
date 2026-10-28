"""Les classifications d'ecoulement, et la cascade de Kolmogorov.

Chapitre I du cours, meme fascicule que `proprietes` et meme raison
d'avoir manque : il n'etait pas declare.

Le critere d'incompressibilite est celui qui decide de la suite du cours.
La planche le donne sous **deux** formes a la fois : « les ecoulements
gazeux consideres comme incompressibles si les variations de masse
volumique < 5 % (Ma < 0,3) ». Les deux bornes vont ensemble sans que la
planche le justifie, et ce n'est pas evident -- l'une est cinematique,
l'autre thermodynamique. La relation isentropique les relie, et a
`Ma = 0,3` la variation relative de masse volumique vaut 4,4 %, juste
sous les cinq pour cent. Le critere n'est donc pas arbitraire, et c'est
ce qu'un test verifie.

La cascade de Kolmogorov vient avec son exposant, `-5/3`, et son sens :
injection d'energie aux grandes echelles, transfert, dissipation
visqueuse aux petites.
"""

from __future__ import annotations

from typing import Final

MACH_INCOMPRESSIBLE: Final = 0.3
"""En dessous, un ecoulement gazeux se traite comme incompressible."""

VARIATION_ADMISE: Final = 0.05
"""Les cinq pour cent de variation de masse volumique que la planche admet."""

GAMMA_DE_L_AIR: Final = 1.4
"""Le rapport des chaleurs massiques, qui relie les deux bornes du critere."""

EXPOSANT_DE_KOLMOGOROV: Final = -5 / 3
"""La pente du spectre d'energie dans la zone inertielle."""


def nombre_de_mach(vitesse: float, celerite: float) -> float:
    """`Ma = V / c` : la vitesse rapportee a celle du son."""
    if celerite <= 0:
        msg = f"celerite du son nulle ou negative : {celerite} m/s"
        raise ValueError(msg)
    return vitesse / celerite


def est_incompressible(mach: float) -> bool:
    """Vrai sous `Ma = 0,3`, soit 103 m/s dans l'air -- 370 km/h.

    L'ordre de grandeur est ce qui rend le critere utile : la ventilation,
    le genie civil et l'automobile courante y tiennent tous, et c'est
    pourquoi le cours peut traiter l'incompressible d'abord.
    """
    return abs(mach) < MACH_INCOMPRESSIBLE


def variation_de_masse_volumique(mach: float) -> float:
    """`(rho_0 - rho) / rho` : l'ecart d'arret rapporte a la valeur **locale**.

    `rho_0 / rho = (1 + (gamma-1) Ma^2 / 2)^(1/(gamma-1))`, dont on rend
    `rho_0 / rho - 1` : 4,56 % a `Ma = 0,3`. Ce n'est PAS le « 4,4 % » du
    cours, qui rapporte la variation a `rho_0` : voir
    `compressibilite.variation_relative_a_l_arret` (4,36 %). Aux petits
    Mach les deux croissent comme le **carre** du Mach -- ce qui explique
    la permissivite du seuil. Alias plus explicite : `ecart_rapporte_a_rho`.
    """
    if mach < 0:
        msg = f"nombre de Mach negatif : {mach}"
        raise ValueError(msg)
    rapport = (1 + (GAMMA_DE_L_AIR - 1) * mach**2 / 2) ** (1 / (GAMMA_DE_L_AIR - 1))
    return float(rapport - 1)


ecart_rapporte_a_rho = variation_de_masse_volumique
"""Meme fonction, nom qui dit la definition : `(rho_0 - rho) / rho`."""


def densite_spectrale_de_turbulence(nombre_d_onde: float, constante: float) -> float:
    """`E(k) = C k^(-5/3)` : le spectre de la zone inertielle.

    L'energie se repartit sur toutes les echelles entre l'injection et la
    dissipation, en decroissant vers les petites. C'est la raison du cout
    d'une simulation directe : il faut resoudre jusqu'a la dissipation.
    """
    if nombre_d_onde <= 0:
        msg = (
            f"nombre d'onde nul ou negatif : {nombre_d_onde} m^-1 -- la "
            f"cascade commence a l'echelle d'injection, qui est finie"
        )
        raise ValueError(msg)
    # L'exposant est un flottant : mypy rend `Any` d'une puissance a
    # exposant non entier, d'ou le `float()` explicite.
    return float(constante * nombre_d_onde**EXPOSANT_DE_KOLMOGOROV)
