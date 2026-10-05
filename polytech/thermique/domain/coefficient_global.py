"""Le coefficient global de transmission `K`, et la linearisation du rayonnement.

Chapitre 1 du tome 2 du syllabus, et clef de voute du tome : tout
dimensionnement de paroi, d'isolation ou d'echangeur s'y ramene. Absent
de la bibliotheque jusqu'ici parce que la mesure de couverture ne
regardait que les planches.

Deux idees, et la seconde est la plus utile du chapitre.

**Les resistances s'ajoutent, pas les coefficients.** `1/K` est la somme
des resistances en serie -- convection interieure, conduction de la
paroi, convection exterieure -- si bien que `K` est toujours plus petit
que le plus petit des coefficients. Additionner les coefficients donnerait
un `K` bien trop grand, donc une paroi qu'on croirait plus deperditive
qu'elle n'est. Et comme la plus **grande** resistance commande la somme,
ameliorer une resistance deja faible ne sert a rien : c'est le
raisonnement qui decide ou mettre l'isolant.

**Le rayonnement se linearise par la temperature moyenne.** Le syllabus
developpe `T_p^4 - T_a^4` en serie de Mac Laurin et note que mettre
`T_a^4` en evidence donnerait `4 T_a^3 (T_p - T_a)` ; « il est donc plus
exact de prendre » la temperature moyenne du systeme. Le coefficient est
alors « approximativement proportionnel a la troisieme puissance de la
temperature absolue moyenne ». Cette superiorite est **mesuree** par les
tests et non admise : l'erreur de la moyenne est d'ordre deux en l'ecart,
celle de l'ambiante d'ordre un.

Son utilite, le syllabus la nomme : « juger par un calcul rapide si le
rayonnement est important ou negligeable par rapport a un autre mode de
transmission ». A temperature ambiante il vaut quelques watts par metre
carre et par kelvin -- le meme ordre que la convection naturelle.
"""

from __future__ import annotations

from typing import Final

from polytech.thermique.domain.rayonnement.corps_noir import STEFAN_BOLTZMANN

CONSTANTE_DE_STEFAN: Final = STEFAN_BOLTZMANN
"""`sigma`, en W/(m^2 K^4) : celle du corps noir, ecrite une seule fois."""

FACTEUR_DE_LINEARISATION: Final = 4
"""Le `4` de `4 sigma T^3` : l'exposant de `T^4` passe en coefficient."""


def _exiger_le_rayonnement(emissivite: float, *temperatures: float) -> None:
    """Les gardes communes aux deux coefficients de rayonnement."""
    if not 0 <= emissivite <= 1:
        msg = f"emissivite hors de [0, 1] : {emissivite}"
        raise ValueError(msg)
    for temperature in temperatures:
        if temperature < 0:
            msg = f"temperature absolue negative : {temperature} K"
            raise ValueError(msg)


def coefficient_de_rayonnement_exact(
    emissivite: float,
    paroi: float,
    ambiante: float,
) -> float:
    """`epsilon sigma (T_p^4 - T_a^4) / (T_p - T_a)`, sans approximation.

    A l'ecart nul il se prolonge par continuite en `4 epsilon sigma T^3`,
    qui est exactement ce que la linearisation rend : les deux coincident
    donc en ce point, et c'est leur point de tangence.
    """
    _exiger_le_rayonnement(emissivite, paroi, ambiante)
    if paroi == ambiante:
        return FACTEUR_DE_LINEARISATION * emissivite * CONSTANTE_DE_STEFAN * paroi**3
    numerateur = emissivite * CONSTANTE_DE_STEFAN * (paroi**4 - ambiante**4)
    return numerateur / (paroi - ambiante)


def coefficient_de_rayonnement_linearise(
    emissivite: float,
    paroi: float,
    ambiante: float,
) -> float:
    """`4 epsilon sigma T_m^3`, la temperature moyenne au cube."""
    _exiger_le_rayonnement(emissivite, paroi, ambiante)
    moyenne = (paroi + ambiante) / 2
    return FACTEUR_DE_LINEARISATION * emissivite * CONSTANTE_DE_STEFAN * moyenne**3


def coefficient_global_plan(
    echange_interieur: float,
    epaisseur: float,
    conductivite: float,
    echange_exterieur: float,
) -> float:
    """`K = 1 / (1/h_i + e/lambda + 1/h_e)`, en W/(m^2 K)."""
    if conductivite <= 0:
        msg = f"conductivite nulle ou negative : {conductivite} W/(m K)"
        raise ValueError(msg)
    if echange_interieur <= 0 or echange_exterieur <= 0:
        msg = (
            f"coefficient d'echange nul ou negatif : {echange_interieur} "
            f"et {echange_exterieur} W/(m^2 K)"
        )
        raise ValueError(msg)
    if epaisseur < 0:
        msg = f"epaisseur negative : {epaisseur} m"
        raise ValueError(msg)
    return 1.0 / (
        1 / echange_interieur + epaisseur / conductivite + 1 / echange_exterieur
    )


def temperature_de_paroi(fluide: float, flux: float, echange: float) -> float:
    """`t_p = t_fluide - phi / h` : la temperature superficielle.

    Elle sert a verifier qu'un isolant reste sous sa temperature
    admissible -- paragraphe 1.4 du chapitre.
    """
    if echange <= 0:
        msg = f"coefficient d'echange nul ou negatif : {echange} W/(m^2 K)"
        raise ValueError(msg)
    return fluide - flux / echange
