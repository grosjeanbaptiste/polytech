"""L'epaisseur de la lame d'air entre un cylindre pose au milieu d'une caisse carree."""

from __future__ import annotations


def epaisseur_de_la_lame(cote_interieur: float, rayon: float) -> float:
    """`(cote - 2 rayon) / 2`, en metres ; leve si la casserole touche la paroi."""
    lame = (cote_interieur - 2 * rayon) / 2
    if lame <= 0:
        msg = (
            f"aucune lame : cote {cote_interieur} m, rayon {rayon} m "
            "(la casserole touche la paroi ou ne rentre pas)"
        )
        raise ValueError(msg)
    return lame
