"""La borne de stabilite d'Euler explicite, verifiee et non citee.

Le syllabus annonce « on peut montrer qu'il faut » puis une formule que
la conversion n'a pas rendue -- c'etait une image. Affirmer `Fo <= 1/(2d)`
sur cette foi serait exactement ce que ce depot refuse : un resultat qui
tombe du ciel.

Il est donc verifie par ce qu'il predit. Un schema explicite au-dela de
la borne doit **diverger**, et en dessous **converger** vers l'etat
stationnaire -- ici une barre dont les deux bouts sont tenus a zero et
qui part d'une temperature uniforme. Les deux comportements sont
observes, et c'est ce qui etablit la borne.

Le mode qui diverge le premier est le plus oscillant du maillage -- une
dent de scie, un noeud chaud entre deux froids. C'est pourquoi l'etat
initial du test de divergence en porte une : partir d'un champ lisse
retarderait la divergence sans l'empecher, et rendrait le test lent au
lieu de le rendre faux.
"""

from __future__ import annotations

from typing import Final

import pytest

from polytech.thermique.domain.differences_finies import (
    nombre_de_fourier_de_maille,
)
from polytech.thermique.domain.stabilite_numerique import (
    fourier_critique,
    fourier_critique_du_maillage,
)

NOEUDS: Final = 11
"""Une barre discretisee, bords tenus a zero."""

PAS_DE_TEMPS: Final = 400
"""Assez pour qu'une instabilite se voie, et qu'une stabilite s'etablisse."""


def _avancer(temperatures: list[float], fourier: float, pas: int) -> list[float]:
    """Le schema explicite a une dimension, `pas` fois.

    `t_i^+ = t_i + Fo (t_{i-1} - 2 t_i + t_{i+1})`, les deux bords tenus
    a zero. C'est exactement la forme canonique du syllabus, avec la
    derivee temporelle evaluee a l'instant connu -- d'ou « explicite ».
    """
    champ = list(temperatures)
    for _ in range(pas):
        suivant = list(champ)
        for i in range(1, len(champ) - 1):
            suivant[i] = champ[i] + fourier * (
                champ[i - 1] - 2.0 * champ[i] + champ[i + 1]
            )
        champ = suivant
    return champ


def test_sous_la_borne_le_schema_converge_vers_l_etat_stationnaire() -> None:
    """Bords a zero, depart uniforme : la barre doit se vider de sa chaleur."""
    fourier = fourier_critique(1) * 0.9
    final = _avancer([0.0, *[100.0] * (NOEUDS - 2), 0.0], fourier, PAS_DE_TEMPS)
    assert max(abs(t) for t in final) < 1.0


def test_au_dela_de_la_borne_le_schema_diverge() -> None:
    """Le meme probleme, un pas de temps a peine plus grand : il explose.

    C'est la verification qui donne son poids a la borne. L'ecart entre
    les deux pas de temps est de dix pour cent, et il separe une barre
    qui refroidit d'une barre dont la temperature depasse mille fois sa
    valeur initiale.
    """
    fourier = fourier_critique(1) * 1.1
    depart = [0.0, *[100.0 * (-1) ** i for i in range(NOEUDS - 2)], 0.0]
    final = _avancer(depart, fourier, PAS_DE_TEMPS)
    assert max(abs(t) for t in final) > 1e5


def test_la_borne_exacte_du_maillage_est_la_ou_le_comportement_change() -> None:
    """Et c'est la borne du **maillage**, non celle du cours, qui tranche.

    Trouvaille de cette verification, et elle est instructive : a 1,01
    fois la borne du cours, le schema reste stable sur onze noeuds. La
    borne `1/(2d)` suppose un mode de longueur d'onde deux mailles, que
    le maillage fini ne porte pas tout a fait -- son mode le plus
    oscillant est un peu moins raide, donc un peu plus tolerant.

    La borne qui tranche vraiment se deduit de ce mode, et c'est elle que
    ce test encadre a deux pour cent : en dessous la barre refroidit, au
    dessus elle explose.
    """
    depart = [0.0, *[100.0 * (-1) ** i for i in range(NOEUDS - 2)], 0.0]
    exacte = fourier_critique_du_maillage(NOEUDS)
    dessous = _avancer(depart, exacte * 0.98, PAS_DE_TEMPS)
    dessus = _avancer(depart, exacte * 1.02, PAS_DE_TEMPS)
    assert max(abs(t) for t in dessous) < 1.0
    assert max(abs(t) for t in dessus) > 1e6


def test_la_borne_du_maillage_tend_vers_celle_du_cours() -> None:
    """`1/(2d)` est l'asymptote, et c'est ce qui la justifie.

    Six noeuds la depassent de onze pour cent, deux cents de un
    dix-milliemes. La borne du cours est donc **suffisante** a toute
    finesse, ce qui est exactement ce qu'un dimensionnement demande --
    et c'est pourquoi `est_stable_en_explicite` s'y tient.
    """
    grossier = fourier_critique_du_maillage(6) / fourier_critique(1)
    fin = fourier_critique_du_maillage(201) / fourier_critique(1)
    assert grossier == pytest.approx(1.106, abs=0.001)
    assert fin == pytest.approx(1.0001, abs=0.0001)
    assert fourier_critique_du_maillage(201) > fourier_critique(1)


def test_un_maillage_de_moins_de_trois_noeuds_est_refuse() -> None:
    """Deux bords et rien entre eux : il n'y a aucun noeud a avancer."""
    with pytest.raises(ValueError, match="moins de trois noeuds"):
        fourier_critique_du_maillage(2)


def test_un_cas_physique_retrouve_la_borne_par_ses_grandeurs() -> None:
    """Le chemin que suit un dimensionnement reel, et il doit concorder.

    On part des grandeurs -- diffusivite de l'acier, maillage de deux
    centimetres -- on calcule le pas de temps critique, puis le nombre de
    Fourier de maille qu'il donne. Celui-ci doit retomber sur la borne :
    les deux expressions sont inverses l'une de l'autre, et une faute de
    frappe sur l'une se verrait ici.
    """
    diffusivite, maille = 1.2e-5, 0.02
    critique = fourier_critique(1) * maille**2 / diffusivite
    retrouve = nombre_de_fourier_de_maille(
        diffusivite=diffusivite,
        pas_de_temps=critique,
        pas_d_espace=maille,
    )
    assert retrouve == pytest.approx(fourier_critique(1))
