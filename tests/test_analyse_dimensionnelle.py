"""Le theoreme de Vaschy-Buckingham, et la methode des variables repetees.

Le depot porte un syllabus entier sur ce theoreme -- l'extrait du
chapitre 7 de Cengel -- et rien ne le portait dans la bibliotheque,
parce que la mesure de couverture ne regardait que les planches.

Le document est explicite la ou il compte : les six etapes sont
enumerees, le theoreme est ecrit `k = n - j`, et la regle de repli est
donnee en clair -- « if the analysis does not work out, verify that you
have included enough parameters in step 1. Otherwise, go back and reduce
j by one and try again ».

La verification la plus forte n'est pas arithmetique. Le theoreme
**predit** combien de groupes sans dimension un probleme porte, et la
bibliotheque de ce depot en emploie deja : la trainee est correlee par
le seul Reynolds, la convection forcee par Reynolds et Prandtl. Les deux
comptes doivent concorder, et c'est ce que les deux derniers tests
exigent. Un theoreme mal transcrit s'y verrait.
"""

from __future__ import annotations

import pytest

from polytech.fluides.domain.analyse_dimensionnelle import (
    ETAPES_DE_LA_METHODE,
    nombre_de_groupes,
    reduction_apres_echec,
)


def test_la_methode_compte_six_etapes_dans_l_ordre() -> None:
    """Et l'ordre porte le sens : on ne choisit les repetees qu'en etape 4.

    Les choisir avant d'avoir compte les dimensions primaires serait les
    choisir sans savoir combien il en faut -- `j` est precisement ce
    nombre.
    """
    assert len(ETAPES_DE_LA_METHODE) == 6
    assert ETAPES_DE_LA_METHODE[0].startswith("lister les parametres")
    assert "repetees" in ETAPES_DE_LA_METHODE[3]


def test_le_theoreme_retranche_la_reduction_du_nombre_de_parametres() -> None:
    """`k = n - j` : l'enonce meme, equation 7-14 du document."""
    assert nombre_de_groupes(parametres=5, reduction=3) == 2
    assert nombre_de_groupes(parametres=7, reduction=4) == 3


def test_un_probleme_sans_dimension_garde_tous_ses_parametres() -> None:
    """A reduction nulle, aucun groupe ne se forme : `k = n`.

    Cas limite, et il n'est pas d'ecole : un probleme dont tous les
    parametres sont deja sans dimension n'a rien a adimensionner.
    """
    assert nombre_de_groupes(parametres=4, reduction=0) == 4


def test_la_reduction_decroit_d_une_unite_apres_un_echec() -> None:
    """« go back and reduce j by one and try again », dit le document.

    C'est la regle qui rend la methode utilisable : `j` est devine, non
    calcule, et le devin peut se tromper. Sans cette regle, un echec
    d'algebre bloquerait l'analyse au lieu de la relancer.
    """
    assert reduction_apres_echec(3) == 2
    assert reduction_apres_echec(1) == 0


def test_une_reduction_deja_nulle_ne_peut_plus_decroitre() -> None:
    """A zero il n'y a plus rien a retrancher : l'echec vient d'ailleurs.

    Le document le dit avant la regle de repli : « verify that you have
    included enough parameters in step 1 ». La garde renvoie donc a
    l'etape 1 plutot que de rendre une reduction negative.
    """
    with pytest.raises(ValueError, match="reduction deja nulle"):
        reduction_apres_echec(0)


def test_une_reduction_superieure_au_nombre_de_parametres_est_refusee() -> None:
    """Elle donnerait un nombre de groupes negatif, qui ne veut rien dire."""
    with pytest.raises(ValueError, match="reduction superieure au nombre"):
        nombre_de_groupes(parametres=3, reduction=4)


def test_la_trainee_ne_depend_que_d_un_seul_groupe_sans_dimension() -> None:
    """Et c'est pourquoi `C_D` se correle par le seul Reynolds.

    Cinq parametres -- force, vitesse, longueur, masse volumique,
    viscosite -- et trois dimensions primaires : masse, longueur, temps.
    Le theoreme annonce deux groupes, donc une relation entre deux
    nombres : `C_D = f(Re)`. C'est exactement la forme que
    `polytech.fluides.domain.aerodynamique` emploie, et la concordance
    est la verification.
    """
    assert nombre_de_groupes(parametres=5, reduction=3) == 2


def test_la_convection_forcee_porte_trois_groupes_sans_dimension() -> None:
    """Nusselt, Reynolds, Prandtl -- ni deux, ni quatre.

    Sept parametres -- coefficient d'echange, longueur, vitesse, masse
    volumique, viscosite, chaleur massique, conductivite -- et quatre
    dimensions primaires, la temperature s'ajoutant aux trois de la
    mecanique. Le theoreme annonce trois groupes, et c'est le nombre
    exact que les correlations de `polytech.thermique` emploient :
    `Nu = f(Re, Pr)`.

    La temperature est ce qui fait la difference entre ce probleme et
    celui de la trainee, et l'oublier donnerait quatre groupes -- une
    correlation de plus a chercher, qui n'existe pas.
    """
    assert nombre_de_groupes(parametres=7, reduction=4) == 3
    assert nombre_de_groupes(parametres=7, reduction=3) == 4
