"""L'exercice 12 : un disque tournant dans un champ magnetique.

C'est le **disque de Faraday**, premier generateur electrique de
l'histoire. Un disque conducteur de diametre `D` tourne a la vitesse
angulaire `omega` dans un champ `B` perpendiculaire ; on recueille une
tension entre son centre et son bord.

Le corrige manuscrit mene le calcul juste et encadre
`I = B omega D^2 / (8 R)`. Cinq choses lui manquent, et ce fichier les
etablit.

**Deux routes independantes donnent la meme tension.** L'integrale de la
force de Laplace, que le manuscrit fait ; et le **flux balaye par
unite de temps**, qui ne passe par aucune integrale de chemin. Leur
accord verifie le facteur huit.

**Le disque equivaut a une tige mobile a la vitesse du rayon moyen.** Et
cela tient **parce que** l'integrande est lineaire en `r` : la moyenne
d'une fonction lineaire est sa valeur au milieu. Un integrande non
lineaire romprait l'equivalence -- un test le montre, sans quoi
l'equivalence passerait pour une regle generale.

**Le produit vectoriel est radial**, et c'est ce qui permet au produit
scalaire avec `dr u_r` de ne rien perdre. Verifie par un vrai produit
vectoriel plutot que par l'identite citee.

**La tension croit comme le carre du diametre.** Doubler le disque
quadruple la tension -- ce qui n'est pas evident, le rayon intervenant
deja dans la vitesse.

**Le paradoxe du disque.** Le flux a travers un contour ferme ne change
pas : `B` est uniforme et la geometrie fixe. On ne peut donc **pas**
appliquer `e = -dPhi/dt` naivement, et c'est pourtant la loi qu'on
enseigne d'abord. Le flux balaye est la bonne lecture, et le dernier
test fixe la difference.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.magnetostatique.disque_tournant import (
    courant_debite,
    produit_vectoriel_cylindrique,
    tension_induite,
    tension_par_la_tige_equivalente,
    tension_par_le_flux_balaye,
    vitesse_au_rayon_moyen,
    vitesse_lineaire,
)

INDUCTION = sp.Symbol("B", positive=True)
PULSATION = sp.Symbol("omega", positive=True)
DIAMETRE = sp.Symbol("D", positive=True)
RESISTANCE = sp.Symbol("R", positive=True)
RAYON = sp.Symbol("r", positive=True)


def test_la_vitesse_croit_lineairement_avec_le_rayon() -> None:
    """`v = omega r` : c'est la seule chose qui depend de `r` dans le calcul.

    Et c'est elle qui rend l'integrande lineaire, donc l'equivalence a la
    tige du rayon moyen possible.
    """
    assert sp.simplify(vitesse_lineaire(PULSATION, RAYON) - PULSATION * RAYON) == 0


def test_le_produit_vectoriel_de_la_vitesse_et_du_champ_est_radial() -> None:
    """`u_phi ^ u_z = u_r`, verifie par un vrai produit vectoriel.

    Le manuscrit cite l'identite ; on la calcule. C'est ce caractere
    radial qui permet au produit scalaire avec `dr u_r` de ne rien
    perdre -- une composante azimutale ou axiale ne contribuerait pas au
    chemin du centre vers le bord.
    """
    azimutal = sp.Matrix([0, 1, 0])
    axial = sp.Matrix([0, 0, 1])
    radial = sp.Matrix([1, 0, 0])
    obtenu = produit_vectoriel_cylindrique(azimutal, axial)
    assert sp.simplify(obtenu - radial) == sp.zeros(3, 1)


def test_la_tension_induite_est_celle_du_manuscrit() -> None:
    """`e = B omega D^2 / 8` : l'oracle est le calcul encadre du manuscrit."""
    attendue = INDUCTION * PULSATION * DIAMETRE**2 / 8
    obtenue = tension_induite(INDUCTION, PULSATION, DIAMETRE)
    assert sp.simplify(obtenue - attendue) == 0


def test_le_courant_debite_est_celui_qu_encadre_le_manuscrit() -> None:
    """`I = B omega D^2 / (8 R)`, le resultat encadre."""
    attendu = INDUCTION * PULSATION * DIAMETRE**2 / (8 * RESISTANCE)
    obtenu = courant_debite(INDUCTION, PULSATION, DIAMETRE, RESISTANCE)
    assert sp.simplify(obtenu - attendu) == 0


def test_le_flux_balaye_donne_la_meme_tension() -> None:
    """Route independante : aucune integrale de chemin n'y intervient.

    Le rayon balaie une aire par unite de temps, et la tension est le
    produit de `B` par cette aire balayee. C'est la lecture correcte de
    la loi de Faraday pour cette configuration, et elle n'emploie ni la
    force de Laplace ni le produit vectoriel.
    """
    par_laplace = tension_induite(INDUCTION, PULSATION, DIAMETRE)
    par_le_flux = tension_par_le_flux_balaye(INDUCTION, PULSATION, DIAMETRE)
    assert sp.simplify(par_laplace - par_le_flux) == 0


def test_la_tige_au_rayon_moyen_donne_la_meme_tension() -> None:
    """Le disque equivaut a une tige de longueur `D/2` a la vitesse moyenne.

    Troisieme route, et la plus parlante : la tension est celle qu'une
    tige rectiligne de longueur `D/2` donnerait si elle se deplacait tout
    entiere a la vitesse du point situe a mi-rayon.
    """
    par_laplace = tension_induite(INDUCTION, PULSATION, DIAMETRE)
    par_la_tige = tension_par_la_tige_equivalente(
        INDUCTION,
        PULSATION,
        DIAMETRE,
    )
    assert sp.simplify(par_laplace - par_la_tige) == 0


def test_la_vitesse_du_rayon_moyen_est_la_moitie_de_celle_du_bord() -> None:
    """`omega D / 4` : la vitesse a mi-rayon, moitie de celle du bord.

    Elle vaut aussi la **moyenne** des vitesses entre le centre et le
    bord, et c'est cette coincidence -- propre aux fonctions lineaires --
    qui fonde l'equivalence a la tige.
    """
    au_bord = vitesse_lineaire(PULSATION, DIAMETRE / 2)
    moyenne = vitesse_au_rayon_moyen(PULSATION, DIAMETRE)
    assert sp.simplify(moyenne - au_bord / 2) == 0


def test_l_equivalence_a_la_tige_echoue_si_l_integrande_n_est_pas_lineaire() -> None:
    """Le contraste, sans lequel l'equivalence passerait pour generale.

    Elle ne tient que parce que la moyenne d'une fonction **lineaire**
    est sa valeur au milieu. Avec un integrande en `r^2` -- ce qu'un
    champ non uniforme produirait -- la valeur au milieu n'est plus la
    moyenne, et l'equivalence tombe.
    """
    borne = DIAMETRE / 2
    vraie = sp.integrate(RAYON**2, (RAYON, 0, borne))
    par_le_milieu = (borne / 2) ** 2 * borne
    assert sp.simplify(vraie - par_le_milieu) != 0


def test_la_tension_croit_comme_le_carre_du_diametre() -> None:
    """Doubler le disque quadruple la tension.

    Ce n'est pas evident : le rayon intervient **deux** fois -- dans la
    longueur du chemin et dans la vitesse -- et c'est leur produit qui
    donne le carre.
    """
    simple = tension_induite(INDUCTION, PULSATION, DIAMETRE)
    double = tension_induite(INDUCTION, PULSATION, 2 * DIAMETRE)
    assert sp.simplify(double / simple - 4) == 0


def test_le_flux_a_travers_un_contour_fixe_ne_change_pas() -> None:
    """Le paradoxe du disque : aucun flux ne varie, et pourtant il y a tension.

    `B` est uniforme et le contour geometrique est fixe : le flux qui le
    traverse est constant, donc sa derivee est nulle. Appliquer
    `e = -dPhi/dt` a ce contour donnerait **zero**, ce qui est faux.

    La raison est que dans un conducteur en rotation le « circuit » n'est
    pas defini : les porteurs changent de place. C'est le flux **balaye**
    qui compte, non le flux enlace -- et c'est pourquoi le test
    precedent, qui emploie l'aire balayee, donne la bonne reponse la ou
    celui-ci donne zero.
    """
    temps = sp.Symbol("t", positive=True)
    aire_du_contour = sp.pi * (DIAMETRE / 2) ** 2
    flux = INDUCTION * aire_du_contour
    assert sp.diff(flux, temps) == 0
    assert sp.simplify(tension_induite(INDUCTION, PULSATION, DIAMETRE)) != 0
