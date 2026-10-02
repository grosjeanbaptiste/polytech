"""L'exercice 8 : une sphere dont la charge varie en racine du rayon.

Le corrige manuscrit est, pour une fois, complet : il mene l'exercice par
**les deux methodes** -- Poisson puis Gauss -- et constate qu'elles
s'accordent. Ce fichier refait les deux, par deux machineries
genuinement differentes : une resolution d'equation differentielle d'un
cote, une integrale triple en spheriques de l'autre.

Trois choses restent a dire, et ce sont elles que ce fichier etablit.

**Le passage a `C = 0` du manuscrit est un raisonnement inverse.** Il
ecrit « en `r = 0`, `E_r = 0`, donc `C = 0` » -- mais on ne peut pas
evaluer `r^2 E_r = C` en zero pour en conclure que `C` est nul : il
faudrait deja savoir que `E_r` y est borne. Le vrai argument est
l'inverse : le terme ecarte est `C/r^2`, qui **diverge** a l'origine, et
rien ne peut y produire un champ infini puisqu'il n'y a pas de charge
ponctuelle au centre. La resolution symbolique exhibe ce terme, ce qui
rend l'argument visible au lieu de le supposer.

**Le champ exterieur est exactement celui d'une charge ponctuelle.** Le
manuscrit l'obtient sans le nommer. C'est pourtant le fait le plus utile
de l'exercice : vue de l'exterieur, toute distribution a symetrie
spherique se reduit a sa charge totale placee au centre.

**Le champ est maximal sur la surface.** Il croit a l'interieur, decroit
a l'exterieur : le maximum est en `r = R` exactement, ce qui se verifie.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.electrostatique.sphere_radiale import (
    champ_exterieur,
    champ_par_gauss,
    champ_par_poisson,
    charge_contenue,
    charge_volumique,
    divergence_spherique,
    terme_divergent_ecarte,
)

CONSTANTE = sp.Symbol("A", positive=True)
RAYON = sp.Symbol("r", positive=True)
RAYON_DE_LA_SPHERE = sp.Symbol("R", positive=True)
PERMITTIVITE = sp.Symbol("varepsilon_0", positive=True)


def test_la_charge_volumique_est_la_loi_de_l_enonce() -> None:
    """`eta = A sqrt(r)` : plus de charge vers l'exterieur qu'au centre.

    La loi n'est pas uniforme, et c'est tout l'interet de l'exercice :
    Gauss reste applicable -- la symetrie spherique suffit -- mais la
    charge contenue n'est plus proportionnelle au volume.
    """
    assert (
        sp.simplify(
            charge_volumique(CONSTANTE, RAYON) - CONSTANTE * sp.sqrt(RAYON),
        )
        == 0
    )


def test_les_deux_methodes_donnent_le_meme_champ_interieur() -> None:
    """Poisson et Gauss s'accordent, et le constat du manuscrit tient.

    Les deux machineries n'ont rien en commun : l'une resout une equation
    differentielle du premier ordre, l'autre integre une densite sur un
    volume en coordonnees spheriques. Qu'elles aboutissent a la meme
    expression est donc une verification, non une tautologie.
    """
    assert (
        sp.simplify(
            champ_par_poisson(CONSTANTE, RAYON) - champ_par_gauss(CONSTANTE, RAYON),
        )
        == 0
    )


def test_le_champ_interieur_est_celui_qu_encadre_le_manuscrit() -> None:
    """`E_i = 2A r^{3/2} / (7 eps_0)` : l'oracle est le resultat encadre.

    Ecrit ici a la main d'apres le scan, donc externe a la bibliotheque.
    """
    encadre = 2 * CONSTANTE * RAYON ** sp.Rational(3, 2) / (7 * PERMITTIVITE)
    assert sp.simplify(champ_par_poisson(CONSTANTE, RAYON) - encadre) == 0


def test_la_divergence_du_champ_trouve_redonne_la_densite() -> None:
    """Le controle en boucle fermee, et le plus severe de l'exercice.

    On repart du champ obtenu, on en calcule la divergence spherique, et
    l'on doit retomber sur `eta / eps_0` -- l'equation de Poisson d'ou
    l'on etait parti. Une erreur de coefficient ou d'exposant commise en
    integrant ne survivrait pas a la derivation inverse.
    """
    champ = champ_par_poisson(CONSTANTE, RAYON)
    attendu = charge_volumique(CONSTANTE, RAYON) / PERMITTIVITE
    assert sp.simplify(divergence_spherique(champ, RAYON) - attendu) == 0


CONSTANTE_D_INTEGRATION = sp.Symbol("C", positive=True)
"""Prise positive pour que la limite soit `+oo` et non `oo . sign(C)`.

Le signe ne change rien a l'argument : c'est le **module** qui diverge,
et une constante negative donnerait `-oo`, tout aussi impossible.
"""


def test_le_terme_ecarte_diverge_a_l_origine() -> None:
    """`C/r^2` : voila pourquoi on l'ecarte, et le manuscrit ne le dit pas.

    Il ecrit « en `r = 0`, `E_r = 0`, donc `C = 0` », ce qui suppose ce
    qu'il faut montrer. Le vrai argument est que ce terme **diverge** en
    zero, et que rien ne peut y produire un champ infini faute de charge
    ponctuelle au centre.
    """
    terme = terme_divergent_ecarte(CONSTANTE_D_INTEGRATION, RAYON)
    assert sp.limit(terme, RAYON, 0, "+") is sp.oo


def test_le_terme_ecarte_satisfait_pourtant_l_equation_homogene() -> None:
    """Il n'est pas faux mathematiquement -- seulement non physique.

    Sa divergence spherique est nulle : il est donc solution de
    l'equation sans source, et c'est exactement pourquoi il apparait dans
    la solution generale. Le rejeter demande un argument **physique**, et
    c'est ce qui rend la lacune du manuscrit interessante plutot que
    anecdotique.
    """
    assert (
        sp.simplify(
            divergence_spherique(
                terme_divergent_ecarte(CONSTANTE_D_INTEGRATION, RAYON),
                RAYON,
            )
        )
        == 0
    )


def test_la_charge_contenue_croit_comme_la_puissance_sept_demis() -> None:
    """`Q(r) = 8 pi A r^{7/2} / 7`, et l'exposant n'est pas trois.

    Pour une densite uniforme la charge croitrait comme `r^3`. Ici la
    densite croit elle-meme en `sqrt(r)`, ce qui ajoute un demi a
    l'exposant : `3 + 1/2 = 7/2`. Le verifier controle l'integration.
    """
    attendue = 8 * sp.pi * CONSTANTE * RAYON ** sp.Rational(7, 2) / 7
    assert sp.simplify(charge_contenue(CONSTANTE, RAYON) - attendue) == 0


def test_le_champ_exterieur_est_celui_d_une_charge_ponctuelle_au_centre() -> None:
    """Le fait le plus utile, et le manuscrit ne le nomme pas.

    Vue de l'exterieur, toute distribution a symetrie spherique se reduit
    a sa charge totale placee au centre. On le verifie en comparant le
    champ exterieur a `Q_tot / (4 pi eps_0 r^2)`, ou `Q_tot` est
    l'integrale de la densite sur la sphere entiere.
    """
    totale = charge_contenue(CONSTANTE, RAYON_DE_LA_SPHERE)
    ponctuelle = totale / (4 * sp.pi * PERMITTIVITE * RAYON**2)
    dehors = champ_exterieur(CONSTANTE, RAYON, RAYON_DE_LA_SPHERE)
    assert sp.simplify(dehors - ponctuelle) == 0


def test_le_champ_est_continu_a_la_traversee_de_la_surface() -> None:
    """Les deux expressions coincident en `r = R`, et il faut le verifier.

    Le manuscrit **emploie** cette continuite pour fixer sa constante,
    mais ne la verifie jamais sur le couple final. C'est pourtant la que
    les deux moities du resultat se recollent.

    La continuite n'est d'ailleurs pas automatique : elle tient parce
    qu'il n'y a **pas** de charge surfacique en `r = R`. Une couche de
    charge y creerait un saut.
    """
    dedans = champ_par_poisson(CONSTANTE, RAYON_DE_LA_SPHERE)
    dehors = champ_exterieur(
        CONSTANTE,
        RAYON_DE_LA_SPHERE,
        RAYON_DE_LA_SPHERE,
    )
    assert sp.simplify(dedans - dehors) == 0


def test_le_champ_est_maximal_exactement_sur_la_surface() -> None:
    """Il croit dedans, decroit dehors : le maximum est en `r = R`.

    Verifie par les signes des deux derivees plutot qu'en cherchant un
    extremum : la fonction n'est pas derivable en `R` -- sa derivee y
    saute -- et un `solve` sur la derivee ne trouverait rien.
    """
    dedans = sp.diff(champ_par_poisson(CONSTANTE, RAYON), RAYON)
    dehors = sp.diff(
        champ_exterieur(CONSTANTE, RAYON, RAYON_DE_LA_SPHERE),
        RAYON,
    )
    assert sp.simplify(dedans).is_positive
    assert sp.simplify(dehors).is_negative
