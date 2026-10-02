"""La seance 3.1 : quatre regimes transitoires, recoupes par les noeuds.

Le manuscrit ecrit pour chaque exercice ses equations de maille, les
transforme, puis substitue. Ce fichier resout les memes circuits par la
loi des noeuds en domaine de Laplace -- conditions initiales comprises --
et compare aux **reponses imprimees du recueil**
(`Electromagnetisme/EXS/Exs_Notes_manuscrites.pdf`, section 3.2.3,
exercices 1, 2, 4 et 7).

Les circuits sont ceux des figures 3.2.10, 3.2.11, 3.2.13 et 3.2.16,
decrits dans `enonces_seance_31`.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electromagnetisme.application.enonces_seance_31 import (
    BOBINE_DU_DEUX,
    MATRICE_DU_QUATRE,
    PRIMAIRE,
    SECOND_MEMBRE_DU_QUATRE,
    SECONDAIRE,
    circuit_du_deux,
    circuit_du_quatre,
    circuit_du_sept,
    circuit_du_sept_apres_l_impulsion,
)
from polytech.electromagnetisme.application.enonces_seance_31_relais import (
    BOBINE_DU_RELAIS,
    circuit_du_relais,
    instant_d_enclenchement,
)
from polytech.electromagnetisme.application.symboles_seance_31 import (
    DUREE,
    L1,
    L2,
    L12,
    R1,
    R2,
    SEUIL_DU_RELAIS,
    TENSION,
    C,
    R,
)
from polytech.electromagnetisme.domain.symboles import TEMPS
from polytech.electromagnetisme.domain.transitoire.laplace import (
    VARIABLE,
    original,
    valeur_finale,
    valeur_initiale,
)

AMORTISSEMENT = (R1 + R2) / (C * R1 * R2)
"""L'inverse de la constante de temps de l'exercice 7."""


def test_exercice_1_le_courant_s_etablit_comme_dans_le_manuscrit() -> None:
    """`i(t) = 0,226 (1 - exp(-21,24 t))`, les deux nombres du manuscrit."""
    courant = original(circuit_du_relais().courant(BOBINE_DU_RELAIS))
    assert float(sp.limit(courant, TEMPS, sp.oo)) == pytest.approx(0.226, abs=5e-4)
    pente = sp.diff(courant, TEMPS).subs(TEMPS, 0)
    assert float(pente / sp.limit(courant, TEMPS, sp.oo)) == pytest.approx(
        21.24, abs=5e-3
    )


def test_exercice_1_le_seuil_est_franchi_entre_onze_et_douze_millisecondes() -> None:
    """Le calcul donne 11,8 ms ; le recueil et le manuscrit ecrivent 11 ms.

    L'enonce dit « quelque 0,05 A » : le seuil n'est pas donne a mieux
    que cela, et le resultat non plus. Onze millisecondes est la
    troncature, non l'arrondi -- et a cet instant le courant vaut 0,047 A.
    """
    instant = instant_d_enclenchement()
    assert float(instant) == pytest.approx(11.77e-3, abs=5e-6)
    courant = original(circuit_du_relais().courant(BOBINE_DU_RELAIS))
    a_onze = float(courant.subs(TEMPS, sp.Rational(11, 1000)))
    assert a_onze == pytest.approx(0.047, abs=5e-4)
    assert a_onze < float(SEUIL_DU_RELAIS)


def test_exercice_2_la_transformee_de_la_tension_est_celle_du_recueil() -> None:
    """La reponse imprimee du recueil, pour la figure 3.2.11.

    `U_C(p) = (R_2 + p L_2) / (p^2 L_2 C R_1 + p (C R_1 R_2 + L_2) + R_1 + R_2)`
    fois `E / p`.
    """
    obtenue = circuit_du_deux().potentiels()["n"]
    denominateur = VARIABLE**2 * L2 * C * R1 + VARIABLE * (C * R1 * R2 + L2) + R1 + R2
    attendue = (R2 + VARIABLE * L2) / denominateur * TENSION / VARIABLE
    assert sp.simplify(obtenue - attendue) == 0


def test_exercice_2_la_tension_de_regime_est_celle_du_diviseur() -> None:
    """`U_C = R_2 E / (R_1 + R_2)`, par le theoreme de la valeur finale.

    Le manuscrit l'obtient en redessinant le circuit en regime ; ici elle
    sort de la transformee elle-meme. Deux chemins, une valeur.
    """
    transformee = circuit_du_deux().potentiels()["n"]
    assert sp.simplify(valeur_finale(transformee) - R2 * TENSION / (R1 + R2)) == 0


def test_exercice_2_la_bobine_garde_son_courant_nul_a_l_instant_initial() -> None:
    """`i_2(0+) = 0` : le courant d'une inductance ne saute pas."""
    courant = circuit_du_deux().courant(BOBINE_DU_DEUX)
    assert valeur_initiale(courant) == 0


def test_exercice_4_le_circuit_donne_le_systeme_matriciel_du_recueil() -> None:
    """Les trois transformees satisfont les trois lignes de la matrice.

    L'oracle est la reponse imprimee : la matrice, le vecteur des
    inconnues et le second membre. On y reporte ce que la loi des noeuds
    donne, et chaque ligne doit s'annuler.
    """
    reseau = circuit_du_quatre()
    inconnues = sp.Matrix(
        [
            reseau.courant(PRIMAIRE),
            reseau.courant(SECONDAIRE),
            reseau.potentiels()["k"],
        ],
    )
    residu = MATRICE_DU_QUATRE * inconnues - SECOND_MEMBRE_DU_QUATRE
    assert sp.simplify(residu) == sp.zeros(3, 1)


def test_exercice_4_la_matrice_est_celle_que_le_recueil_imprime() -> None:
    """Ses neuf coefficients et son second membre, relus sur la page 75.

    Sans ce test, la matrice ne serait comparee qu'au circuit : une faute
    commune aux deux passerait. Elle est donc aussi ecrite ici, a la
    main, depuis la page imprimee.
    """
    p = VARIABLE
    imprimee = sp.Matrix(
        [[R + R1 + p * L1, p * L12, 0], [p * L12, R2 + p * L2, -1], [0, 1, p * C]],
    )
    second = sp.Matrix([TENSION / p + L1 * TENSION / R, L12 * TENSION / R, 0])
    assert imprimee == MATRICE_DU_QUATRE
    assert second == SECOND_MEMBRE_DU_QUATRE


def test_exercice_4_en_regime_le_secondaire_est_au_repos() -> None:
    """Reponse du recueil : `i_2 = 0 A` et `u_C = 0 V` en regime.

    Le manuscrit pose le systeme et s'arrete ; l'enonce demandait aussi
    ces deux valeurs.
    """
    reseau = circuit_du_quatre()
    assert sp.simplify(valeur_finale(reseau.courant(SECONDAIRE))) == 0
    assert sp.simplify(valeur_finale(reseau.potentiels()["k"])) == 0
    assert (
        sp.simplify(valeur_finale(reseau.courant(PRIMAIRE)) - TENSION / (R + R1)) == 0
    )


def test_exercice_7_la_reponse_pendant_l_impulsion_est_celle_du_recueil() -> None:
    """`u = E R_2/(R_1+R_2) + E R_1/(R_1+R_2) exp(-a t)` pour `0 < t < T`."""
    obtenue = original(circuit_du_sept().potentiels()["n"])
    somme = R1 + R2
    attendue = TENSION * R2 / somme + TENSION * R1 / somme * sp.exp(
        -AMORTISSEMENT * TEMPS
    )
    assert sp.simplify(obtenue - attendue) == 0


def test_exercice_7_la_reponse_apres_l_impulsion_est_celle_du_recueil() -> None:
    """`u = (q(T)/C) exp(-a (t - T))`, avec `q(T)` donne par le recueil.

    Obtenue ici **sans superposition** : on repart de l'instant `T`, la
    source eteinte, avec pour seule memoire la tension du condensateur a
    cet instant. Le temps est compte depuis `T`.
    """
    obtenue = original(circuit_du_sept_apres_l_impulsion().potentiels()["n"])
    charge = TENSION * C * R1 / (R1 + R2) * (sp.exp(-AMORTISSEMENT * DUREE) - 1)
    attendue = charge / C * sp.exp(-AMORTISSEMENT * TEMPS)
    assert sp.simplify(obtenue - attendue) == 0


def test_exercice_7_la_superposition_du_manuscrit_donne_la_meme_reponse() -> None:
    """Echelon montant en zero moins echelon montant en `T` : la route du manuscrit.

    Comparee a la route precedente, qui n'emploie aucune superposition :
    leur accord verifie le signe et le retard.
    """
    echelon = original(circuit_du_sept().potentiels()["n"])
    apres = echelon.subs(TEMPS, TEMPS + DUREE) - echelon
    direct = original(circuit_du_sept_apres_l_impulsion().potentiels()["n"])
    assert sp.simplify(apres - direct) == 0


def test_exercice_7_la_tension_saute_de_la_hauteur_de_l_impulsion() -> None:
    """En `t = T` la sortie chute exactement de `E` : le condensateur ne saute pas.

    Sa tension etant continue, tout le saut de la source se retrouve aux
    bornes de `R_2`. C'est pourquoi la sortie devient **negative** juste
    apres l'impulsion, ce que le manuscrit note sur sa figure.
    """
    echelon = original(circuit_du_sept().potentiels()["n"])
    avant = echelon.subs(TEMPS, DUREE)
    apres = original(circuit_du_sept_apres_l_impulsion().potentiels()["n"]).subs(
        TEMPS, 0
    )
    assert sp.simplify(avant - apres - TENSION) == 0
