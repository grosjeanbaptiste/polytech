"""L'exercice 6 du chapitre IV : la force sur une armature de condensateur.

Le manuscrit emploie une **difference finie** et non une derivee :

    |F| = |(dW/dx)_U| ~= |(Delta W / Delta x)_U|

Le signe `~=` est dans l'enonce, et il demande d'etre explique. Pour une
capacite **affine** en `x`, la secante est la tangente et le quotient est
exact ; ce fichier le montre, ainsi que le contre-exemple quadratique.

Ce docstring affirmait que la capacite d'un condensateur plan est affine
en l'**ecartement** : c'est faux, elle y est en `1/e`. Elle n'est affine
qu'en le recouvrement. Ce que le quotient vaut sans cette hypothese --
la force moyenne -- est etabli dans
`tests/test_force_moyenne_sur_un_deplacement.py`.

Le second point du chapitre est le **signe**. A tension constante, la
source fournit du travail en plus de l'energie stockee, et la force vaut
`+dW/dx`. A charge constante, elle vaut `-dW/dx`. Les deux conventions ne
donnent pas la meme formule, et l'enonce precise bien « a tension `U`
constante ».
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.energie_et_forces import (
    DEPLACEMENT,
    energie_d_un_condensateur,
    force_a_courant_constant,
    force_sur_une_armature,
)

# Donnees de la transcription.
TENSION = 10e3
VARIATION_DE_CAPACITE = 4e-12
DEPLACEMENT_IMPOSE = 1e-3


def test_la_force_vaut_deux_dixiemes_de_newton() -> None:
    """`F = (1/2) U^2 |Delta C| / Delta x = 0,2` N."""
    force = force_sur_une_armature(
        TENSION,
        VARIATION_DE_CAPACITE,
        DEPLACEMENT_IMPOSE,
    )
    assert force == pytest.approx(0.2)


def test_la_force_croit_comme_le_carre_de_la_tension() -> None:
    """Doubler `U` quadruple la force : l'energie est en `U^2`.

    C'est ce qui rend ces forces negligeables a basse tension et
    redoutables a dix kilovolts -- l'ordre de grandeur de l'enonce.
    """
    simple = force_sur_une_armature(TENSION, VARIATION_DE_CAPACITE, DEPLACEMENT_IMPOSE)
    double = force_sur_une_armature(
        2 * TENSION,
        VARIATION_DE_CAPACITE,
        DEPLACEMENT_IMPOSE,
    )
    assert double == pytest.approx(4 * simple)


def test_la_difference_finie_coincide_avec_la_derivee_sur_une_capacite_affine() -> None:
    """Le cas ou le `~=` de l'enonce devient une egalite.

    Pour une capacite affine en `x`, la secante et la tangente
    coincident : la difference finie n'approche rien, elle **est** la
    derivee. Le test construit une telle capacite, derive
    symboliquement, et compare au quotient de l'enonce.
    """
    pente = VARIATION_DE_CAPACITE / DEPLACEMENT_IMPOSE
    capacite = 1e-9 + pente * DEPLACEMENT
    energie = energie_d_un_condensateur(capacite, sp.Float(TENSION))
    par_la_derivee = float(force_a_courant_constant(energie))
    par_la_difference = force_sur_une_armature(
        TENSION,
        VARIATION_DE_CAPACITE,
        DEPLACEMENT_IMPOSE,
    )
    assert abs(par_la_derivee) == pytest.approx(par_la_difference)


def test_la_difference_finie_s_ecarte_sur_une_capacite_non_affine() -> None:
    """Et la coincidence cesse des que la capacite n'est plus affine.

    L'egalite exacte tient donc a la linearite ; hors d'elle, seule la
    petitesse du deplacement justifie le `~=`. Sur une capacite quadratique,
    la secante et la tangente diffferent -- la tangente est prise au
    debut de l'intervalle, la secante le moyenne.
    """
    quadratique = 1e-9 + 1e-6 * DEPLACEMENT**2
    energie = energie_d_un_condensateur(quadratique, sp.Float(TENSION))
    tangente_en_zero = float(force_a_courant_constant(energie).subs(DEPLACEMENT, 0))
    variation = float(
        (
            quadratique.subs(DEPLACEMENT, DEPLACEMENT_IMPOSE)
            - quadratique.subs(DEPLACEMENT, 0)
        ),
    )
    secante = force_sur_une_armature(TENSION, variation, DEPLACEMENT_IMPOSE)
    assert tangente_en_zero == pytest.approx(0.0)
    assert secante > 0


def test_le_sens_de_la_force_est_celui_qui_augmente_la_capacite() -> None:
    """Une armature est toujours **attiree** : la force rapproche les plaques.

    La capacite croit quand l'ecartement diminue, donc l'energie a
    tension constante croit aussi : le systeme tend vers le
    rapprochement. Le module rend une grandeur positive, et c'est le
    document qui porte le sens -- l'enonce ne demande que le module.
    """
    force = force_sur_une_armature(TENSION, -VARIATION_DE_CAPACITE, DEPLACEMENT_IMPOSE)
    assert force > 0
    assert force == pytest.approx(
        force_sur_une_armature(TENSION, VARIATION_DE_CAPACITE, DEPLACEMENT_IMPOSE),
    )


def test_un_deplacement_nul_est_refuse() -> None:
    """La difference finie y divise par zero, et la derivee n'est pas definie."""
    with pytest.raises(ValueError, match="deplacement nul"):
        force_sur_une_armature(TENSION, VARIATION_DE_CAPACITE, 0.0)
