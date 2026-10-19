"""Activite d'integration 2 : chaque resultat refait par un chemin sans rapport.

Le corrige obtient ses profils et ses lois par le calcul formel. Comme
pour l'activite 1, un resultat verifie deux fois par le meme moteur n'est
pas verifie : ces recoupements passent par des schemas numeriques de
scipy, qui ne savent rien des formes closes.
"""

from __future__ import annotations

import pytest

from polytech.fluides.application import activite_2_recoupements as rec


def test_le_profil_du_film_par_une_resolution_aux_limites() -> None:
    """`solve_bvp` sur `w*'' = 1`, `w*(0) = 0`, `w*'(1) = 0`."""
    assert rec.ecart_du_profil() < 1e-9


def test_le_debit_du_film_par_quadrature() -> None:
    """L'aire sous le profil normalise vaut `-1/3`."""
    assert rec.ecart_du_debit() < 1e-12


def test_le_remplissage_par_runge_kutta() -> None:
    """`dz*/dt* = 2 (1 - sqrt z*)`, integree jusqu'a la duree de la forme close."""
    assert rec.ecart_du_remplissage() < 1e-6


def test_la_puissance_du_bateau_par_la_formule_symbolique() -> None:
    assert rec.ecart_de_la_puissance() < 1e-12


def test_les_ecarts_sont_tous_non_negatifs() -> None:
    ecarts = (
        rec.ecart_du_profil(),
        rec.ecart_du_debit(),
        rec.ecart_du_remplissage(),
        rec.ecart_de_la_puissance(),
    )
    assert all(e >= 0 for e in ecarts)


class _Echec:
    success = False
    message = "le solveur a echoue"


def test_un_echec_de_la_resolution_aux_limites_leve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rec, "solve_bvp", lambda *_a, **_k: _Echec())
    with pytest.raises(RuntimeError, match="le solveur a echoue"):
        rec.ecart_du_profil()


def test_un_echec_du_runge_kutta_leve(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rec, "solve_ivp", lambda *_a, **_k: _Echec())
    with pytest.raises(RuntimeError, match="le solveur a echoue"):
        rec.niveau_par_runge_kutta(0.5)


@pytest.mark.parametrize("fraction", [0.0, 1.0, 1.2])
def test_une_fraction_hors_de_l_intervalle_ouvert_leve(fraction: float) -> None:
    with pytest.raises(ValueError, match="fraction du niveau maximal hors de"):
        rec.niveau_par_runge_kutta(fraction)


def test_le_remplissage_est_eprouve_jusqu_a_la_saturation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pres de `z_max` la racine raidit l'equation : c'est la que l'on recoupe."""
    lus: list[float] = []

    def lire(fraction: float) -> float:
        lus.append(fraction)
        return fraction

    monkeypatch.setattr(rec, "niveau_par_runge_kutta", lire)
    rec.ecart_du_remplissage()
    assert lus == [0.5, 0.9, 0.99]
