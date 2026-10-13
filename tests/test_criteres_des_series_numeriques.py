"""Annexe A des mathematiques appliquees : les criteres que rien ne portait.

Sources :

- `Mathematique/SYLLABUS/Syllabus_Mathematiques_appliquees_2016_2017.pdf`,
  annexe A (pages 194 a 196) : suite de Cauchy, criteres 1, 2, 4, 5 et 6,
  series de Riemann ;
- `Mathematique/SYLLABUS/Syllabus_Notes_de_cours_2011.pdf`, pages 121 et
  122 (imprimees 107 et 108), qui donnent les memes criteres avec leurs
  exemples : la tranche de `somme 1/n` qui depasse 1/2, `2^n / (n^7 + 5n^5
  + 1)` dont le terme ne tend pas vers 0, `1 / (n^2 + 2n + 1)` majoree par
  `1/n^2`, `1 / (n - 1/2)` minoree par `1/n`.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.series.criteres import (
    Comparaison,
    borne_des_sommes_partielles,
    comparer_par_le_quotient,
    converge_par_majoration,
    diverge_par_minoration,
    est_de_cauchy,
    le_terme_general_tend_vers_zero,
    serie_de_riemann_converge,
    tranche_de_cauchy,
)

N = sp.Symbol("n", integer=True, positive=True)
R = sp.Rational


@pytest.mark.parametrize("rang", [1, 2, 5, 20, 100])
def test_la_tranche_de_la_serie_harmonique_depasse_un_demi(rang: int) -> None:
    # Critere 1 : en prenant p = n, la tranche ne devient jamais petite.
    assert tranche_de_cauchy(1 / N, N, rang, rang) >= R(1, 2)


def test_la_tranche_d_une_serie_convergente_devient_petite() -> None:
    assert tranche_de_cauchy(1 / N**2, N, 100, 100) < R(1, 100)
    assert tranche_de_cauchy(1 / N, N, 1, 2) == R(1, 2) + R(1, 3)


def test_une_tranche_vide_leve() -> None:
    with pytest.raises(ValueError, match="au moins un terme"):
        tranche_de_cauchy(1 / N, N, 3, 0)


def test_le_critere_2_ecarte_l_exemple_des_notes() -> None:
    assert not le_terme_general_tend_vers_zero(2**N / (N**7 + 5 * N**5 + 1), N)
    assert le_terme_general_tend_vers_zero(1 / N, N)


def test_une_suite_convergente_est_de_cauchy_et_reciproquement() -> None:
    assert est_de_cauchy((N + 1) / N, N)
    assert not est_de_cauchy((-1) ** N, N)
    assert not est_de_cauchy(N, N)


@pytest.mark.parametrize(
    ("alpha", "attendu"),
    [(2, True), (R(3, 2), True), (1, False), (R(1, 2), False), (-1, False)],
)
def test_les_series_de_riemann_convergent_au_dela_de_un(
    alpha: object, *, attendu: bool
) -> None:
    assert serie_de_riemann_converge(alpha) is attendu
    assert sp.Sum(1 / N**alpha, (N, 1, sp.oo)).is_convergent() == attendu


def test_un_exposant_de_signe_inconnu_leve() -> None:
    with pytest.raises(ValueError, match="ne se compare pas a 1"):
        serie_de_riemann_converge(sp.Symbol("alpha"))


def test_le_critere_4_les_sommes_partielles_bornees() -> None:
    assert borne_des_sommes_partielles(1 / N**2, N) == sp.pi**2 / 6
    assert borne_des_sommes_partielles(1 / N, N) == sp.oo


def test_le_critere_4_ne_vaut_que_pour_des_termes_positifs() -> None:
    with pytest.raises(ValueError, match="termes positifs"):
        borne_des_sommes_partielles((-1) ** N / N, N)


def test_le_critere_5_par_majoration() -> None:
    assert converge_par_majoration(1 / (N**2 + 2 * N + 1), 1 / N**2, N)


def test_le_critere_5_par_minoration() -> None:
    assert diverge_par_minoration(1 / (N - R(1, 2)), 1 / N, N)


def test_une_majorante_qui_diverge_ne_dit_rien() -> None:
    with pytest.raises(ValueError, match="aucune conclusion"):
        converge_par_majoration(1 / N**2, 1 / N, N)


def test_une_minorante_qui_converge_ne_dit_rien() -> None:
    with pytest.raises(ValueError, match="aucune conclusion"):
        diverge_par_minoration(1 / N, 1 / N**2, N)


def test_une_fausse_majorante_leve() -> None:
    with pytest.raises(ValueError, match="n'est pas etabli"):
        converge_par_majoration(1 / N, 1 / N**2, N)


def test_le_critere_6_et_sa_remarque() -> None:
    assert comparer_par_le_quotient(1 / (N**2 + 1), 1 / N**2, N) is (
        Comparaison.EQUIVALENTES
    )
    assert comparer_par_le_quotient(1 / N**3, 1 / N**2, N) is Comparaison.NEGLIGEABLE
    assert comparer_par_le_quotient(1 / N, 1 / N**2, N) is Comparaison.PREPONDERANTE


def test_le_critere_6_exige_des_termes_positifs() -> None:
    with pytest.raises(ValueError, match="termes positifs"):
        comparer_par_le_quotient((-1) ** N / N, 1 / N, N)


def test_le_critere_5_refuse_un_terme_negatif_meme_majore() -> None:
    with pytest.raises(ValueError, match="termes positifs"):
        converge_par_majoration(-1 / N, 1 / N**2, N)


def test_le_critere_6_exige_aussi_la_seconde_serie_positive() -> None:
    with pytest.raises(ValueError, match="v_n"):
        comparer_par_le_quotient(1 / N, -1 / N, N)
