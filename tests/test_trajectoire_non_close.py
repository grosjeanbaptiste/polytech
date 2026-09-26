"""Une integration qui n'aboutit pas doit le dire, pas rendre un a-peu-pres.

`trajectoire` promet « la position d'une particule partie de (x0, y0) ».
Pour le champ B elle rendait une expression contenant une constante C1 non
resolue et des integrales non evaluees : la condition initiale n'avait pas
ete appliquee, et rien ne le signalait. Le defaut etait latent -- la
methode n'etait appelee que sur le champ C.
"""

import pytest
import sympy as sp

from polytech.fluides.application import enonce
from polytech.fluides.domain.cinematique.lignes import (
    TrajectoireNonElementaire,
    trajectoire,
)
from polytech.fluides.domain.cinematique.symboles import T


def test_la_trajectoire_de_c_est_close_et_verifie_sa_condition_initiale() -> None:
    """x0 e^{2t}, y0 e^{-2t} : en t = 0 on doit retrouver le point de depart."""
    abscisse, ordonnee = trajectoire(enonce.CHAMP_C.u, enonce.CHAMP_C.v, 3.0, 5.0)
    assert sp.simplify(abscisse.subs(T, 0) - 3.0) == 0
    assert sp.simplify(ordonnee.subs(T, 0) - 5.0) == 0


def test_la_trajectoire_de_b_est_declaree_non_elementaire() -> None:
    """dy/dt = x - y avec x = x0 e^{t^2/2} n'a pas de primitive elementaire."""
    with pytest.raises(TrajectoireNonElementaire, match="forme close"):
        trajectoire(enonce.CHAMP_B.u, enonce.CHAMP_B.v, 1.5, 2.0)


def test_le_champ_sait_dire_si_sa_trajectoire_est_close() -> None:
    """Le corrige doit pouvoir l'affirmer sans provoquer d'exception."""
    assert enonce.CHAMP_C.trajectoire_est_elementaire() is True
    assert enonce.CHAMP_B.trajectoire_est_elementaire() is False


def test_la_composante_x_de_b_reste_elementaire() -> None:
    """Seule l'ordonnee resiste : dx/dt = xt s'integre sans difficulte."""
    assert enonce.CHAMP_B.trajectoire_est_elementaire() is False
    abscisse = sp.dsolve(
        sp.Eq(
            sp.Function("f")(T).diff(T),
            enonce.CHAMP_B.u.subs(sp.Symbol("x", real=True), sp.Function("f")(T)),
        ),
        sp.Function("f")(T),
        ics={sp.Function("f")(0): 1.5},
    ).rhs
    assert not abscisse.has(sp.Integral)
