"""L'aerodynamique, dernier fascicule du depot a entrer dans la mesure.

Sa structure a du etre transcrite a la main : sa conversion n'avait laisse
aucun titre. Les trente-deux titres ainsi recouvres disent ce que le
chapitre enseigne, et ce que le module ne portait pas :

* la **couche limite** et ses trois epaisseurs -- a 99 pour cent, de
  deplacement, de quantite de mouvement -- qui ne mesurent pas la meme
  chose et dont la confusion est l'erreur du chapitre ;
* l'equation de **Prandtl**, obtenue en comparant les ordres de grandeur
  dans Navier-Stokes adimensionnel, et la solution auto-similaire de
  **Blasius** qui en decoule ;
* le **frottement parietal** et les deux formes de trainee -- friction et
  pression -- dont la planche montre qu'elles se partagent differemment
  selon la forme du corps ;
* le **decrochage**, et la trainee **induite** par la portance, qui est
  celle qu'un profil fini ne peut pas eviter.

Les ancrages sont les constantes de Blasius, qui sont exactes :
`delta_99 = 5,0 x / racine(Re_x)`, `C_f = 0,664 / racine(Re_x)` en
laminaire, et le rapport `delta* / delta_99 = 0,344` qui ne depend ni de
la vitesse ni de la position.
"""

from __future__ import annotations

import pytest

from polytech.fluides.domain.aerodynamique import (
    Ecoulement,
    SurfacePortante,
    trainee_induite,
)
from polytech.fluides.domain.couche_limite import (
    CONSTANTE_DE_BLASIUS,
    coefficient_de_frottement_local,
    epaisseur_de_couche_limite,
    epaisseur_de_deplacement,
    epaisseur_de_quantite_de_mouvement,
    est_laminaire,
    reynolds_local,
)
from polytech.fluides.domain.grandeurs import MasseVolumique

AIR = MasseVolumique(kg_par_m3=1.2)


def test_le_reynolds_local_croit_avec_la_distance_au_bord() -> None:
    """`Re_x = U x / nu` : il mesure la distance parcourue, pas la geometrie."""
    proche = reynolds_local(vitesse=10.0, distance=0.1, viscosite=1.5e-5)
    loin = reynolds_local(vitesse=10.0, distance=0.2, viscosite=1.5e-5)
    assert loin == pytest.approx(2 * proche)


def test_une_viscosite_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="viscosite"):
        reynolds_local(vitesse=10.0, distance=0.1, viscosite=0.0)


def test_la_couche_limite_reste_laminaire_sous_cinq_cent_mille() -> None:
    """La transition se situe vers `5.10^5` : le fascicule le pose ainsi."""
    assert est_laminaire(1e5)
    assert not est_laminaire(1e6)


def test_l_epaisseur_de_couche_limite_croit_comme_la_racine_de_la_distance() -> None:
    """`delta = 5,0 x / racine(Re_x)`, donc `delta` croit comme `racine(x)`.

    C'est la forme que Blasius donne, et la constante `5,0` est exacte --
    elle vient de la resolution numerique de son equation.
    """
    assert pytest.approx(5.0) == CONSTANTE_DE_BLASIUS
    proche = epaisseur_de_couche_limite(vitesse=10.0, distance=0.1, viscosite=1.5e-5)
    loin = epaisseur_de_couche_limite(vitesse=10.0, distance=0.4, viscosite=1.5e-5)
    assert loin == pytest.approx(2 * proche)


def test_l_epaisseur_de_couche_limite_refuse_le_regime_turbulent() -> None:
    """Blasius est une solution **laminaire** : l'appliquer au-dela serait faux.

    Le dire en levant vaut mieux que de rendre une epaisseur qui aurait
    l'air d'un resultat.
    """
    with pytest.raises(ValueError, match="solution de Blasius"):
        epaisseur_de_couche_limite(vitesse=100.0, distance=1.0, viscosite=1.5e-5)


def test_les_trois_epaisseurs_sont_dans_un_rapport_fixe() -> None:
    """`delta* / delta = 0,344` et `theta / delta = 0,133`, en laminaire.

    Ces rapports ne dependent ni de la vitesse ni de la position : ils
    sont une propriete du **profil** de Blasius. C'est ce qui les rend
    utilisables, et c'est ce que le fascicule demande de retenir.
    """
    arguments = {"vitesse": 10.0, "distance": 0.1, "viscosite": 1.5e-5}
    delta = epaisseur_de_couche_limite(**arguments)
    deplacement = epaisseur_de_deplacement(**arguments)
    mouvement = epaisseur_de_quantite_de_mouvement(**arguments)
    assert deplacement / delta == pytest.approx(0.344, abs=0.001)
    assert mouvement / delta == pytest.approx(0.133, abs=0.001)


def test_les_trois_epaisseurs_sont_ordonnees() -> None:
    """`theta < delta* < delta` : les confondre change le resultat d'un facteur sept."""
    arguments = {"vitesse": 10.0, "distance": 0.1, "viscosite": 1.5e-5}
    assert (
        epaisseur_de_quantite_de_mouvement(**arguments)
        < epaisseur_de_deplacement(**arguments)
        < epaisseur_de_couche_limite(**arguments)
    )


def test_le_coefficient_de_frottement_local_de_blasius() -> None:
    """`C_f = 0,664 / racine(Re_x)` : il **decroit** le long de la paroi.

    C'est contre-intuitif -- le frottement total croit, mais le
    coefficient local decroit, parce que la couche limite epaissit et
    adoucit le gradient a la paroi.
    """
    au_bord = coefficient_de_frottement_local(1e4)
    plus_loin = coefficient_de_frottement_local(4e4)
    assert au_bord == pytest.approx(0.664 / 100.0)
    assert plus_loin == pytest.approx(au_bord / 2)


def test_le_frottement_local_refuse_le_regime_turbulent() -> None:
    with pytest.raises(ValueError, match="solution de Blasius"):
        coefficient_de_frottement_local(1e6)


def test_la_trainee_induite_croit_comme_le_carre_de_la_portance() -> None:
    """`C_Di = C_L^2 / (pi e AR)` : doubler la portance la quadruple.

    C'est la trainee qu'un profil **fini** ne peut pas eviter -- elle vient
    des tourbillons de bout d'aile -- et c'est pourquoi un planeur a de
    tres grands allongements.
    """
    simple = trainee_induite(portance=0.5, allongement=8.0)
    double = trainee_induite(portance=1.0, allongement=8.0)
    assert double == pytest.approx(4 * simple)


def test_un_grand_allongement_reduit_la_trainee_induite() -> None:
    """La raison des ailes longues et fines, et elle est quantitative."""
    court = trainee_induite(portance=1.0, allongement=4.0)
    long_ = trainee_induite(portance=1.0, allongement=16.0)
    assert long_ == pytest.approx(court / 4)


def test_un_allongement_nul_est_refuse() -> None:
    with pytest.raises(ValueError, match="allongement"):
        trainee_induite(portance=1.0, allongement=0.0)


def test_les_coefficients_existants_ne_sont_pas_alteres() -> None:
    """Non-regression : les ajouts ne touchent pas a ce qui servait deja."""
    aile = SurfacePortante(
        ecoulement=Ecoulement(fluide=AIR, vitesse=50.0),
        surface_de_reference=20.0,
    )
    assert aile.finesse(1.2, 0.08) == pytest.approx(15.0)
    assert aile.coefficient(aile.force(0.5)) == pytest.approx(0.5)
