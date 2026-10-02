"""Un reseau lineaire en regime stationnaire, resolu par ses noeuds.

Les corriges manuscrits de Thevenin et Norton procedent tous de la meme
facon : une loi des mailles bien choisie, puis une substitution. C'est
rapide, et c'est aussi ce qui rend l'erreur de signe facile -- le choix
de la maille est fait a la main.

Ce module resout le meme reseau par une route qui ne choisit rien : la
loi des noeuds ecrite a chaque noeud, les sources de tension ajoutant
chacune une inconnue. Les deux methodes ne partagent que les lois de
Kirchhoff, et leur accord verifie les corriges.

Trois proprietes generales sont eprouvees ici sur des reseaux dont le
resultat se connait de tete. Les exercices du recueil ont leur propre
fichier.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.circuits.reseau import (
    Condensateur,
    Inductance,
    Reseau,
    Resistance,
    SourceDeCourant,
    SourceDeTension,
)

TENSION = sp.Symbol("E", positive=True)
COURANT = sp.Symbol("I", positive=True)
PREMIERE, SECONDE = sp.symbols("R_1 R_2", positive=True)


def test_un_diviseur_de_tension_partage_selon_les_resistances() -> None:
    """`V = E R_2 / (R_1 + R_2)` au point milieu : le cas d'ecole."""
    reseau = Reseau(
        elements=(
            SourceDeTension("haut", "masse", TENSION),
            Resistance("haut", "milieu", PREMIERE),
            Resistance("milieu", "masse", SECONDE),
        ),
        reference="masse",
    )
    attendu = TENSION * SECONDE / (PREMIERE + SECONDE)
    assert sp.simplify(reseau.potentiels()["milieu"] - attendu) == 0


def test_une_source_de_courant_impose_sa_tension_par_la_loi_d_ohm() -> None:
    """Le courant va de `masse` a `haut` dans la source, donc `V = R I`."""
    reseau = Reseau(
        elements=(
            SourceDeCourant("masse", "haut", COURANT),
            Resistance("haut", "masse", PREMIERE),
        ),
        reference="masse",
    )
    assert sp.simplify(reseau.potentiels()["haut"] - PREMIERE * COURANT) == 0


def test_le_noeud_de_reference_est_au_potentiel_zero() -> None:
    """Il figure dans le resultat, pour que toute difference s'ecrive."""
    reseau = Reseau(
        elements=(
            Resistance("a", "masse", PREMIERE),
            SourceDeCourant("masse", "a", COURANT),
        ),
        reference="masse",
    )
    assert reseau.potentiels()["masse"] == 0


def test_le_courant_d_une_source_de_tension_est_celui_qu_elle_debite() -> None:
    """Dans une maille simple : `E / (R_1 + R_2)`, sortant par la borne plus."""
    source = SourceDeTension("haut", "masse", TENSION)
    reseau = Reseau(
        elements=(
            source,
            Resistance("haut", "milieu", PREMIERE),
            Resistance("milieu", "masse", SECONDE),
        ),
        reference="masse",
    )
    debite = reseau.courant_debite(source)
    assert sp.simplify(debite - TENSION / (PREMIERE + SECONDE)) == 0


def test_en_regime_une_inductance_est_un_fil() -> None:
    """Elle court-circuite la resistance qu'elle double : le noeud se confond."""
    reseau = Reseau(
        elements=(
            SourceDeTension("haut", "masse", TENSION),
            Resistance("haut", "milieu", PREMIERE),
            Inductance("haut", "milieu"),
            Resistance("milieu", "masse", SECONDE),
        ),
        reference="masse",
    )
    assert sp.simplify(reseau.potentiels()["milieu"] - TENSION) == 0


def test_en_regime_un_condensateur_est_un_circuit_ouvert() -> None:
    """Aucun courant ne traverse sa branche : la resistance en serie ne chute rien."""
    reseau = Reseau(
        elements=(
            SourceDeTension("haut", "masse", TENSION),
            Resistance("haut", "milieu", PREMIERE),
            Condensateur("milieu", "masse"),
        ),
        reference="masse",
    )
    assert sp.simplify(reseau.potentiels()["milieu"] - TENSION) == 0


def test_un_noeud_flottant_est_refuse() -> None:
    """Sans chemin vers la reference, son potentiel n'est pas determine."""
    reseau = Reseau(
        elements=(
            Resistance("a", "masse", PREMIERE),
            SourceDeCourant("masse", "a", COURANT),
            Resistance("b", "c", SECONDE),
        ),
        reference="masse",
    )
    with pytest.raises(ValueError, match="pas de solution unique"):
        reseau.potentiels()


def test_deux_sources_de_tension_contradictoires_sont_refusees() -> None:
    """Deux tensions differentes imposees aux memes bornes : aucun etat possible."""
    reseau = Reseau(
        elements=(
            SourceDeTension("a", "masse", sp.Integer(1)),
            SourceDeTension("a", "masse", sp.Integer(2)),
        ),
        reference="masse",
    )
    with pytest.raises(ValueError, match="pas de solution unique"):
        reseau.potentiels()


def test_une_reference_absente_du_reseau_est_refusee() -> None:
    """Nommer un noeud qui n'existe pas rendrait tous les potentiels arbitraires."""
    reseau = Reseau(elements=(Resistance("a", "b", PREMIERE),), reference="masse")
    with pytest.raises(ValueError, match="reference"):
        reseau.potentiels()
