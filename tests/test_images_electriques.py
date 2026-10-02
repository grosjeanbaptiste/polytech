"""L'exercice 5 de statique : charge devant un plan, par images.

La methode des images remplace un probleme a conditions aux limites par
un probleme a deux charges libres. Elle est exacte -- pas approchee --
par un argument d'unicite : si une distribution fictive produit le bon
potentiel sur la frontiere, elle produit le bon champ partout dans le
demi-espace, parce que la solution de Laplace a conditions fixees est
unique.

Le corrige manuscrit mene le calcul vectoriel juste et s'arrete a la
densite induite. Il ne fait pas la verification qui confirmerait tout :
**l'integrale de la densite sur le plan entier doit valoir exactement
l'oppose de la charge**. Elle le vaut, et ce test le montre -- ce qui
confirme d'un coup le coefficient et l'exposant trois demis, les deux
endroits ou une erreur se logerait.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.electrostatique.images import (
    champ_sur_le_plan,
    charge_induite_totale,
    densite_induite,
    energie_d_interaction,
    force_sur_la_charge,
)

CHARGE, DISTANCE = sp.symbols("q a", positive=True)
ORDONNEE, COTE = sp.symbols("y z", real=True)


def test_le_champ_sur_le_plan_est_normal_au_plan() -> None:
    """Les composantes tangentielles s'annulent, et c'est le resultat.

    Les deux charges sont symetriques : leurs contributions tangentielles
    sont egales et opposees, leurs contributions normales s'ajoutent. Un
    conducteur impose precisement cela -- le champ y est normal -- et le
    retrouver verifie la methode plutot que de la croire.
    """
    champ = champ_sur_le_plan(CHARGE, DISTANCE, ORDONNEE, COTE)
    assert sp.simplify(champ[1]) == 0
    assert sp.simplify(champ[2]) == 0
    assert sp.simplify(champ[0]) != 0


def test_le_champ_est_dirige_vers_la_charge() -> None:
    """Sa composante normale est **negative** pour une charge positive.

    Le champ pointe de la charge vers le plan, donc selon `-u_x` : les
    lignes de champ partent de la charge positive et aboutissent
    perpendiculairement au conducteur.
    """
    champ_x = champ_sur_le_plan(CHARGE, DISTANCE, ORDONNEE, COTE)[0]
    assert sp.simplify(champ_x).could_extract_minus_sign()


def test_le_champ_decroit_comme_la_puissance_trois_de_la_distance() -> None:
    """Au loin, le champ decroit en `1/r^3` et non en `1/r^2`.

    C'est la signature d'un **dipole** : la charge et son image en
    forment un, et vu de loin un dipole decroit d'une puissance de plus
    qu'une charge seule. Le test le verifie par un developpement
    asymptotique, ce qui est un controle de la **forme** et non d'une
    valeur.
    """
    rayon = sp.Symbol("r", positive=True)
    champ_x = champ_sur_le_plan(CHARGE, DISTANCE, rayon, sp.Integer(0))[0]
    terme = sp.limit(champ_x * rayon**3, rayon, sp.oo)
    assert sp.simplify(terme - (-CHARGE * DISTANCE / (2 * sp.pi * _epsilon()))) == 0


def _epsilon() -> sp.Symbol:
    """La permittivite du vide, symbolique."""
    return sp.Symbol("varepsilon_0", positive=True)


def test_la_densite_induite_est_le_champ_fois_la_permittivite() -> None:
    """`sigma = u_n . D` avec `D = epsilon_0 E`, le conducteur annulant l'autre cote.

    La relation de passage a deux termes, et l'un est nul : a l'interieur
    d'un conducteur en equilibre il n'y a ni champ ni deplacement. C'est
    ce qui laisse `sigma = epsilon_0 E_n`.
    """
    champ = champ_sur_le_plan(CHARGE, DISTANCE, ORDONNEE, COTE)
    densite = densite_induite(CHARGE, DISTANCE, ORDONNEE, COTE)
    assert sp.simplify(densite - _epsilon() * champ[0]) == 0


def test_la_densite_induite_est_maximale_au_pied_de_la_charge() -> None:
    """Au point le plus proche, et elle y vaut `-q / (2 pi a^2)`.

    La decroissance ensuite est rapide -- en `1/r^3` au loin -- de sorte
    que la charge induite se concentre sur un disque de rayon comparable
    a la distance `a`.
    """
    au_pied = densite_induite(CHARGE, DISTANCE, sp.Integer(0), sp.Integer(0))
    assert sp.simplify(au_pied + CHARGE / (2 * sp.pi * DISTANCE**2)) == 0


def test_la_charge_induite_totale_vaut_exactement_l_oppose_de_la_charge() -> None:
    """La verification que le corrige officiel ne fait pas, et la plus forte.

    `integrale de sigma sur le plan = -q`, exactement. Elle confirme d'un
    coup le coefficient `1/(2 pi)` et l'exposant trois demis : une erreur
    sur l'un ou sur l'autre donnerait un multiple de `-q`, ou une
    integrale divergente.

    Et elle a un sens physique : le plan relie a la terre porte
    exactement la charge de l'image, ce qui est la coherence meme de la
    methode.
    """
    assert sp.simplify(charge_induite_totale(CHARGE, DISTANCE) + CHARGE) == 0


def test_la_force_sur_la_charge_est_celle_de_son_image() -> None:
    """`F = q^2 / (4 pi eps_0 (2a)^2)` : la loi de Coulomb a distance `2a`.

    C'est l'autre consequence de la methode, et elle surprend : la force
    sur la charge reelle est **exactement** celle qu'exercerait une
    charge opposee placee a `2a`. La distance qui compte est donc celle
    de la charge a son image, non celle au plan.
    """
    force = force_sur_la_charge(CHARGE, DISTANCE)
    attendue = CHARGE**2 / (4 * sp.pi * _epsilon() * (2 * DISTANCE) ** 2)
    assert sp.simplify(force - attendue) == 0


def test_la_force_decroit_comme_le_carre_de_la_distance_au_plan() -> None:
    """Doubler la distance au plan divise la force par quatre.

    Verifie par un **rapport** et non par une forme : sympy developpe
    `1/(2a)^2` en `1/(16 a^2)` des la construction, de sorte qu'aucun
    test ne peut porter sur l'ecriture. Le rapport, lui, est la propriete
    physique -- et c'est elle qu'on veut tenir.
    """
    simple = force_sur_la_charge(CHARGE, DISTANCE)
    double = force_sur_la_charge(CHARGE, 2 * DISTANCE)
    assert sp.simplify(simple / double - 4) == 0


def test_l_energie_d_interaction_est_la_moitie_de_celle_de_la_paire() -> None:
    """Le piege classique de la methode : la force est entiere, l'energie moitie.

    L'energie d'une **paire** de charges opposees distantes de `2a` vaut
    `-q^2 / (4 pi eps_0 2a)`. Celle de la charge devant le plan n'en vaut
    que la **moitie**, et la raison est que l'image n'est pas une charge
    reelle : on ne peut pas compter une energie d'interaction mutuelle
    avec un objet fictif. Le calcul rigoureux -- par le travail
    d'amenee -- donne le facteur un demi.

    Ce test le fixe, parce que c'est la faute que la methode des images
    invite a commettre.
    """
    paire = -(CHARGE**2) / (4 * sp.pi * _epsilon() * 2 * DISTANCE)
    energie = energie_d_interaction(CHARGE, DISTANCE)
    assert sp.simplify(energie - paire / 2) == 0


def test_la_composante_radiale_de_la_force_est_negative_donc_attractive() -> None:
    """`F_a = -dW/da < 0` : la force pousse vers le plan, pas loin de lui.

    `a` croissant eloigne la charge du plan ; une composante negative
    selon `a` est donc une attraction. C'est la **composante signee**, a
    distinguer du module que rend `force_sur_la_charge` -- et cette
    distinction est exactement ce que le premier jet de ce test avait
    confondu, en soustrayant un module d'une composante.
    """
    energie = energie_d_interaction(CHARGE, DISTANCE)
    assert sp.simplify(-sp.diff(energie, DISTANCE)).is_negative


def test_le_module_de_la_force_est_la_derivee_de_l_energie() -> None:
    """`|F| = dW/da`, et c'est la verification croisee de l'exercice.

    Deux chemins independants menent au meme resultat : la loi de Coulomb
    appliquee a l'image, et la derivation de l'energie. Qu'ils
    s'accordent valide les deux -- et c'est precisement le facteur un
    demi de l'energie qui le permet. Avec l'energie de la paire entiere,
    la derivee donnerait le double du module, donc une contradiction.

    Le signe `+` n'est pas une commodite : `-dW/da` est la composante
    signee, negative parce que la force est attractive, et son oppose est
    donc le module. Le test precedent tient ce signe separement.
    """
    energie = energie_d_interaction(CHARGE, DISTANCE)
    derivee = sp.diff(energie, DISTANCE)
    assert sp.simplify(derivee - force_sur_la_charge(CHARGE, DISTANCE)) == 0


def test_l_energie_d_interaction_est_negative() -> None:
    """Negative, donc liante : il faut fournir du travail pour eloigner.

    C'est la signature d'une attraction, et elle doit s'accorder avec le
    signe de la force -- qui est rendue en module, positive.
    """
    assert energie_d_interaction(CHARGE, DISTANCE).is_negative
