"""Les gestes d'integration, et pourquoi ce sont eux qu'il faut exposer.

`sympy.integrate` rend une primitive. Il ne rend pas la methode -- et la
methode est tout ce qu'un corrige pas a pas apporte au-dela du corrige
officiel, qui donne deja la reponse. Le fascicule du cours le dit
lui-meme : ses solutions sont des indications de geste, « changement de
variable u = sin x », « integrations par parties », et non des resultats
secs.

Les valeurs attendues ci-dessous sont celles que **le fascicule du cours
publie** : `Mathematique/EXS/Exs_01_Integration_matrices_determinants.pdf`
donne quatorze primitives. Ses enonces sont des images dont rien ne
s'extrait, mais ses reponses sont du texte -- et une reponse de primitive
se verifie en la derivant. C'est donc l'integrande qui est reconstruit
par derivation, puis reintegre : si la bibliotheque retombe sur la reponse
du cours, elle traite bien la classe d'integrale que le cours examine.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.integration.parties import IntegrationParParties
from polytech.maths.domain.integration.primitives import Primitive
from polytech.maths.domain.integration.substitution import Substitution
from polytech.maths.domain.symboles import CONSTANTE
from polytech.maths.domain.symboles import VARIABLE as X


def test_une_integration_par_parties_rend_l_etape_et_non_le_resultat() -> None:
    """U v - integrale de v du : c'est cette ligne que le lecteur doit voir."""
    geste = IntegrationParParties(u=sp.log(X), derivee_de_v=sp.Integer(1))
    etape = geste.etape()
    assert sp.simplify(etape - (X * sp.log(X) - sp.Integral(sp.Integer(1), X))) == 0


def test_l_integration_par_parties_du_cours_sur_le_logarithme() -> None:
    """Reponse 4 du fascicule : x ln x - x + C."""
    geste = IntegrationParParties(u=sp.log(X), derivee_de_v=sp.Integer(1))
    assert sp.simplify(geste.resultat() - (X * sp.log(X) - X)) == 0


def test_l_integration_par_parties_se_repete_jusqu_a_epuisement() -> None:
    """Reponse 3 du fascicule : (x^3 - 3x^2 + 6x - 6) e^x + C.

    Trois parties successives. Le corrige doit pouvoir les montrer une a
    une, d'ou un geste qui s'itere au lieu d'un appel qui conclut.
    """
    geste = IntegrationParParties(u=X**3, derivee_de_v=sp.exp(X))
    attendu = (X**3 - 3 * X**2 + 6 * X - 6) * sp.exp(X)
    assert sp.simplify(geste.resultat() - attendu) == 0


def test_une_substitution_rend_l_integrale_en_la_nouvelle_variable() -> None:
    """Reponse 1 du fascicule : u = sin^2 x, primitive e^{sin^2 x} + C."""
    integrande = sp.diff(sp.exp(sp.sin(X) ** 2), X)
    geste = Substitution(integrande=integrande, expression=sp.sin(X) ** 2)
    assert sp.simplify(geste.resultat() - sp.exp(sp.sin(X) ** 2)) == 0


def test_une_substitution_nomme_la_variable_nouvelle_et_sa_differentielle() -> None:
    """Le lecteur doit voir `u = sin^2 x` et `du = 2 sin x cos x dx`."""
    geste = Substitution(integrande=sp.exp(X), expression=sp.sin(X) ** 2)
    assert geste.nouvelle_variable().name == "u"
    attendu = sp.diff(sp.sin(X) ** 2, X)
    assert sp.simplify(geste.differentielle() - attendu) == 0


def test_une_substitution_qui_ne_simplifie_rien_est_refusee() -> None:
    """Si la variable d'origine subsiste, le changement a manque son but.

    Le dire en levant vaut mieux que de rendre une integrale en `u` ou
    tramaine encore un `x` : elle serait inutilisable, et le corrige
    l'afficherait sans que rien ne signale la faute.
    """
    geste = Substitution(integrande=sp.exp(X) + X**5, expression=sp.sin(X))
    with pytest.raises(ValueError, match="ne fait pas disparaitre"):
        geste.integrale_transformee()


@pytest.mark.parametrize(
    ("attendue", "commentaire"),
    [
        (sp.exp(sp.sin(X) ** 2), "reponse 1 : changement de variable u = sin^2 x"),
        ((X**3 - 3 * X**2 + 6 * X - 6) * sp.exp(X), "reponse 3 : parties repetees"),
        (X * sp.log(X) - X, "reponse 4 : parties sur ln x"),
        (X * sp.asin(X) + sp.sqrt(1 - X**2), "reponse 4 : parties sur arcsin x"),
        (
            sp.Rational(1, 2) * (sp.asin(X) - X * sp.sqrt(1 - X**2)),
            "reponse 11 : t = arcsin x",
        ),
    ],
)
def test_la_primitive_se_verifie_en_la_derivant(
    attendue: sp.Expr,
    commentaire: str,
) -> None:
    """Le seul controle qui ne depende d'aucune table : deriver la reponse.

    L'integrande est reconstruit en derivant la reponse publiee par le
    cours ; la bibliotheque le reintegre et doit retomber dessus.
    """
    primitive = Primitive.de(sp.diff(attendue, X))
    assert primitive.verifiee, commentaire
    assert sp.simplify(primitive.expression - attendue) == 0, commentaire


def test_une_primitive_que_sympy_ne_sait_pas_calculer_est_nommee() -> None:
    """Rendre une `Integral` non evaluee ferait composer « int(...) dx ».

    Le document afficherait alors l'integrale au lieu de sa valeur, sans
    qu'aucun test ne tombe. La garde est donc au moment du calcul.
    """
    with pytest.raises(ValueError, match="primitive non elementaire"):
        Primitive.de(sp.sin(X) / sp.log(X))


def test_une_primitive_porte_la_constante_du_cours() -> None:
    """Le cours ecrit `+ C` explicitement : le corrige doit pouvoir aussi.

    La constante est celle de `symboles`, et non un `Symbol("C")` forge
    ici : sympy tiendrait les deux pour differents, puisque l'un porte
    l'hypothese `real` et l'autre non. C'est le piege que le module des
    symboles existe precisement pour eviter.
    """
    primitive = Primitive.de(sp.Integer(1))
    assert primitive.avec_constante().has(CONSTANTE)
