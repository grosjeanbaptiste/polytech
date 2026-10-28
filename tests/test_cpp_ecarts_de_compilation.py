"""Programmation et algorithmique : deux ecarts que seul un compilateur etablit.

Les docstrings de `test_cpp_structures_de_controle.py` et de
`test_cpp_fichiers_et_structures.py` disent « confirme en compilant » sans que
rien ne le rejoue. Ici le compilateur du poste (celui de
`polytech.algorithmique.infrastructure.clang`, derriere le port
`polytech.algorithmique.domain.compilateur`) est appele sur la
planche telle qu'imprimee, puis sur sa seule correction : la difference d'un
caractere (planche 53) ou de deux (planche 152-153) fait passer le programme
du refus a l'acceptation. Sans compilateur sur le poste, ces tests se
suspendent, en le disant.

Sources : `ProgrammationAlgorithmique/SLIDES/Slides_2023_2024_Partie_1.pdf`,
planche 53 (switch du calcul de circonference et de surface) et planches
152-153 (`fonct(E *a)`). Le `void main` imprime est remplace par `int main`
dans les deux versions : un compilateur le refuse a part, et ce n'est pas
l'ecart etudie.
"""

from __future__ import annotations

import pytest

from polytech.algorithmique.domain.compilateur import (
    Diagnostic,
    EditionDeLiensImpossible,
)
from polytech.algorithmique.infrastructure.clang import (
    CompilateurClang,
    compilateur_disponible,
)

pytestmark = pytest.mark.skipif(
    not compilateur_disponible(), reason="aucun compilateur C++ sur le poste"
)
COMPILATEUR = CompilateurClang()

PLANCHE_53 = """#include <iostream>
using namespace std;
int main() {{
  const float PI = 3.14159; float rayon = 3.5;
  float circonference, surface; int choix = 1;
  switch (choix) {{
    case 1 : circonference = 2*PI*rayon; break;
    case 2 : surface = PI*rayon*rayon{point_virgule}
             break;
    default : cout << "Mauvais choix\\n";
  }}
  return 0;
}}
"""

PLANCHE_152 = """struct E {{ int x; int y; }};
void fonct(E *a) {{ {ecriture_x} = 1; {ecriture_y} = 2; }}
int main() {{ E e; fonct(&e); return 0; }}
"""


def _compile(source: str) -> Diagnostic:
    return COMPILATEUR.diagnostiquer(source)


def test_planche_53_sans_point_virgule_devant_le_second_break() -> None:
    """Imprimee : `surface = PI*rayon*rayon` puis `break;` -- refusee."""
    verdict = _compile(PLANCHE_53.format(point_virgule=""))
    assert not verdict.accepte
    assert verdict.mentionne("expected ';'")


def test_planche_53_avec_le_point_virgule_compile() -> None:
    """La correction tient en un caractere."""
    verdict = _compile(PLANCHE_53.format(point_virgule=";"))
    assert verdict.accepte, verdict.messages


def test_planche_152_etoile_a_point_x_ne_compile_pas() -> None:
    """`*a.x` se lit `*(a.x)` : `a` est un pointeur, il n'a pas de membre `x`."""
    source = PLANCHE_152.format(ecriture_x="*a.x", ecriture_y="*a.y")
    verdict = _compile(source)
    assert not verdict.accepte
    assert verdict.mentionne("member reference type")


@pytest.mark.parametrize(
    ("ecriture_x", "ecriture_y"),
    [("(*a).x", "(*a).y"), ("a->x", "a->y")],
)
def test_planche_152_les_deux_corrections_compilent(
    ecriture_x: str, ecriture_y: str
) -> None:
    """`(*a).x` et `a->x` designent le meme membre."""
    source = PLANCHE_152.format(ecriture_x=ecriture_x, ecriture_y=ecriture_y)
    verdict = _compile(source)
    assert verdict.accepte, verdict.messages


# --- Le reste du courrier au professeur de programmation ------------------
# Chaque affirmation « ne compile pas / ambigu / refuse » de
# `travaux/Transversal/courriers-aux-enseignants/06-programmation.md` est
# rejouee ici par le compilateur, avec le message qu'il rend.

AFFICHE = (
    "void affiche(int a,int b){} void affiche(float a,float b){}"
    " void affiche(int a){}\n"
)
DIAMANT = (
    "struct Personne{int id;}; struct Etudiant:Personne{}; struct Prof:Personne{};"
    " struct EA:Etudiant,Prof{};\n"
)
REFUS = {
    "pl14_auto_sans_initialiseur": (
        "int main(){ auto K, L; return 0; }",
        "with deduced type 'auto' requires an initializer",
    ),
    "pl14_chaine_hello_dans_cinq_cases": (
        'int main(){ char H[5]="Hello"; }',
        "array size is 5 but initializer has size 6",
    ),
    "pl122_affiche_5_2_6_3_ambigu": (
        AFFICHE + "int main(){ affiche(5.2,6.3); }",
        "call to 'affiche' is ambiguous",
    ),
    "pl122_affiche_float_int_ambigu": (
        AFFICHE + "int main(){ affiche(1.5f,2); }",
        "call to 'affiche' is ambiguous",
    ),
    "pl122_affiche_long_long_ambigu": (
        AFFICHE + "int main(){ affiche(1L,2L); }",
        "call to 'affiche' is ambiguous",
    ),
    "s4_pl16_defauts_redonnes_dans_la_definition": (
        (
            "class A{ public: A(int X=0,int Y=5); };\n"
            "A::A(int X=0,int Y=5){}\nint main(){}"
        ),
        "redefinition of default argument",
    ),
    "s4_pl20_obj_sans_argument_ambigu": (
        "class A{ public: A(){} A(int X=0,int Y=5){} };\nint main(){ A Obj; }",
        "call to constructor of 'A' is ambiguous",
    ),
    "s5_pl18_champ_de_type_incomplet": (
        "class Arbre{ public: int v; Arbre filsG, filsD; };\nint main(){}",
        "field has incomplete type 'Arbre'",
    ),
    "s5_pl18_int_etoile_recoit_tab_etoile": (
        "class Tab{}; int main(){ int* TabD = new Tab[3]; }",
        "cannot initialize a variable of type 'int *' with an rvalue of type 'Tab *'",
    ),
    "s5_pl22_defauts_ajoutes_a_la_redeclaration": (
        (
            "class Pixel{ public: Pixel(int,int,int); };\n"
            "Pixel::Pixel(int abs=1,int ord=2,int cl=3){}\nint main(){}"
        ),
        (
            "addition of default argument on redeclaration makes this"
            " constructor a default constructor"
        ),
    ),
    "s5_pl16_e_id_ambigu_en_heritage_en_diamant": (
        DIAMANT + "int main(){ EA e; e.id=1; }",
        "non-static member 'id' found in multiple base-class subobjects",
    ),
    "s5_pl60_begin_plus_n_sur_une_liste": (
        (
            "#include <list>\n"
            "int main(){ std::list<int> L2={1,2,3}; auto it=L2.begin()+1; }"
        ),
        "invalid operands to binary expression",
    ),
    "s5_pl61_auto_it_sans_initialiseur": (
        "int main(){ auto it; }",
        "with deduced type 'auto' requires an initializer",
    ),
    "min_2_9_ambigu_sous_using_namespace_std": (
        (
            "#include <algorithm>\nusing namespace std;\n"
            "template<class T> T min(T a,T b){return a<b?a:b;}\n"
            "int main(){ return min(2,9); }"
        ),
        "call to 'min' is ambiguous",
    ),
    "pl15_static_i_sans_type": (
        "struct P{ P(){ static i=1; } };\nint main(){}",
        "a type specifier is required for all declarations",
    ),
}

ACCEPTES = {
    "pl14_chaine_hello_dans_six_cases": 'int main(){ char H[6]="Hello"; }',
    "s5_pl16_e_etudiant_id_designe_une_seule_personne": (
        DIAMANT + "int main(){ EA e; e.Etudiant::id=1; }"
    ),
    "s5_pl72_advance_sur_une_liste": (
        "#include <list>\n#include <iterator>\n"
        "int main(){ std::list<int> L{1,2,3,4,5,6,7,8,9,10}; auto it=L.begin();"
        " std::advance(it, L.size()/2); return *it == 6 ? 0 : 1; }"
    ),
    "min_2_9_vaut_2_sans_la_directive": (
        "template<class T> constexpr T min(T a,T b){return a<b?a:b;}\n"
        'static_assert(min(2,9) == 2, "min(2,9) doit valoir 2");\nint main(){}'
    ),
    "templates_2_3_et_5_2_deduisent_double": (
        "#include <type_traits>\n"
        "template<class T> T t(T a,T b){return a<b?a:b;}\n"
        "static_assert(std::is_same_v<decltype(t(2.3,5.2)), double>);\n"
        "int main(){}"
    ),
    "pl15_static_int_i_compile": "struct P{ P(){ static int i=1; } };\nint main(){}",
}


@pytest.mark.parametrize("cas", sorted(REFUS))
def test_le_compilateur_refuse_l_exemple_du_courrier(cas: str) -> None:
    source, message = REFUS[cas]
    verdict = _compile(source)
    assert not verdict.accepte
    assert verdict.mentionne(message)


@pytest.mark.parametrize("cas", sorted(ACCEPTES))
def test_le_compilateur_accepte_la_version_du_courrier(cas: str) -> None:
    verdict = _compile(ACCEPTES[cas])
    assert verdict.accepte, verdict.messages


def _lancer(source: str) -> str:
    """Compile, lie et lance ; saute si l'edition de liens est impossible ici."""
    try:
        return COMPILATEUR.executer(source)
    except EditionDeLiensImpossible as erreur:
        pytest.skip(f"edition de liens impossible sur ce poste : {erreur}")


def test_le_compteur_de_la_planche_14_rend_1_2_1_puis_0() -> None:
    source = """#include <iostream>
class C{ public: static int K; C(){K++;} ~C(){K--;} };
int C::K=0;
void f(){ C a; std::cout<<C::K<<' '; { C b; std::cout<<C::K<<' ';}
  std::cout<<C::K<<' '; }
int main(){ f(); std::cout<<C::K<<"\\n"; }
"""
    assert _lancer(source) == "1 2 1 0\n"


def test_la_planche_15_affiche_dix_lignes_chaque_numero_deux_fois() -> None:
    source = """#include <iostream>
struct P{ int id; P(){ static int i=1; id=i++; Affiche(); }
  void Affiche(){ std::cout<<id<<"\\n"; } };
int main(){ P a[5]; for(auto&p:a) p.Affiche(); }
"""
    lignes = _lancer(source).split()
    assert len(lignes) == 10
    assert sorted(lignes) == ["1", "1", "2", "2", "3", "3", "4", "4", "5", "5"]


def test_min_2_9_vaut_2_a_l_execution_sans_la_directive() -> None:
    source = """#include <iostream>
template<class T> T min(T a,T b){return a<b?a:b;}
int main(){ std::cout<<min(2,9); }
"""
    assert _lancer(source) == "2"
