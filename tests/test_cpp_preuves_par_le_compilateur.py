"""Les valeurs du courrier 06, rejouees par le compilateur puis par le modele.

Chaque valeur que le courrier au professeur de programmation cite est
d'abord **affichee par un vrai programme** (port `Compilateur`, adaptateur
clang++), puis comparee a ce que `polytech.algorithmique` en dit : le modele
doit rendre, au caractere pres, l'affichage du compilateur. Sans compilateur
sur le poste, ces tests se suspendent, en le disant.

Sources : `ProgrammationAlgorithmique/SYLLABUS/
Syllabus_Introduction_a_la_programmation_2015.pdf` -- programmes 2.3-2.8
(pages 15 a 19), exemple du `goto` (page 35), tableau ASCII (page 10) et
programme 4.4 (page 45).
"""

from __future__ import annotations

import math
from typing import Final

import pytest

from polytech.algorithmique.domain.caracteres import caractere_de_code, code_ascii
from polytech.algorithmique.domain.cercle import disque_du_syllabus
from polytech.algorithmique.domain.chaines import NPOS, find
from polytech.algorithmique.domain.duree_de_vie import (
    CONSTRUCTION,
    DESTRUCTION,
    Execution,
)
from polytech.algorithmique.domain.expressions import evaluer
from polytech.algorithmique.domain.flux_d_entree import Clavier
from polytech.algorithmique.domain.flux_de_sortie import ecrire_un_reel
from polytech.algorithmique.domain.iterateurs import plage
from polytech.algorithmique.domain.patrons import minimum
from polytech.algorithmique.domain.planches_fautives import (
    doublon_de_la_planche_37,
    insertion_de_la_planche_62,
)
from polytech.algorithmique.domain.sequences import Sequence
from polytech.algorithmique.domain.types_cpp import diviser
from polytech.algorithmique.infrastructure.clang import (
    CompilateurClang,
    compilateur_disponible,
)

pytestmark = pytest.mark.skipif(
    not compilateur_disponible(), reason="aucun compilateur C++ sur le poste"
)
COMPILATEUR = CompilateurClang()

PI_IMPRIME: Final = 3.141596
"""Programmes 2.3 a 2.8 : la constante que le syllabus ecrit."""

DISQUE = """#include <iostream>
using namespace std;
int main() {{
  const double PI = {pi}; double rayon = {rayon};
  double temp = PI * rayon;
  cout << temp * rayon << ' ' << 2 * temp << endl;
}}
"""


def _disque_du_compilateur(pi: str, rayon: int) -> tuple[str, str]:
    surface, circonference = COMPILATEUR.executer(
        DISQUE.format(pi=pi, rayon=rayon)
    ).split()
    return surface, circonference


def test_npos_est_la_plus_grande_valeur_de_size_t() -> None:
    """Page 45 : `find` d'un motif absent rend `string::npos`."""
    source = (
        '#include <iostream>\n#include <string>\nint main(){ std::string s("ABCDEF");'
        ' std::cout << s.find("XYZ") << std::endl; }'
    )
    affiche = COMPILATEUR.executer(source).strip()
    assert affiche == str(NPOS) == "18446744073709551615"
    assert find("ABCDEF", "XYZ") == int(affiche)


@pytest.mark.parametrize("rayon", [2, 5])
def test_le_disque_du_syllabus_a_la_constante_imprimee(rayon: int) -> None:
    """3.141596 : 12.5664 au rayon 2 ; 78.5399 et 31.416 au rayon 5."""
    disque = disque_du_syllabus(PI_IMPRIME, rayon)
    assert (
        ecrire_un_reel(disque.surface),
        ecrire_un_reel(disque.circonference),
    ) == _disque_du_compilateur("3.141596", rayon)


def test_le_disque_au_vrai_pi_donne_78_5398_et_31_4159() -> None:
    """Avec pi a sept chiffres (3.141593) : la surface passe a 78.5398."""
    assert ecrire_un_reel(math.pi, 7) == "3.141593"
    disque = disque_du_syllabus(float(ecrire_un_reel(math.pi, 7)), 5)
    attendu = _disque_du_compilateur("3.141593", 5)
    assert attendu == ("78.5398", "31.4159")
    assert (
        ecrire_un_reel(disque.surface),
        ecrire_un_reel(disque.circonference),
    ) == attendu


@pytest.mark.parametrize("diviseur", [-1, -2])
def test_zero_divise_par_un_negatif_s_affiche_moins_zero(diviseur: int) -> None:
    """Page 35 : `(float)0 / -2` s'affiche `-0`, non `0.00`."""
    source = (
        f"#include <iostream>\nint main(){{ std::cout << (float)0 / {diviseur}"
        " << std::endl; }"
    )
    assert COMPILATEUR.executer(source).strip() == ecrire_un_reel(
        diviser(0.0, diviseur)
    )


def test_le_code_de_l_apostrophe_est_0x27_et_0x2c_est_la_virgule() -> None:
    """Page 10 : le tableau donne 0x2C pour l'apostrophe ; c'est la virgule."""
    source = (
        "#include <iostream>\nint main(){ std::cout << (int)'\\'' << ' '"
        " << (char)0x2C << std::endl; }"
    )
    code, caractere = COMPILATEUR.executer(source).split()
    assert int(code) == code_ascii("'") == 0x27
    assert caractere == caractere_de_code(0x2C) == ","


def _lancer(corps: str, entetes: str = "") -> str:
    """Un `main` autour de `corps`, avec `iostream` et les `entetes` demandes."""
    source = (
        f"#include <iostream>\n{entetes}using namespace std;\nint main(){{ {corps} }}\n"
    )
    return COMPILATEUR.executer(source)


def test_la_boucle_de_la_planche_62_recopie_une_case_de_trop() -> None:
    """Planche 62 : `N = 6`, `P = 2` ; la case 7 recoit `Tab[6]`."""
    affiche = _lancer(
        "int Tab[11]={1,9,5,2,3,7,-1,-1,-1,-1,-1}; int N=6, P=2;"
        " for(int i=N;i>P;i--) Tab[i+1]=Tab[i];"
        " for(int i=0;i<8;i++) cout<<Tab[i]<<' ';"
    )
    modele = insertion_de_la_planche_62([1, 9, 5, 2, 3, 7, -1, -1, -1, -1, -1], 6, 2)
    assert affiche.split() == [str(c) for c in modele[:8]]
    assert affiche.split() == ["1", "9", "5", "2", "2", "3", "7", "-1"]


def test_la_fonction_doublon_de_la_planche_37_en_laisse_passer() -> None:
    """Planche 37 sur `Ax Ax Ax By By Ax` : `Ax Ax By`, non `Ax By`."""
    affiche = _lancer(
        'vector<string> t={"Ax","Ax","Ax","By","By","Ax"}; int n=6;'
        " for(int i=0;i<n;i++) for(int j=i+1;j<n;j++) if(t[i]==t[j]){"
        " for(int k=j;k<n-1;k++) t[k]=t[k+1]; n--; }"
        " for(int i=0;i<n;i++) cout<<t[i]<<' ';",
        "#include <vector>\n#include <string>\n",
    )
    modele = doublon_de_la_planche_37(["Ax", "Ax", "Ax", "By", "By", "Ax"])
    assert affiche.split() == modele == ["Ax", "Ax", "By"]


def test_l_objet_local_meurt_avant_l_objet_statique() -> None:
    """Seance 4, planche 60 : construction V, S ; destruction S ; B ; V puis B."""
    affiche = _lancer(
        'struct T{ const char*n; T(const char*x):n(x){cout<<"C "<<n<<"\\n";}'
        ' ~T(){cout<<"D "<<n<<"\\n";} };'
        ' T v("V"); { T s("S"); } static T b("B");'
    )
    execution = Execution()
    execution.ouvrir()
    execution.locale("V")
    execution.ouvrir()
    execution.locale("S")
    execution.fermer()
    execution.statique("B")
    execution.terminer()
    lettre = {CONSTRUCTION: "C", DESTRUCTION: "D"}
    modele = [f"{lettre[genre]} {nom}" for genre, nom in execution.journal]
    assert affiche.splitlines() == modele
    assert modele[-2:] == ["D V", "D B"]


def test_insert_place_les_trois_b_devant_l_element_de_rang_2() -> None:
    """Tableau « vector » : `insert(2, 'B', 3)` sur `a b d d e`."""
    affiche = _lancer(
        "vector<char> v={'a','b','d','d','e'}; v.insert(v.begin()+2,3,'B');"
        " for(char c : v) cout<<c;",
        "#include <vector>\n",
    )
    vecteur = Sequence("vector", "abdde")
    vecteur.insert(2, "B", 3)
    assert affiche.strip() == "".join(vecteur.elements) == "abBBBdde"


def test_le_parcours_d_une_chaine_garde_le_blanc_final() -> None:
    """Planche 60 : `Un message ` sort `U,n, ,m,e,s,s,a,g,e, ,`."""
    affiche = _lancer(
        "string s=\"Un message \"; for(size_t i=0;i<s.size();i++) cout<<s[i]<<',';",
        "#include <string>\n",
    )
    assert affiche == "".join(f"{c}," for c in plage("Un message ", 0, 11))


def test_l_operateur_virgule_rend_son_operande_de_droite() -> None:
    """Syllabus de C, p. 16 : `5, 7` vaut 7 et `("%d", n)` vaut `n`."""
    affiche = _lancer("int n=42; cout << (5, 7) << ' ' << (\"%d\", n);")
    assert affiche.split() == [str(evaluer("5, 7")), "42"]


def test_cin_ne_lit_que_le_premier_mot() -> None:
    """Exercice 6.5 : `cin >> Tab` s'arrete au premier blanc."""
    affiche = _lancer(
        'istringstream in("Les sanglots longs\\n"); string t; in >> t; cout << t;',
        "#include <sstream>\n#include <string>\n",
    )
    assert affiche == Clavier("Les sanglots longs\n").mot() == "Les"


def test_min_de_2_et_9_vaut_2_sans_la_directive() -> None:
    """Seance templates : le patron de la planche rend 2 pour `(2, 9)`."""
    source = (
        "#include <iostream>\ntemplate<class T> T min(T a,T b){return a<b?a:b;}\n"
        "int main(){ std::cout << min(2,9) << ' ' << min(2.3,5.2); }\n"
    )
    assert COMPILATEUR.executer(source).split() == [
        str(minimum(2, 9)),
        str(minimum(2.3, 5.2)),
    ]
