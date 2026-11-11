# Polytech

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Typing](https://img.shields.io/badge/Typing-Mypy%20Strict-brightgreen.svg)](https://mypy.readthedocs.io/)
[![Architecture](https://img.shields.io/badge/Architecture-Domain--Driven%20Design-orange.svg)](#architecture--principes-de-conception)
[![Engine](https://img.shields.io/badge/Engine-SymPy%20%7C%20NumPy%20%7C%20SciPy-informational.svg)](#)
[![Proof](https://img.shields.io/badge/Proofs-Lean%204%20%7C%20Rust-purple.svg)](#composants-natifs-et-preuves-formelles)

**Polytech** est une bibliothèque scientifique et d'ingénierie de haute précision, combinant **calcul formel symbolique**, **résolution numérique exacte** et **dérivation pas-à-pas** des démarches de sciences appliquées.

Conçue selon les principes du **Domain-Driven Design (DDD)**, elle découpe le cursus des sciences de l'ingénieur en contextes bornés (*Bounded Contexts*) étanches, chacun utilisant son langage ubiquitaire propre.

---

## 🔬 Philosophie : Les étapes, pas seulement le résultat

Contrairement aux bibliothèques de calcul standard qui renvoient un résultat brut ou une approximation numérique immédiate :
1. **Arithmétique pure et traçable** : aucune constante magique codée en dur. Chaque valeur découle des équations fondamentales, de constantes physiques unifiées ou d'adimensionnements validés.
2. **Modélisation par étapes** : la bibliothèque expose la démarche de résolution (changement de variable, décomposition en éléments simples, pivot du simplexe, dérivation particulaire, bilans enthalpiques).
3. **Double certification** : vérification par suites de tests automatisées (> 2 500 tests unitaires) et certification de théorèmes physiques par des preuves formelles en **Lean 4**.

---

## 📚 Cartographie des 11 Domaines Scientifiques

La bibliothèque est structurée en 11 modules de domaine accessibles via le paquet racine :

| Domaine | Contexte | Principales notions & capacités |
| :--- | :--- | :--- |
| [`polytech.fluides`](file:///Users/baptistegrosjean/Source/polytech/polytech/fluides) | Mécanique des fluides | Statique des fluides, hydrostatique, cinématique (trajectoires, lignes d'émission/courant), dérivation particulaire, équations de Navier-Stokes, théorème de Bernoulli généralisé, pertes de charge régulières (Darcy-Weisbach, Blasius) et singulières, couche limite, turbulence (loi logarithmique de paroi, Boussinesq, Prandtl), traînée et portance. |
| [`polytech.maths`](file:///Users/baptistegrosjean/Source/polytech/polytech/maths) | Mathématiques de l'ingénieur | Algèbre linéaire exacte, espaces vectoriels, réduction matricielle (Jordan, diagonalisation), analyse complexe et théorème des résidus, intégration méthodique par étapes, EDO de tout ordre, EDP, séries de Fourier et transformée de Laplace. |
| [`polytech.thermique`](file:///Users/baptistegrosjean/Source/polytech/polytech/thermique) | Thermique & Transferts | Conduction stationnaire et transitoire (Fourier 1D/2D/cylindrique, surfaces ailettées), convection forcée et naturelle (corrélations de Nusselt, similitude), rayonnement thermique (corps noir de Planck, zones grises, facteurs de forme) et échangeurs de chaleur (méthodes DTLM et NUT). |
| [`polytech.electronique`](file:///Users/baptistegrosjean/Source/polytech/polytech/electronique) | Électronique & Instrumentation | Amplificateurs opérationnels réels et idéaux, circuits linéaires et non-linéaires, synthèse de filtres actifs (Sallen-Key, Rauch, biquad, Butterworth, Chebyshev), quadripôles (matrices ABCD, impédances), bruit thermique de Johnson-Nyquist et chaîne d'acquisition. |
| [`polytech.electromagnetisme`](file:///Users/baptistegrosjean/Source/polytech/polytech/electromagnetisme) | Électromagnétisme & Circuits | Équations de Maxwell sous formes locale et intégrale, électrostatique (théorème de Gauss, potentiels), magnétostatique (loi de Biot et Savart, théorème d'Ampère, matériaux magnétiques), induction de Faraday, théorèmes de Thévenin/Norton et régimes transitoires RLC. |
| [`polytech.signaux`](file:///Users/baptistegrosjean/Source/polytech/polytech/signaux) | Signaux, Systèmes & Automatique | Analyse temporelle et fréquentielle des systèmes LTI, fonctions de transfert en boucle ouverte/fermée, diagrammes de Bode et Nyquist, marges de phase et de gain, critère de Routh-Hurwitz, correcteurs PID et méthode de Ziegler-Nichols. |
| [`polytech.optimisation`](file:///Users/baptistegrosjean/Source/polytech/polytech/optimisation) | Recherche opérationnelle & Aide à la décision | **Programmation mathématique** : Simplexe primal 2-phases, dual simplexe, dualité forte/faible, sensibilité, Goal Programming, Branch & Bound, coupures de Gomory, transport (VAM, MODI) et affectation (Hongrois).<br>**Suite ORSolver intégrée** : Réseaux & graphes (Dijkstra, Bellman-Ford, Floyd-Warshall, A*, Flot Max Edmonds-Karp, Min Cost Flow, Kruskal, Prim, PERT/CPM), Théorie des jeux (Minimax von Neumann, Alpha-Beta), Processus de Markov (MDP), MCDA (AHP de Saaty, TOPSIS), Décision dans l'incertain (Wald, Savage, Hurwicz, Laplace, EVPI), Files d'attente (M/M/1, M/M/s Erlang-C), Stocks (Wilson EOQ), Fiabilité (Weibull, MTBF), Métaheuristiques TSP (2-Opt, recuit simulé), Simulation Monte Carlo et parser CPLEX `.lp`. |
| [`polytech.algorithmique`](file:///Users/baptistegrosjean/Source/polytech/polytech/algorithmique) | Algorithmique & Systèmes C++ | Oracles de conformité algorithmique, complexité spatio-temporelle, structures de données (listes, arbres, graphes, tables de hachage), patrons de conception STL et vérification formelle des comportements d'exécution. |
| [`polytech.chemistry`](file:///Users/baptistegrosjean/Source/polytech/polytech/chemistry) | Chimie appliquée | Gaz parfaits et gaz réels de Van der Waals, stœchiométrie, équilibres chimiques en phase aqueuse, thermodynamique des réactions, potentiel électrochimique (Nernst, piles) et désintégration radioactive. |
| [`polytech.mecanique`](file:///Users/baptistegrosjean/Source/polytech/polytech/mecanique) | Mécanique rationnelle | Dynamique newtonienne, champs de pesanteur local et champ de gravitation universelle de Newton. |
| [`polytech.raisonnement`](file:///Users/baptistegrosjean/Source/polytech/polytech/raisonnement) | Raisonnement symbolique | Modélisation et audit formel des chaînes d'inférence déductive et des étapes intermédiaires de calcul. |

---

## 🚀 Guide de Démarrage Rapide (Quickstart)

### 1. Mécanique des fluides : Écoulement en conduite & nombre de Reynolds

```python
from polytech.fluides.domain.regime_de_reynolds import (
    nombre_de_reynolds,
    REYNOLDS_LAMINAIRE,
    REYNOLDS_TURBULENT,
)
from polytech.fluides.domain.conduite import (
    coefficient_de_blasius,
    perte_de_charge_reguliere,
)

# Caractéristiques d'un écoulement d'eau dans une conduite lisse
vitesse = 1.8         # m/s
diametre = 0.05       # 50 mm
viscosite_cinematique = 1e-6  # m²/s (eau à 20°C)
longueur = 100.0      # 100 m

# Calcul du nombre de Reynolds
re = nombre_de_reynolds(vitesse, diametre, viscosite_cinematique)
print(f"Reynolds: {re:.0f}")  # 90 000 -> Régime turbulent

# Facteur de frottement selon Blasius pour régime turbulent lisse
lambda_blasius = coefficient_de_blasius(re)
print(f"Coefficient de frottement de Blasius: {lambda_blasius:.5f}")

# Perte de charge linéaire régulière
delta_h = perte_de_charge_reguliere(lambda_blasius, longueur, diametre, vitesse)
print(f"Perte de charge: {delta_h:.2f} m de colonne de fluide")
```

---

### 2. Mathématiques : Résolution d'équations différentielles pas à pas

```python
import sympy as sp
from polytech.maths.domain.differentielles.second_ordre import resoudre_second_ordre_constant

t = sp.Symbol("t", real=True)
y = sp.Function("y")

# Résolution analytique de l'oscillateur amorti : y'' + 2*zeta*omega_0*y' + omega_0^2*y = 0
omega_0 = 10
zeta = sp.Rational(1, 2)  # Amortissement sous-critique

solution = resoudre_second_ordre_constant(
    a=1,
    b=2 * zeta * omega_0,
    c=omega_0**2,
    second_membre=0,
    variable=t,
)
print("Solution générale de l'EDO :", solution.forme_symbolique)
```

---

### 3. Thermique : Échangeur de chaleur à contre-courant (DTLM & NUT)

```python
from polytech.thermique.domain.echangeurs import (
    difference_de_temperature_logarithmique,
    efficacite_contre_courant,
    puissance_echangee,
)

# Températures d'entrée et de sortie des fluides chaud et froid (°C)
t_chaud_in, t_chaud_out = 90.0, 50.0
t_froid_in, t_froid_out = 20.0, 45.0

# Écart logarithmique moyen (DTLM)
dtlm = difference_de_temperature_logarithmique(
    delta_t_1=t_chaud_in - t_froid_out,
    delta_t_2=t_chaud_out - t_froid_in,
)
print(f"DTLM: {dtlm:.2f} °C")
```

---

### 4. Traitement du signal : Stabilité d'une boucle fermée (Routh-Hurwitz)

```python
import sympy as sp
from polytech.signaux.domain.stabilite import tableau_de_routh, est_strictement_stable

p = sp.Symbol("p")

# Dénominateur de la fonction de transfert en boucle fermée
# D(p) = p^3 + 3 p^2 + 3 p + 1 + K
K = 5
polynome_caracteristique = [1, 3, 3, 1 + K]

tableau = tableau_de_routh(polynome_caracteristique)
stable = est_strictement_stable(tableau)
print(f"Le système asservi est-il stable pour K={K} ? {'Oui' if stable else 'Non'}")
```

---

## 🏛 Architecture & Principes de Conception

### 1. Bounded Contexts étanches (DDD)
Chaque sous-système mathématique ou physique possède sa propre ontologie :
- Un `champ` en cinématique des fluides ($\vec{V}(x, y, z, t)$) n'est pas le même objet conceptuel qu'un `champ` électrostatique ($\vec{E}$).
- Aucun domaine n'importe la logique interne d'un autre sans frontière explicite.

### 2. Composants natifs et Preuves formelles
Le projet combine trois niveaux de vérification :
- **Python pur (`polytech/`)** : Couche d'abstraction principale, calcul symbolique SymPy et interfaces haut niveau.
- **Rust (`polytech-core/`)** : Moteur natif en Rust relié par PyO3 pour les calculs physiques intensifs.
- **Lean 4 (`preuves/`)** : Formalisation mathématique certifiée par machine des théorèmes clés et points d'ancrage.

---

## ⚙️ Installation & Configuration

### Prérequis
- **Python >= 3.12**
- Gestionnaire de paquets moderne recommandé : [`uv`](https://github.com/astral-sh/uv) ou `pip`

### Installation locale en mode développement

```bash
# Avec uv (ultra-rapide)
uv venv
source .venv/bin/activate
uv pip install -e .

# Ou avec pip standard
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## 🧪 Tests, Typage & Qualité de Code

La bibliothèque applique une discipline de vérification continue stricte :

```bash
# Lancer l'intégralité de la suite de tests unitaires
pytest

# Vérification du typage statique strict (mode strict sans compromis)
mypy --strict polytech

# Formatage et analyse statique du code (linter)
ruff check polytech tests
```

---

## 📄 Licence

Ce projet est distribué sous licence libre. Consultez le fichier `LICENSE` pour plus de détails.
