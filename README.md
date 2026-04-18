# Projet-python-pour-la-data-science
Projet réalisé dans le cadre du cours de python pour la data science en 2A ENSAI
# Analyse des accidents corporels de la circulation en France (2015–2024)

## Problématique

 **Quels facteurs (environnementaux, comportementaux, infrastructurels) influencent la gravité des accidents de la route en France, et comment ont-ils évolué entre 2015 et 2024 ?**


## Structure du projet

```
.
├── accidents_routiers.ipynb   # Notebook principal (analyse complète)
├── README.md                  # Ce fichier
├── fonctions ?  (il va falloir faire ça )
└── requirements.txt           # Dépendances Python
```

## Source des données

**Base de données BAAC (Bulletin d'Analyse des Accidents Corporels)**
- Producteur : Ministère de l'Intérieur / ONISR
- Licence : Licence Ouverte / Open Licence
- Lien : https://www.data.gouv.fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2024

- Période couverte : 2015 à 2024 (2021–2022 exclus pour incompatibilité de format)
- Volume : ~2 millions de lignes (usagers impliqués dans des accidents)

Les données sont téléchargées **automatiquement via l'API data.gouv.fr** au lancement du notebook — aucun fichier à télécharger manuellement.

Chaque année est composée de 4 fichiers CSV joints sur l'identifiant `Num_Acc` :

| Fichier | Contenu |
|---------|---------|
| `caracteristiques` | Date, heure, météo, luminosité, localisation GPS |
| `lieux` | Type de route, état de la chaussée, nombre de voies |
| `vehicules` | Type de véhicule, manœuvre effectuée |
| `usagers` | Gravité, âge, sexe, équipement de sécurité |

---

##  Contenu du notebook

### 1. Récupération des données
Interrogation de l'API data.gouv.fr pour découverte automatique des fichiers, jointure des 4 tables par année, détection automatique du séparateur CSV (`,` ou `;` selon les millésimes).

### 2. Nettoyage
- Remplacement des valeurs sentinelles (-1, 0) par `NaN`
- Correction des coordonnées GPS (format décimal français)
- Reconstruction des dates et heures
- Création de variables dérivées : `heure`, `week_end`, `tranche_age`, `grave` (variable cible binaire)

### 3. Analyse descriptive
- Évolution temporelle du nombre d'accidents et de tués (2015–2024)
- Impact de la météo et de la luminosité sur la gravité
- Profil des victimes par âge et sexe
- Accidentalité par heure de la journée

### 4. Visualisation
- Graphiques temporels avec marqueur COVID-2020
- Carte de chaleur interactive (Folium) des accidents graves
- Histogrammes comparatifs par catégories

### 5. Modélisation
Prédiction de la **gravité d'un accident** (grave = tué ou hospitalisé) avec 3 modèles comparés :
- Régression logistique
- **Random Forest** ← modèle retenu
- Gradient Boosting

Évaluation : AUC-ROC, F1-score, matrice de confusion, importance des variables.

---

## 🚀 Lancer le notebook

### Prérequis
- Python 3.9+
- Jupyter Notebook ou JupyterLab

### Installation

```bash
# Cloner le dépôt
git clone https://github.com/<votre-compte>/<votre-repo>.git
cd <votre-repo>

# Installer les dépendances
pip install -r requirements.txt

# Lancer Jupyter
jupyter notebook accidents_routiers.ipynb
```

### ⚠️ Important
Le notebook télécharge les données depuis internet au premier lancement (~500 Mo au total). Assurez-vous d'avoir une connexion stable. Le téléchargement peut prendre 5 à 15 minutes selon votre débit.

---

## 🧰 Stack technique comme étudié dans ce cours et des connaissances antérieurs

| Outil | Usage |
|-------|-------|
| `pandas` | Manipulation des données tabulaires |
| `numpy` | Calculs numériques |
| `matplotlib` / `seaborn` | Visualisations statiques |
| `folium` | Carte interactive |
| `scikit-learn` | Modélisation ML |
| `requests` | Appel à l'API data.gouv.fr |

---

## 📋 Principaux résultats

- **Tendance baissière** du nombre de tués entre 2015 et 2024, avec une rupture COVID en 2020.
- La **nuit sans éclairage** est la condition environnementale la plus meurtrière.
- Le **brouillard** est plus dangereux que la pluie (les conducteurs ralentissent sous la pluie mais pas sous le brouillard).
- Les **hommes de 18-24 ans** et les **75 ans et plus** sont les profils les plus vulnérables.
- Le **type de véhicule** (moto, vélo) est le meilleur prédicteur individuel de la gravité.
- Le Random Forest obtient un AUC-ROC satisfaisant, confirmant que les données contiennent une information prédictive réelle.

---

## 👥 Auteurs

Projet réalisé par DANSOU Victoire, SOUMAILA Nadia, TOE Dieudonn — ENSAI, 2025-2026.

---
