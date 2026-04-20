#!/usr/bin/env python
# coding: utf-8

# ### Analyse des accidents corporels de la circulation en France (2015–2020)
# 
# Ce notebook regroupe toutes les fonctions utilisées dans l'analyse.  
# Il doit être exécuté **avant** `main.ipynb` (ou importé via `%run fonctions.ipynb`).
# 
# **Source :** [Base de données BAAC – data.gouv.fr](https://www.data.gouv.fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2024)
# 
# ---
# **Sommaire des fonctions :**
# 1. Import des bibliothèques
# 2. Récupération des données (`recuperer_ressources`, `normaliser_colonnes`, `lire_csv_auto`, `charger_annee`, `charger_toutes_annees`)
# 3. Nettoyage (`nettoyer_dataset`)
# 4. Analyse descriptive (`calculer_stats_annuelles`)
# 5. Visualisations (`plot_*`, `generer_carte_chaleur`)
# 6. Modélisation (`preparer_donnees_modelisation`, `splitter_donnees`, `entrainer_random_forest`, `evaluer_modele`, `plot_importance_variables`, `comparer_modeles`)

# ## 1. Import des bibliothèques

# In[4]:


# Décommenter si nécessaire
# !pip install pandas numpy matplotlib seaborn folium scikit-learn plotly geopandas


# In[5]:


# Import des bibliothèques principales pour la data science
import pandas as pd              # Manipulation de données (DataFrame, nettoyage, transformation)
import numpy as np               # Calcul numérique (tableaux, fonctions mathématiques)

import matplotlib.pyplot as plt  # Visualisation de base (graphiques)
import matplotlib.ticker as mticker  # Formatage des axes (ex : pourcentages, grandes valeurs)
import seaborn as sns            # Visualisation avancée (plus esthétique que matplotlib)

import warnings                  # Gestion des avertissements (warnings)
import requests
import re

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, roc_auc_score,
                              ConfusionMatrixDisplay, f1_score)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# -------------------------------------------------------------------------
# Configuration de l'environnement
# -------------------------------------------------------------------------

warnings.filterwarnings('ignore')
# → Supprime les messages d'avertissement (utile pour un rendu propre,
#   mais à utiliser avec prudence en phase d'analyse)

pd.set_option('display.max_columns', 50)
# → Permet d'afficher jusqu'à 50 colonnes dans les DataFrame
#   (sinon pandas tronque l'affichage)

sns.set_theme(style='whitegrid', palette='muted')
# → Définit un thème graphique par défaut pour seaborn :
#   - fond avec grille (whitegrid)
#   - couleurs douces (muted)
#   → utile pour des graphiques lisibles dans un rapport

# Features et target utilisées pour la modélisation (partagées entre fonctions)
FEATURES = ['lum', 'agg', 'int', 'atm', 'col',           # Caractéristiques
            'catr', 'circ', 'nbv', 'surf', 'infra', 'situ',   # Lieux
            'catv', 'manv',                                # Véhicules
            'sexe', 'trajet', 'secu1',                     # Usagers
            'heure', 'week_end', 'annee']                  # Dérivées temporelles
TARGET = 'grave'

print('Bibliothèques importées avec succès.')


# ---
# ## 2. Fonctions — Récupération des données
# 
# Chaque année est composée de 4 fichiers CSV :
# - **caracteristiques** : date, heure, météo, luminosité, coordonnées GPS
# - **lieux** : type de route, état de la chaussée, vitesse limite
# - **vehicules** : type de véhicule, manœuvre
# - **usagers** : gravité, âge, sexe, équipement de sécurité

# In[6]:


def recuperer_ressources(dataset_id, annees_cibles):
    """
    Interroge l'API data.gouv.fr pour récupérer les URLs des fichiers CSV
    correspondant aux années cibles.

    Plutôt que de coder en dur des identifiants de fichiers susceptibles de
    changer, on interroge directement l'API officielle de data.gouv.fr.
    Le dictionnaire ALIASES gère les variantes orthographiques du nom
    'caracteristiques' (présentes selon les millésimes), ce qui rend le code
    robuste face aux incohérences de nommage du producteur.

    Paramètres
    ----------
    dataset_id : str
        Identifiant du dataset sur data.gouv.fr.
    annees_cibles : list[int]
        Liste des années à récupérer (ex. list(range(2015, 2021))).

    Retourne
    --------
    dict : {annee: {type_fichier: url}}
    """
    api_url = f"https://www.data.gouv.fr/api/1/datasets/{dataset_id}/"
    resp = requests.get(api_url, timeout=30)
    resp.raise_for_status()
    meta = resp.json()
    ressources_brutes = meta.get("resources", [])

    #  Mapping des variantes → nom standard
    aliases = {
        "caracteristiques": "caracteristiques",
        "caract":           "caracteristiques",
        "carcteristiques":  "caracteristiques",   # typo présente dans certains millésimes
        "caracteristique":  "caracteristiques",   # parfois sans 's'
        "lieux":            "lieux",
        "vehicules":        "vehicules",
        "usagers":          "usagers",
    }

    # Regex basée sur les variantes pour extraire type et année depuis le titre
    pattern = re.compile(
        r'(' + '|'.join(aliases.keys()) + r')[^0-9]*(20\d{2})',
        re.IGNORECASE
    )

    ressources = {}
    for r in ressources_brutes:
        if (r.get("format") or "").lower() != "csv":
            continue
        titre = (r.get("title") or r.get("description") or "").lower()
        m = pattern.search(titre)
        if not m:
            continue
        type_fichier = aliases[m.group(1).lower()]
        annee = int(m.group(2))
        if annee not in annees_cibles:
            continue
        url = r.get("url", "")
        if not url:
            continue
        ressources.setdefault(annee, {})[type_fichier] = url

    return ressources


# In[7]:


def normaliser_colonnes(df):
    """
    Normalise les noms de colonnes et la clé de jointure Num_Acc.

    La clé Num_Acc peut apparaître en minuscules ou majuscules selon les années.
    Cette fonction harmonise la casse pour permettre la jointure entre les
    4 tables de chaque année.
    """
    df.columns = [c.strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        if c.strip().lower() == "num_acc":
            rename[c] = "Num_Acc"
    return df.rename(columns=rename)


def lire_csv_auto(url):
    """
    Lit un CSV en essayant plusieurs séparateurs et encodages.

    Le format CSV a évolué au fil des années (virgule avant 2019,
    point-virgule après) — le code teste les deux et sélectionne celui
    qui produit le plus de colonnes.

    Retourne (pd.DataFrame, dict) ou lève RuntimeError.
    """
    essais = [
        {"sep": ";", "encoding": "latin-1"},
        {"sep": ",", "encoding": "latin-1"},
        {"sep": ";", "encoding": "utf-8"},
        {"sep": ",", "encoding": "utf-8"},
    ]
    derniere_erreur = None
    for params in essais:
        try:
            # Test rapide sur 5 lignes avant chargement complet
            tmp = pd.read_csv(url, sep=params["sep"], encoding=params["encoding"],
                              low_memory=False, nrows=5)
            if tmp.shape[1] <= 1:  # Si une seule colonne → mauvais séparateur
                continue
            df = pd.read_csv(url, sep=params["sep"], encoding=params["encoding"],
                             low_memory=False)
            return df, params
        except Exception as e:
            derniere_erreur = e
    raise RuntimeError(f"Lecture impossible pour {url} ({derniere_erreur})")


def charger_annee(annee, urls, types_fichiers=None):
    """
    Charge les 4 fichiers d'une année et les joint sur Num_Acc.

    La jointure inner sur Num_Acc produit une ligne par usager impliqué
    dans un accident, enrichie de toutes les informations de contexte.

    Paramètres
    ----------
    annee : int
    urls : dict  →  {type_fichier: url}
    types_fichiers : list[str], optionnel

    Retourne pd.DataFrame ou None si erreur.
    """
    if types_fichiers is None:
        types_fichiers = ["caracteristiques", "lieux", "vehicules", "usagers"]

    dfs = {}
    for type_f in types_fichiers:
        if type_f not in urls:
            print(f"   {type_f} absent pour {annee}, année ignorée.")
            return None
        try:
            df_tmp, params = lire_csv_auto(urls[type_f])
            df_tmp = normaliser_colonnes(df_tmp)
            if "Num_Acc" not in df_tmp.columns:
                print(f"  ✗ {type_f} ({annee}) : colonne 'Num_Acc' introuvable")
                return None
            dfs[type_f] = df_tmp
            print(f"  ✓ {type_f} ({annee}) : {len(df_tmp):,} lignes "
                  f"[sep='{params['sep']}', enc='{params['encoding']}']")
        except Exception as e:
            print(f"  ✗ Impossible de charger {type_f} ({annee}) : {e}")
            return None

    # Jointure des 4 tables sur Num_Acc
    df_merged = dfs["caracteristiques"].copy()
    for nom in ["lieux", "vehicules", "usagers"]:
        df_merged = df_merged.merge(dfs[nom], on="Num_Acc", how="inner",
                                    suffixes=("", f"_{nom}"))
    df_merged["annee"] = annee
    return df_merged


def charger_toutes_annees(ressources):
    """
    Charge et concatène les données pour toutes les années disponibles.

    Paramètre  : ressources (dict) — résultat de recuperer_ressources().
    Retourne   : pd.DataFrame consolidé.
    Lève       : ValueError si aucune année n'a pu être chargée.
    """
    print("=== Chargement des données 2015–2020 ===")
    liste_dfs = []
    for annee in sorted(ressources):
        print(f"\n→ Année {annee}")
        df_annee = charger_annee(annee, ressources[annee])
        if df_annee is not None:
            liste_dfs.append(df_annee)

    if not liste_dfs:
        raise ValueError("Aucune donnée chargée.")

    df = pd.concat(liste_dfs, ignore_index=True)
    print(f"\n✓ Dataset consolidé : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    print(f"  Années présentes : {sorted(df['annee'].unique().tolist())}")
    return df


# ---
# ## 3. Fonction — Nettoyage et préparation
# 
# **Référence des codes BAAC :**
# - `grav` : 1=Indemne, 2=Tué, 3=Blessé hospitalisé, 4=Blessé léger
# - `lum` : 1=Plein jour, 2=Crépuscule/aube, 3=Nuit sans éclairage, 4=Nuit éclairage éteint, 5=Nuit éclairage allumé
# - `atm` : 1=Normale, 2=Pluie légère, 3=Pluie forte, 4=Neige/grêle, 5=Brouillard, 6=Vent fort, 7=Éblouissant, 8=Couvert
# - `catr` : 1=Autoroute, 2=Route Nationale, 3=Route Départementale, 4=Voie Communale
# - `sexe` : 1=Masculin, 2=Féminin
# - `catv` : 01=Bicyclette, 02=Cyclomoteur, 07=VL, 10=VU, 33=Moto, 37=Bus

# In[8]:


def nettoyer_dataset(df):
    """
    Nettoyage et enrichissement du dataset BAAC.

    Opérations réalisées :
    1. Valeurs sentinelles (-1, 0) → NaN
       Dans la base BAAC, -1 et 0 encodent souvent l'absence d'information
       (ex. météo inconnue). On les remplace par NaN pour ne pas biaiser
       les analyses statistiques.
    2. Coordonnées GPS : correction de la virgule décimale française et
       filtrage des coordonnées hors métropole.
    3. Reconstruction de la date : la colonne 'an' encode l'année sur 2
       chiffres (15 pour 2015), ce qui nécessite un repadding avant conversion.
    4. Variables dérivées : heure, jour_semaine, week_end, trimestre,
       tranche_horaire.
    5. Âge et tranche d'âge.
    6. Variable cible 'grave' : binaire (1 = tué ou hospitalisé,
       0 = indemne ou blessé léger) pour la modélisation.
    """
    df = df.copy()

    # --- Valeurs sentinelles → NaN ---
    # Dans la base BAAC, -1 et 0 signifient souvent 'non renseigné'
    cols_sentinelles = ['lum', 'agg', 'int', 'atm', 'col', 'catr', 'circ',
                        'nbv', 'prof', 'plan', 'surf', 'infra', 'situ',
                        'sexe', 'trajet', 'secu1', 'secu2', 'secu3', 'catv']
    for col in cols_sentinelles:
        if col in df.columns:
            df[col] = df[col].replace([-1, 0], np.nan)

    # --- Coordonnées GPS ---
    for coord in ['lat', 'long']:
        if coord in df.columns:
            df[coord] = (
                df[coord].astype(str)
                .str.replace(',', '.', regex=False)  # virgule décimale française → point
                .pipe(pd.to_numeric, errors='coerce')
            )
    # Filtrer coordonnées aberrantes (France métropolitaine uniquement)
    if 'lat' in df.columns and 'long' in df.columns:
        mask_coords = df['lat'].between(41, 52) & df['long'].between(-5, 10)
        df.loc[~mask_coords, ['lat', 'long']] = np.nan

    # --- Date et heure ---
    if 'an' in df.columns and 'mois' in df.columns and 'jour' in df.columns:
        df['an']   = df['an'].astype(str).str.zfill(4)
        df['mois'] = df['mois'].astype(str).str.zfill(2)
        df['jour'] = df['jour'].astype(str).str.zfill(2)
        df['date'] = pd.to_datetime(
            df['an'] + '-' + df['mois'] + '-' + df['jour'], errors='coerce')

    if 'hrmn' in df.columns:
        df['hrmn']  = df['hrmn'].astype(str).str.zfill(4)
        df['heure'] = df['hrmn'].str[:2].pipe(pd.to_numeric, errors='coerce')

    # --- Variables dérivées utiles pour l'analyse temporelle ---
    if 'date' in df.columns:
        df['jour_semaine'] = df['date'].dt.dayofweek  # 0=Lundi, 6=Dimanche
        df['week_end']     = df['jour_semaine'].isin([5, 6]).astype(int)
        df['trimestre']    = df['date'].dt.quarter

    # --- Tranche horaire ---
    if 'heure' in df.columns:
        bins   = [-1, 6, 9, 12, 14, 18, 21, 24]
        labels = ['Nuit (0-6h)', 'Matin (6-9h)', 'Matin (9-12h)',
                  'Midi (12-14h)', 'Après-midi (14-18h)',
                  'Soirée (18-21h)', 'Nuit (21-24h)']
        df['tranche_horaire'] = pd.cut(df['heure'], bins=bins, labels=labels)

    # --- Âge usager ---
    if 'an_nais' in df.columns and 'an' in df.columns:
        df['age'] = df['an'].astype(float) - df['an_nais'].astype(float)
        df.loc[df['age'] < 0,   'age'] = np.nan   # âge négatif → aberrant
        df.loc[df['age'] > 110, 'age'] = np.nan   # âge > 110 → aberrant
        bins_age   = [0, 18, 25, 35, 45, 55, 65, 75, 200]
        labels_age = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75+']
        df['tranche_age'] = pd.cut(df['age'], bins=bins_age, labels=labels_age)

    # --- Labels lisibles pour la gravité ---
    if 'grav' in df.columns:
        df['grav'] = pd.to_numeric(df['grav'], errors='coerce')
        df['gravite_label'] = df['grav'].map({
            1: 'Indemne', 2: 'Tué',
            3: 'Blessé hospitalisé', 4: 'Blessé léger'
        })
        # Variable binaire : 1 = grave (tué ou hospitalisé), 0 = non grave
        df['grave'] = df['grav'].isin([2, 3]).astype(int)

    return df


# ---
# ## 4. Fonction — Analyse descriptive

# In[9]:


def calculer_stats_annuelles(df):
    """
    Calcule les statistiques agrégées par année.

    Retourne un DataFrame avec :
    annee, nb_accidents, nb_victimes, nb_tues,
    nb_blesses_hosp, nb_blesses_legers, taux_mortalite.
    """
    stats = (
        df.groupby('annee')
        .agg(
            nb_accidents      = ('Num_Acc', 'nunique'),
            nb_victimes       = ('grav', 'count'),
            nb_tues           = ('grav', lambda x: (x == 2).sum()),
            nb_blesses_hosp   = ('grav', lambda x: (x == 3).sum()),
            nb_blesses_legers = ('grav', lambda x: (x == 4).sum()),
        )
        .reset_index()
    )
    stats['taux_mortalite'] = (
        stats['nb_tues'] / stats['nb_victimes'] * 100
    ).round(2)
    return stats


# ---
# ## 5. Fonctions — Visualisations

# ---
# #### Evolution du nombre d'accidents et du taux de mortalité

# In[10]:


def plot_accidents_mortalite(stats_annuelles):
    """
    Graphique double axe : évolution du nombre d'accidents (barres bleues)
    et du taux de mortalité (courbe rouge) par année.

    Ce graphique met en évidence que la baisse des accidents en 2020 ne
    s'accompagne pas d'une baisse proportionnelle du taux de mortalité,
    suggérant un effet d'exposition (moins de trafic) plutôt qu'un changement
    de comportement.
    """
    fig, ax1 = plt.subplots(figsize=(11, 6))

    # Barres (accidents) — légèrement transparentes pour laisser voir la courbe
    ax1.bar(stats_annuelles['annee'], stats_annuelles['nb_accidents'], alpha=0.6)
    ax1.set_xlabel("Année", fontsize=12)
    ax1.set_ylabel("Nombre d'accidents", fontsize=12)
    ax1.set_title("Évolution des accidents et du taux de mortalité (2015-2020)",
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(stats_annuelles['annee'])
    ax1.grid(True, axis='y', alpha=0.3)

    # Axe secondaire → ligne rouge bien visible
    ax2 = ax1.twinx()
    ax2.plot(stats_annuelles['annee'], stats_annuelles['taux_mortalite'],
             color='red', marker='o', markersize=8, linewidth=2.5,
             linestyle='-', label="Taux de mortalité")
    ax2.set_ylabel("Taux de mortalité (%)", fontsize=12)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


# --- 
# #### Évolution du nombre de tués par année

# In[ ]:


def plot_accidents_et_tues(stats_annuelles):
    """
    Graphique double : nombre d'accidents (courbe) et nombre de tués (barres)
    par année, avec repère vertical COVID en 2020.

    La comparaison des deux graphiques montre que la diminution des accidents
    en 2020 s'accompagne d'une baisse des tués, mais pas dans les mêmes
    proportions — la gravité moyenne ne diminue pas fortement.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Graphique gauche : évolution du nombre d'accidents
    axes[0].plot(stats_annuelles['annee'], stats_annuelles['nb_accidents'],
                 marker='o', color='steelblue', linewidth=2)
    axes[0].set_title("Nombre d'accidents par année", fontsize=13)
    axes[0].set_xlabel('Année')
    axes[0].set_ylabel("Nombre d'accidents")
    axes[0].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    # Graphique droit : nombre de tués par année
    axes[1].bar(stats_annuelles['annee'], stats_annuelles['nb_tues'],
                color='salmon', edgecolor='white')
    axes[1].set_title('Nombre de tués par année', fontsize=13)
    axes[1].set_xlabel('Année')
    axes[1].set_ylabel('Nombre de tués')

    # Ligne pointillée 2020 : repère visuel pour la rupture COVID
    for ax in axes:
        if 2020 in stats_annuelles['annee'].values:
            ax.axvline(x=2020, color='gray', linestyle='--',
                       alpha=0.7, label='COVID-19')
            ax.legend()

    plt.suptitle("Évolution de l'accidentalité routière en France (2015–2020)",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


# ---
# #### Répartition de la gravité

# In[ ]:


def plot_repartition_gravite(df):
    """
    Graphique : répartition des victimes par gravité (barres horizontales).

    Le fort déséquilibre de classes (beaucoup d'indemnes, peu de tués) est un
    défi pour la modélisation → justifie class_weight='balanced' dans les
    modèles ML.
    """
    if 'gravite_label' not in df.columns:
        print("Colonne 'gravite_label' absente.")
        return

    counts = df['gravite_label'].value_counts()
    colors = ['#2ecc71', '#e74c3c', '#e67e22', '#3498db']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(counts.index, counts.values, color=colors)
    ax.bar_label(bars, fmt=lambda x: f'{x:,.0f}', padding=5)
    ax.set_title('Répartition des victimes par gravité (2015–2020)', fontsize=13)
    ax.set_xlabel('Nombre de victimes')
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    plt.tight_layout()
    plt.show()

    print("\nPourcentages :")
    print((counts / counts.sum() * 100).round(2).to_string())


# --- 
# #### Accidentalité par heure de la journée

# In[ ]:


def plot_accidents_par_heure(df):
    """
    Graphique : nombre d'accidents (barres) et nombre de tués (courbe)
    selon l'heure de la journée.

    Révèle le 'paradoxe nocturne' : la nuit concentre peu d'accidents en valeur
    absolue mais proportionnellement plus de décès (vitesses élevées, fatigue,
    alcool, routes moins fréquentées).
    Note : les deux axes ont des échelles différentes pour la lisibilité.
    """
    if 'heure' not in df.columns:
        print("Colonne 'heure' absente.")
        return

    accidents_heure = df.groupby('heure')['Num_Acc'].nunique()
    tues_heure      = df[df['grav'] == 2].groupby('heure').size()

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    ax1.bar(accidents_heure.index, accidents_heure.values,
            color='steelblue', alpha=0.6, label='Accidents')
    ax2.plot(tues_heure.index, tues_heure.values,
             color='crimson', marker='o', linewidth=2, label='Tués')

    ax1.set_xlabel('Heure de la journée')
    ax1.set_ylabel("Nombre d'accidents", color='steelblue')
    ax2.set_ylabel('Nombre de tués', color='crimson')
    ax1.set_xticks(range(0, 24))
    ax1.set_title("Accidents et décès selon l'heure de la journée", fontsize=13)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.tight_layout()
    plt.show()


# ---
# #### Gravité selon les conditions météo et la luminosité

# In[ ]:


def plot_gravite_meteo_luminosite(df):
    """
    Graphiques : taux d'accidents graves selon les conditions météo
    et selon la luminosité.

    Contre-intuitif : le brouillard et le vent fort sont plus dangereux
    que la pluie, qui incite les conducteurs à ralentir (effet protecteur).
    La nuit sans éclairage public est la condition la plus meurtrière.
    """
    labels_meteo = {1: 'Normale', 2: 'Pluie légère', 3: 'Pluie forte',
                    4: 'Neige/grêle', 5: 'Brouillard', 6: 'Vent fort',
                    7: 'Éblouissant', 8: 'Couvert'}
    labels_lum   = {1: 'Plein jour', 2: 'Crépuscule', 3: 'Nuit sans éclairage',
                    4: 'Nuit, éclairage éteint', 5: 'Nuit, éclairage allumé'}

    df = df.copy()

    if 'atm' in df.columns and 'grave' in df.columns:
        df['atm_label'] = df['atm'].map(labels_meteo)
        taux_meteo = (
            df.dropna(subset=['atm_label'])
            .groupby('atm_label')['grave']
            .mean() * 100
        ).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(9, 5))
        taux_meteo.plot(kind='barh', ax=ax, color='coral')
        ax.set_title("Taux d'accidents graves selon les conditions météo (%)", fontsize=13)
        ax.set_xlabel("% d'accidents graves (tués + hospitalisés)")
        ax.bar_label(ax.containers[0], fmt='%.1f%%', padding=3)
        plt.tight_layout()
        plt.show()

    if 'lum' in df.columns and 'grave' in df.columns:
        df['lum_label'] = df['lum'].map(labels_lum)
        taux_lum = (
            df.dropna(subset=['lum_label'])
            .groupby('lum_label')['grave']
            .mean() * 100
        ).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(9, 4))
        taux_lum.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_title("Taux d'accidents graves selon la luminosité (%)", fontsize=13)
        ax.set_xlabel("% d'accidents graves")
        ax.bar_label(ax.containers[0], fmt='%.1f%%', padding=3)
        plt.tight_layout()
        plt.show()


# --- 
# #### Profil des victimes : âge et sexe

# In[ ]:


def plot_tues_age_sexe(df):
    """
    Graphique : nombre de tués par tranche d'âge et sexe.

    Met en évidence la surreprésentation des hommes parmi les tués
    (rapport ~3 pour 1) et les deux groupes les plus touchés :
    18-34 ans et 75 ans et plus.
    """
    if 'tranche_age' not in df.columns or 'sexe' not in df.columns:
        print("Colonnes 'tranche_age' ou 'sexe' absentes.")
        return

    pivot = (
        df[df['grav'] == 2]   # Tués seulement
        .dropna(subset=['tranche_age', 'sexe'])
        .groupby(['tranche_age', 'sexe'])
        .size()
        .unstack()
        .rename(columns={1: 'Hommes', 2: 'Femmes'})
    )
    pivot.plot(kind='bar', figsize=(10, 5), color=['steelblue', 'salmon'],
               edgecolor='white', width=0.7)
    plt.title("Nombre de tués par tranche d'âge et sexe", fontsize=13)
    plt.xlabel("Tranche d'âge")
    plt.ylabel('Nombre de tués')
    plt.xticks(rotation=0)
    plt.legend(title='Sexe')
    plt.tight_layout()
    plt.show()


# In[ ]:


def generer_carte_chaleur(df, fichier_sortie='carte_accidents_graves.html'):
    """
    Génère une carte de chaleur (heatmap) des accidents graves et la sauvegarde
    en HTML.

    Concentrations attendues : Île-de-France, grandes métropoles, axes
    autoroutiers. L'échantillonnage à 5 000 points est nécessaire pour la
    performance du rendu interactif dans Jupyter.
    """
    try:
        import folium
        from folium.plugins import HeatMap

        df_carte = (
            df[(df['grave'] == 1) & df['lat'].notna() & df['long'].notna()]
            .sample(min(5000, len(df)), random_state=42)  # Limiter pour la performance
            [['lat', 'long']]
            .values.tolist()
        )
        carte = folium.Map(location=[46.5, 2.5], zoom_start=6,
                           tiles='CartoDB positron')
        HeatMap(df_carte, radius=8, blur=10, min_opacity=0.4).add_to(carte)
        carte.save(fichier_sortie)
        print(f"Carte sauvegardée : {fichier_sortie}")
        return carte

    except ImportError:
        print("Folium non installé. Installer avec : pip install folium")
    except Exception as e:
        print(f"Erreur carte : {e}")


# ---
# ## 6. Fonctions — Modélisation
# 
# **Justification du choix du Random Forest :**
# 1. Robustesse aux valeurs manquantes (via l'imputation dans le pipeline)
# 2. Pas de mise à l'échelle nécessaire (contrairement à la régression logistique)
# 3. Capture des interactions non-linéaires (ex. nuit + brouillard + chaussée mouillée)
# 4. Interprétabilité via les importances de variables (Gini)
# 5. Gestion du déséquilibre : `class_weight='balanced'` pondère automatiquement chaque classe
# 
# **Hyperparamètres choisis :**
# - `n_estimators=100` : 100 arbres, bon compromis vitesse/variance
# - `max_depth=10` : limite la profondeur pour éviter l'overfitting
# - `n_jobs=-1` : parallélisation sur tous les cœurs disponibles

# In[11]:


def preparer_donnees_modelisation(df, features=None, target=TARGET):
    """
    Prépare le DataFrame pour la modélisation.

    Les 19 variables retenues couvrent 4 dimensions :
    - Environnementales  : lum, atm
    - Infrastructurelles : catr, circ, nbv, surf
    - Comportementales   : manv, catv, secu1
    - Socio-démographiques + temporelles : sexe, trajet, heure, week_end, annee

    Les variables à plus de 60% de manquants (vma, secu2/3) sont exclues.
    """
    if features is None:
        features = FEATURES

    features_dispo = [f for f in features if f in df.columns]
    print(f"Features disponibles ({len(features_dispo)}) : {features_dispo}")

    df_model = df[features_dispo + [target]].copy()
    # Forcer le type numérique (certaines colonnes peuvent être object)
    for col in features_dispo:
        df_model[col] = pd.to_numeric(df_model[col], errors='coerce')
    df_model = df_model.dropna(subset=[target])

    print(f"Dataset pour modélisation : {len(df_model):,} lignes")
    print(f"Équilibre des classes :\n{df_model[target].value_counts(normalize=True).round(3)}")

    X = df_model[features_dispo]
    y = df_model[target]
    return X, y, features_dispo


def splitter_donnees(X, y, test_size=0.2, random_state=42):
    """
    Découpe les données en train/test avec stratification.

    On réserve 20% des données pour le test (bon comme ce qu'on faisait
    en Daveiga). Le paramètre stratify=y garantit que la proportion de cas
    graves/non graves est identique dans le train et le test, évitant un
    biais d'évaluation dû au déséquilibre des classes.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    print(f"Train : {len(X_train):,} exemples")
    print(f"Test  : {len(X_test):,} exemples")
    return X_train, X_test, y_train, y_test


def entrainer_random_forest(X_train, y_train):
    """
    Entraîne un pipeline Random Forest avec imputation des valeurs manquantes.

    Retourne un sklearn.Pipeline entraîné.
    """
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('model',   RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        class_weight='balanced',  # Gère le déséquilibre des classes
                        random_state=42,
                        n_jobs=-1))
    ])
    print("Entraînement du Random Forest...")
    pipeline.fit(X_train, y_train)
    print("Entraînement terminé.")
    return pipeline


def evaluer_modele(pipeline, X_test, y_test):
    """
    Évalue le modèle sur le jeu de test.

    Métriques produites :
    - AUC-ROC  : capacité à discriminer graves/non-graves (> 0.70 satisfaisant)
    - Precision : parmi les prédits graves, combien le sont vraiment ?
    - Recall   : parmi les vrais graves, combien sont détectés ?
      → Recall faible = accidents graves manqués = risque le plus coûteux

    Résultats attendus (selon l'IA) : AUC ≈ 0.77, recall grave ≈ 0.69
    mais precision faible (0.33) → beaucoup de faux positifs.
    Piste d'amélioration : ajuster le seuil → y_pred = (y_proba > 0.3).astype(int)

    Retourne dict : {'auc': float, 'y_pred': array, 'y_proba': array}
    """
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("=== Rapport de classification ===")
    print(classification_report(y_test, y_pred,
                                target_names=['Non grave', 'Grave']))
    auc = roc_auc_score(y_test, y_proba)
    print(f"AUC-ROC : {auc:.4f}")

    # Matrice de confusion
    # - Vrais négatifs  (haut gauche) : non graves correctement classés
    # - Faux positifs   (haut droite) : non graves prédits graves → sur-alerte
    # - Faux négatifs   (bas gauche)  : graves non détectés → erreur la plus préoccupante
    # - Vrais positifs  (bas droite)  : graves correctement identifiés
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=['Non grave', 'Grave'],
        cmap='Blues', ax=ax)
    ax.set_title('Matrice de confusion – Random Forest', fontsize=13)
    plt.tight_layout()
    plt.show()

    return {'auc': auc, 'y_pred': y_pred, 'y_proba': y_proba}


def plot_importance_variables(pipeline, features_dispo):
    """
    Graphique : importance des variables selon le Random Forest (Gini).

    Variables attendues en tête : catv, catr, heure, lum, atm.
    Ces résultats sont cohérents avec l'analyse descriptive et renforcent
    la validité du modèle.
    """
    rf = pipeline.named_steps['model']
    importances = pd.Series(
        rf.feature_importances_, index=features_dispo
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    importances.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title('Importance des variables – Random Forest', fontsize=13)
    ax.set_xlabel('Importance (Gini)')
    plt.tight_layout()
    plt.show()

    print("\nTop 5 variables les plus importantes :")
    print(importances.tail(5).sort_values(ascending=False).to_string())


def comparer_modeles(X_train, X_test, y_train, y_test):
    """
    Entraîne et compare trois modèles de classification.

    Modèles comparés :
    - Régression logistique : interprétable, rapide, mais linéaire
    - Random Forest         : robuste, interactions, Gini — modèle retenu
    - Gradient Boosting     : souvent le + performant, mais moins interprétable

    Si l'AUC du Gradient Boosting est supérieure, il pourrait être préféré
    en production, mais au détriment de l'interprétabilité.

    Retourne pd.DataFrame avec colonnes : Modèle, AUC-ROC, F1 (grave).
    """
    modeles = {
        'Régression logistique': LogisticRegression(
            class_weight='balanced', max_iter=500, random_state=42),
        'Random Forest':         RandomForestClassifier(
            n_estimators=100, max_depth=10, class_weight='balanced',
            random_state=42, n_jobs=-1),
        'Gradient Boosting':     GradientBoostingClassifier(
            n_estimators=100, max_depth=4, random_state=42),
    }

    # Imputation commune pour que tous les modèles partent des mêmes données
    imputer = SimpleImputer(strategy='most_frequent')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp  = imputer.transform(X_test)

    resultats = []
    for nom, modele in modeles.items():
        print(f"→ {nom}...", end=' ')
        modele.fit(X_train_imp, y_train)
        y_pred_m  = modele.predict(X_test_imp)
        y_proba_m = modele.predict_proba(X_test_imp)[:, 1]
        resultats.append({
            'Modèle':     nom,
            'AUC-ROC':    round(roc_auc_score(y_test, y_proba_m), 4),
            'F1 (grave)': round(f1_score(y_test, y_pred_m), 4),
        })
        print("OK")

    df_resultats = pd.DataFrame(resultats).sort_values('AUC-ROC', ascending=False)
    print("\n=== Comparaison des modèles ===")
    print(df_resultats.to_string(index=False))
    return df_resultats


# ---
# **Toutes les fonctions sont définies.** Nous pouvons maintenant exécuter le fichier `main.ipynb`.
