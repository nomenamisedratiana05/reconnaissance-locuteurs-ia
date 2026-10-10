# 🎙️ Reconnaissance des locuteurs par Intelligence Artificielle

## 📌 Présentation du projet

Ce projet consiste à développer un système de reconnaissance automatique des locuteurs à partir d'enregistrements vocaux, en utilisant des techniques d'intelligence artificielle et de Machine Learning.

Le système analyse les caractéristiques acoustiques d'un fichier audio afin de déterminer à quel locuteur il correspond parmi les classes connues lors de l'entraînement.

Le modèle distingue actuellement trois classes :

- **moi** : voix du développeur du projet ;
- **ami** : voix d'un ami ;
- **autres** : catégorie regroupant les autres voix utilisées dans le jeu de données.

Le projet comprend un processus d'entraînement du modèle et un programme permettant de prédire le locuteur à partir d'un nouvel enregistrement audio.

## 🎯 Objectifs

- Étudier les caractéristiques acoustiques de la voix.
- Extraire automatiquement des caractéristiques pertinentes à partir des fichiers audio.
- Entraîner des réseaux de neurones pour reconnaître les locuteurs.
- Évaluer les performances du modèle sur des données de test.
- Effectuer des prédictions sur de nouveaux fichiers audio.
- Afficher les scores de confiance associés aux classes prédites.

## ⚙️ Technologies utilisées

| Technologie | Utilisation |
|---|---|
| Python | Langage principal du projet |
| NumPy | Calculs numériques et manipulation des tableaux |
| Librosa | Chargement et analyse des fichiers audio |
| Scikit-learn | Standardisation des caractéristiques et entraînement des réseaux de neurones |
| Joblib | Sauvegarde et chargement du modèle entraîné |
| MFCC | Extraction de caractéristiques spectrales de la voix |
| MLPClassifier | Modèle de réseau de neurones utilisé pour la classification |

## 🧠 Fonctionnement du système

Le fonctionnement du projet est organisé en plusieurs étapes.

### 1. Collecte des données audio

Les enregistrements vocaux sont organisés en dossiers selon les classes de locuteurs.

Les données sont séparées en deux ensembles :

- **Entraînement** : données utilisées pour apprendre les caractéristiques des différentes classes.
- **Test** : données utilisées pour évaluer le modèle après l'entraînement.

### 2. Prétraitement et extraction des caractéristiques

Les fichiers audio sont analysés afin d'en extraire plusieurs caractéristiques acoustiques.

Le système utilise notamment :

- **MFCC (Mel-Frequency Cepstral Coefficients)** : caractéristiques représentant la structure spectrale du signal vocal ;
- **Delta MFCC** : informations sur les variations temporelles des MFCC ;
- **Contraste spectral** : informations sur la répartition de l'énergie dans différentes bandes de fréquences ;
- **Caractéristiques liées à la hauteur de la voix (pitch)**.

Ces caractéristiques sont transformées en représentations numériques exploitables par le modèle.

### 3. Entraînement du modèle

Les caractéristiques extraites sont standardisées avant d'être utilisées pour entraîner les réseaux de neurones.

Le projet utilise des classificateurs `MLPClassifier` de Scikit-learn. Plusieurs réseaux de neurones sont entraînés et combinés dans le système de classification.

Le modèle entraîné est ensuite sauvegardé afin de pouvoir être réutilisé sans devoir recommencer l'entraînement à chaque prédiction.

### 4. Reconnaissance d'un locuteur

Lorsqu'un nouvel enregistrement audio est fourni au programme :

1. Le fichier audio est chargé.
2. Ses caractéristiques acoustiques sont extraites.
3. Les caractéristiques sont préparées selon le traitement utilisé pendant l'entraînement.
4. Le modèle analyse les caractéristiques.
5. Le système affiche la classe prédite ainsi que les scores associés aux classes.
6. Un avertissement est affiché lorsque le score maximal est inférieur au seuil de confiance défini dans le programme.

## 📁 Structure du projet

```text
reconnaissance-locuteurs-ia/
│
├── donnees/
│   ├── entrainement/
│   │   ├── moi/
│   │   ├── ami/
│   │   └── autres/
│   │
│   └── test/
│       ├── moi/
│       ├── ami/
│       └── autres/
│
├── models/
│   └── modele_voix.joblib
│
├── scripts/
│   ├── entrainement.py
│   └── prediction.py
│
├── .gitignore
└── README.md
```

**Rôle des principaux fichiers :**

- `donnees/entrainement/` : contient les enregistrements utilisés pour l'entraînement.
- `donnees/test/` : contient les enregistrements utilisés pour les tests.
- `models/modele_voix.joblib` : contient le modèle entraîné sauvegardé.
- `scripts/entrainement.py` : extrait les caractéristiques et entraîne le modèle.
- `scripts/prediction.py` : charge le modèle et prédit le locuteur d'un fichier audio.
- `.gitignore` : empêche l'ajout de certains fichiers temporaires, notamment les caches Python.

## 📋 Prérequis

Pour exécuter le projet, il faut disposer de :

- Python 3 ;
- pip ;
- les bibliothèques Python nécessaires ;
- les fichiers audio du projet ;
- le modèle entraîné, si l'on souhaite effectuer directement des prédictions.

## 🚀 Installation

### 1. Cloner le dépôt GitHub

```bash
git clone https://github.com/nomenamisedratiana05/reconnaissance-locuteurs-ia.git
```

### 2. Accéder au dossier du projet

```bash
cd reconnaissance-locuteurs-ia
```

### 3. Créer un environnement virtuel

Sous Linux :

```bash
python3 -m venv venv
```

Activer l'environnement virtuel :

```bash
source venv/bin/activate
```

### 4. Installer les dépendances

Installer les bibliothèques utilisées par le projet :

```bash
pip install numpy librosa scikit-learn joblib
```

> Remarque : cette commande installe les bibliothèques principales. Si le projet nécessite des versions précises, il est préférable de les définir dans un fichier `requirements.txt`.

## ▶️ Utilisation

### Option 1 : utiliser le modèle déjà entraîné

Le dépôt contient le modèle sauvegardé dans :

```text
models/modele_voix.joblib
```

Pour tester la reconnaissance d'un fichier audio, exécuter :

```bash
python scripts/prediction.py "donnees/test/ami/Recording_75.wav"
```

Remplacer `nom_du_fichier_audio.mp3` par le nom réel d'un fichier audio présent dans le dossier indiqué.

Par exemple, pour connaître les fichiers disponibles :

```bash
find donnees/test -type f
```

Choisir ensuite un chemin existant et le fournir au programme.

Le programme affiche notamment :

- le nom du fichier analysé ;
- le nombre de segments analysés ;
- la classe prédite ;
- les scores associés aux classes ;
- un avertissement si le score maximal est inférieur au seuil de confiance défini.

**Important :** un score de confiance élevé ne garantit pas que la prédiction est correcte.

### Option 2 : entraîner le modèle

Pour réentraîner le système à partir des données disponibles :

```bash
python scripts/entrainement.py
```

Le programme utilise les fichiers organisés dans `donnees/entrainement/` et `donnees/test/`, selon l'implémentation du script.

Le modèle entraîné est sauvegardé dans :

```text
models/modele_voix.joblib
```

> Attention : l'exécution de l'entraînement peut remplacer le modèle précédemment sauvegardé.

## 📊 Résultats

Lors d'une évaluation précédente sur le jeu de données de test, le système a obtenu une précision de **96,30 %**.

Cette valeur correspond au résultat observé lors de cette évaluation. Elle ne garantit pas les mêmes performances sur d'autres enregistrements, d'autres personnes ou des conditions d'enregistrement différentes.

Les performances peuvent notamment dépendre :

- du nombre et de la diversité des enregistrements ;
- de la qualité du microphone ;
- du bruit ambiant ;
- de la durée des fichiers audio ;
- de la similarité entre les voix ;
- de la représentativité des données d'entraînement.



## 👩‍💻 Auteur

**Nomena Misedratiana**

Étudiante en Master 1 — Génie Logiciel

Université de Vakinankaratra

Madagascar

