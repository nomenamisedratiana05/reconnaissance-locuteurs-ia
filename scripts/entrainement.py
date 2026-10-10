import warnings
from pathlib import Path

import joblib
import librosa
import numpy as np

from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Masquer les avertissements répétitifs de lecture audio (mp3/mpeg).
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# 1. CONFIGURATION
# ============================================================

RACINE = Path(__file__).resolve().parent.parent

DOSSIER_ENTRAINEMENT = RACINE / "donnees" / "entrainement"
DOSSIER_TEST = RACINE / "donnees" / "test"
DOSSIER_MODELES = RACINE / "models"
FICHIER_MODELE = DOSSIER_MODELES / "modele_voix.joblib"

CLASSES = ["moi", "ami", "autres"]

EXTENSIONS_AUDIO = {
    ".wav", ".flac", ".ogg", ".mp3", ".mpeg", ".mp4", ".m4a", ".aac",
}

FREQUENCE = 16000
DUREE_SEGMENT = 3.0      # secondes par segment
PAS_SEGMENT = 1.5        # chevauchement de 50 %
NB_AUGMENTATIONS = 3     # copies modifiées de chaque fichier d'entraînement
NB_RESEAUX = 5           # nombre de réseaux dans l'ensemble
GRAINE = 42


# ============================================================
# 2. LECTURE ET DÉCOUPAGE DE L'AUDIO
# ============================================================

def charger_signal(fichier_audio):
    """Lit l'audio en mono 16 kHz, retire le silence aux extrémités
    et normalise le volume."""

    signal, _ = librosa.load(str(fichier_audio), sr=FREQUENCE, mono=True)

    if len(signal) < 2048:
        raise ValueError("Enregistrement audio trop court.")
    if np.max(np.abs(signal)) < 1e-5:
        raise ValueError("Enregistrement audio silencieux.")

    # Enlever les silences au début et à la fin.
    signal_utile, _ = librosa.effects.trim(signal, top_db=30)
    if len(signal_utile) >= 2048:
        signal = signal_utile

    # Normaliser l'amplitude.
    return signal / (np.max(np.abs(signal)) + 1e-9)


def decouper(signal):
    """Découpe le signal en segments de 3 s (avec chevauchement).
    Un signal plus court est utilisé tel quel, complété par répétition."""

    taille = int(DUREE_SEGMENT * FREQUENCE)
    pas = int(PAS_SEGMENT * FREQUENCE)

    if len(signal) <= taille:
        repetitions = int(np.ceil(taille / len(signal)))
        return [np.tile(signal, repetitions)[:taille]]

    segments = [
        signal[debut:debut + taille]
        for debut in range(0, len(signal) - taille + 1, pas)
    ]
    return segments


# ============================================================
# 3. CARACTÉRISTIQUES AUDIO (136 valeurs par segment)
# ============================================================

def caracteristiques_segment(segment):
    """MFCC + deltas + contraste spectral + hauteur de la voix."""

    mfcc = librosa.feature.mfcc(
        y=segment, sr=FREQUENCE, n_mfcc=30, n_fft=512, hop_length=160
    )
    delta = librosa.feature.delta(mfcc)
    contraste = librosa.feature.spectral_contrast(
        y=segment, sr=FREQUENCE, n_fft=512, hop_length=160
    )

    # Hauteur (f0) : médiane et écart-type sur les trames énergétiques.
    f0 = librosa.yin(
        segment, fmin=60, fmax=400, sr=FREQUENCE,
        frame_length=1024, hop_length=160
    )
    energie = librosa.feature.rms(
        y=segment, frame_length=1024, hop_length=160
    )[0][:len(f0)]
    f0 = f0[:len(energie)]
    f0_voisee = f0[energie > 0.5 * np.median(energie)]
    if len(f0_voisee) == 0:
        f0_voisee = f0
    log_f0 = np.log(f0_voisee)

    return np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        delta.mean(axis=1), delta.std(axis=1),
        contraste.mean(axis=1), contraste.std(axis=1),
        [np.median(log_f0), np.std(log_f0)],
    ])


def extraire_caracteristiques(fichier_audio):
    """Retourne une matrice (nb_segments, 136) pour un fichier."""

    signal = charger_signal(fichier_audio)
    return np.array([caracteristiques_segment(s) for s in decouper(signal)])


# ============================================================
# 4. AUGMENTATION DES DONNÉES (entraînement uniquement)
# ============================================================

def augmenter(signal, rng):
    """Crée une version légèrement modifiée du signal."""

    s = signal.copy()

    # Vitesse légèrement différente.
    s = librosa.effects.time_stretch(s, rate=rng.uniform(0.92, 1.08))

    # Petit décalage de hauteur (très faible pour garder l'identité).
    s = librosa.effects.pitch_shift(
        s, sr=FREQUENCE, n_steps=rng.uniform(-0.4, 0.4)
    )

    # Volume aléatoire.
    s = s * rng.uniform(0.6, 1.0)

    # Bruit de fond léger.
    s = s + rng.normal(0, rng.uniform(0.001, 0.01), size=len(s))

    return np.clip(s, -1.0, 1.0)


# ============================================================
# 5. CHARGEMENT DES DONNÉES
# ============================================================

def lister_fichiers(dossier):
    """Retourne la liste (fichier, classe)."""

    resultat = []
    for classe in CLASSES:
        dossier_classe = dossier / classe
        if not dossier_classe.is_dir():
            print(f"Dossier absent : {dossier_classe}")
            continue
        fichiers = sorted(
            f for f in dossier_classe.iterdir()
            if f.is_file() and f.suffix.lower() in EXTENSIONS_AUDIO
        )
        print(f"Catégorie : {classe} | Fichiers trouvés : {len(fichiers)}")
        resultat.extend((f, classe) for f in fichiers)
    return resultat


def charger_par_fichier(dossier, augmentation=False):
    """Retourne une liste d'éléments (X_segments, classe, nom_fichier).
    Les segments d'un même fichier restent groupés, ce qui permet de
    séparer entraînement et validation par FICHIER (sans fuite)."""

    rng = np.random.default_rng(GRAINE)
    elements = []

    for fichier, classe in lister_fichiers(dossier):
        try:
            signal = charger_signal(fichier)
            versions = [signal]
            if augmentation:
                versions += [augmenter(signal, rng)
                             for _ in range(NB_AUGMENTATIONS)]

            segments = []
            for v in versions:
                segments += [caracteristiques_segment(s) for s in decouper(v)]

            elements.append((np.array(segments), classe, fichier.name))
        except Exception as erreur:
            print(f"Fichier ignoré : {fichier.name} | {str(erreur)[:200]}")

    return elements


def aplatir(elements):
    X = np.vstack([e[0] for e in elements])
    # Étiquettes en nombres entiers (0, 1, 2) : nécessaire pour
    # l'arrêt anticipé de MLPClassifier selon les versions de scikit-learn.
    y = np.concatenate(
        [[CLASSES.index(e[1])] * len(e[0]) for e in elements]
    ).astype(int)
    return X, y


def equilibrer(X, y, rng):
    """Duplique au hasard les segments des classes minoritaires."""

    numeros = range(len(CLASSES))
    maximum = max(np.sum(y == c) for c in numeros)
    indices = []
    for c in numeros:
        idx = np.where(y == c)[0]
        supplement = rng.choice(idx, size=maximum - len(idx), replace=True)
        indices.extend(idx)
        indices.extend(supplement)
    indices = rng.permutation(indices)
    return X[indices], y[indices]


# ============================================================
# 6. RÉSEAUX DE NEURONES
# ============================================================

def creer_reseau(couches, alpha, graine):
    return MLPClassifier(
        hidden_layer_sizes=couches,
        activation="relu",
        solver="adam",
        alpha=alpha,                 # régularisation L2
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=600,
        early_stopping=True,         # arrêt quand la validation stagne
        validation_fraction=0.15,
        n_iter_no_change=25,
        random_state=graine,
    )


def predire_fichier(reseaux, scaler, segments):
    """Moyenne des probabilités de tous les réseaux sur tous les segments."""

    X = scaler.transform(segments)
    probas = np.mean([r.predict_proba(X) for r in reseaux], axis=0)
    moyenne = probas.mean(axis=0)
    return moyenne


def choisir_architecture(elements_train, rng):
    """Compare quelques architectures sur des fichiers de validation."""

    candidats = [
        ((128, 64), 1e-3),
        ((256, 128), 1e-3),
        ((256, 128, 64), 1e-2),
        ((512, 256), 1e-2),
    ]

    # Séparation par fichier (20 % de validation, par classe).
    val, ent = [], []
    for classe in CLASSES:
        groupe = [e for e in elements_train if e[1] == classe]
        ordre = rng.permutation(len(groupe))
        nb_val = max(1, int(0.2 * len(groupe))) if len(groupe) >= 5 else 0
        val += [groupe[i] for i in ordre[:nb_val]]
        ent += [groupe[i] for i in ordre[nb_val:]]

    if len(val) == 0:
        print("Pas assez de fichiers pour comparer : architecture par défaut.")
        return candidats[1]

    X, y = aplatir(ent)
    scaler = StandardScaler().fit(X)
    X, y = equilibrer(scaler.transform(X), y, rng)

    meilleur, meilleur_score = candidats[1], -1
    for couches, alpha in candidats:
        reseau = creer_reseau(couches, alpha, GRAINE).fit(X, y)
        bonnes = 0
        for segments, classe, _ in val:
            proba = predire_fichier([reseau], scaler, segments)
            bonnes += reseau.classes_[np.argmax(proba)] == CLASSES.index(classe)
        score = bonnes / len(val)
        print(f"  {couches} alpha={alpha} -> validation {score * 100:.1f} %")
        if score > meilleur_score:
            meilleur, meilleur_score = (couches, alpha), score
    return meilleur


# ============================================================
# 7. PROGRAMME PRINCIPAL
# ============================================================

def main():
    DOSSIER_MODELES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(GRAINE)

    print("\n=== CHARGEMENT DES DONNÉES D'ENTRAÎNEMENT ===")
    elements = charger_par_fichier(DOSSIER_ENTRAINEMENT, augmentation=True)

    if not elements:
        print("ERREUR : aucun enregistrement valide.")
        return
    if {e[1] for e in elements} != set(CLASSES):
        print("ERREUR : il faut des audios lisibles dans les 3 catégories.")
        return

    X, y = aplatir(elements)
    print(f"\nFichiers : {len(elements)} | Segments (avec augmentation) : {len(X)}")

    print("\n=== CHOIX DE L'ARCHITECTURE ===")
    couches, alpha = choisir_architecture(elements, rng)
    print(f"Architecture retenue : {couches}, alpha={alpha}")

    print("\n=== ENTRAÎNEMENT DE L'ENSEMBLE DE RÉSEAUX ===")
    scaler = StandardScaler().fit(X)
    X_norm = scaler.transform(X)

    reseaux = []
    for i in range(NB_RESEAUX):
        Xb, yb = equilibrer(X_norm, y, rng)
        reseau = creer_reseau(couches, alpha, GRAINE + i).fit(Xb, yb)
        print(f"Réseau {i + 1}/{NB_RESEAUX} : {reseau.n_iter_} itérations")
        reseaux.append(reseau)

    joblib.dump(
        {"reseaux": reseaux, "scaler": scaler, "classes": CLASSES},
        FICHIER_MODELE,
    )
    print(f"\nModèle sauvegardé : {FICHIER_MODELE}")

    print("\n=== ÉVALUATION SUR LES DONNÉES DE TEST ===")
    test = charger_par_fichier(DOSSIER_TEST, augmentation=False)
    if not test:
        print("Aucune donnée de test lisible.")
        return

    vrais, predits = [], []
    classes_modele = list(reseaux[0].classes_)
    for segments, classe, _ in test:
        proba = predire_fichier(reseaux, scaler, segments)
        vrais.append(classe)
        predits.append(CLASSES[classes_modele[int(np.argmax(proba))]])

    print(f"\nPrécision (par fichier) : {accuracy_score(vrais, predits) * 100:.2f} %")
    print(classification_report(
        vrais, predits, labels=CLASSES, zero_division=0
    ))


if __name__ == "__main__":
    main()