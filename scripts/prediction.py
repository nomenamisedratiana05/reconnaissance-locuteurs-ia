from pathlib import Path
import argparse

import joblib
import numpy as np

from entrainement import extraire_caracteristiques

RACINE = Path(__file__).resolve().parent.parent
FICHIER_MODELE = RACINE / "models" / "modele_voix.joblib"

SEUIL_CONFIANCE = 0.60


def predire_locuteur(fichier_audio):
    fichier_audio = Path(fichier_audio)

    if not fichier_audio.is_file():
        print("ERREUR : le fichier audio n'existe pas.")
        return

    if not FICHIER_MODELE.exists():
        print("ERREUR : modèle introuvable. Lance d'abord entrainement.py.")
        return

    donnees = joblib.load(FICHIER_MODELE)
    reseaux = donnees["reseaux"]
    scaler = donnees["scaler"]

    try:
        # Une ligne par segment de 3 secondes.
        segments = extraire_caracteristiques(fichier_audio)
        X = scaler.transform(segments)

        # Moyenne des probabilités : tous les réseaux, tous les segments.
        probas = np.mean([r.predict_proba(X) for r in reseaux], axis=0)
        probabilites = probas.mean(axis=0)

        # Les classes du réseau sont des numéros : retrouver les noms.
        noms = donnees["classes"]
        classes = [noms[int(c)] for c in reseaux[0].classes_]
        indice = int(np.argmax(probabilites))
        prediction = classes[indice]

        print("\n=== RESULTAT DE LA RECONNAISSANCE ===")
        print(f"Fichier audio : {fichier_audio.name}")
        print(f"Segments analysés : {len(segments)}")
        print(f"Locuteur prédit : {prediction}")

        print("\nScores estimés par catégorie :")
        for classe, p in zip(classes, probabilites):
            print(f"{classe} : {p * 100:.2f} %")

        if probabilites[indice] < SEUIL_CONFIANCE:
            print(
                "\nATTENTION : la prédiction est incertaine. "
                "Il faut vérifier le résultat."
            )

    except Exception as erreur:
        print(f"Impossible d'analyser cet audio : {erreur}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconnaissance du locuteur")
    parser.add_argument("audio", help="Chemin vers le fichier audio à analyser")
    predire_locuteur(parser.parse_args().audio)