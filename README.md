<div align="center">

# 🔍 Détecteur d'images générées par IA

**Analyse statistique des gradients locaux pour distinguer photographies réelles
et images générées par intelligence artificielle.**

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pillow](https://img.shields.io/badge/Pillow-Image_I%2FO-0A0A0A?style=for-the-badge)
![unittest](https://img.shields.io/badge/unittest-stdlib-4B8BBE?style=for-the-badge&logo=python&logoColor=white)
![setuptools](https://img.shields.io/badge/setuptools-pyproject.toml-0E7C0E?style=for-the-badge&logo=setuptools&logoColor=white)
![JSON](https://img.shields.io/badge/JSON-mod%C3%A8les-000000?style=for-the-badge&logo=json&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)

**100 % local · 2 dépendances · aucun service en ligne · modèle lisible en JSON**

</div>

---

## 🛠️ Stack technique

Le projet est une chaîne **Python + NumPy + Pillow**, sans serveur, sans base de
données et sans appel réseau :

- **[Python 3.11+](https://www.python.org/)** — calcul du champ de vecteurs,
  features statistiques, entraînement et classification ; scripts en ligne de
  commande (`argparse`, `pathlib`, `csv`, `json`).
- **[NumPy](https://numpy.org/)** — traitement vectorisé des images 16 bits,
  histogrammes, covariances et calcul des features.
- **[Pillow](https://python-pillow.org/)** — décodage des images (JPEG, PNG,
  TIFF, WebP) et écriture des sorties PNG 16 bits ; les visualisations sont
  rendues directement en pixels, **sans bibliothèque de graphiques**.
- **[setuptools / pyproject.toml](https://setuptools.pypa.io/)** — paquet
  installable `detecteur_image_ia` (layout `src/`), compatible PEP 621.
- **[unittest](https://docs.python.org/3/library/unittest.html)** — tests de la
  bibliothèque standard pour le champ de vecteurs et les features.
- **[JSON](https://www.json.org/)** — modèles de régression logistique stockés
  en clair dans `models/` (features, poids, biais, seuil).
- **[GitHub](https://github.com/)** — hébergement du dépôt et des modèles.

> **Aucune installation du paquet n'est nécessaire** : les scripts ajoutent
> `src/` au chemin d'import. Seules les deux dépendances externes sont à
> installer.

```bash
git clone https://github.com/Davidlouiz/detecteur_image_ia.git
cd detecteur_image_ia
pip install numpy Pillow
```

## 🧠 Principe

L'approche repose sur le fait que les images générées par IA et les photographies
réelles présentent des signatures statistiques différentes au niveau des
transitions entre pixels voisins.

Pour chaque pixel, on calcule un vecteur de log-ratio avec ses voisins de droite
et du bas :

```text
vx = log2((g(x+1, y) + 1) / (g(x, y) + 1))
vy = log2((g(x, y+1) + 1) / (g(x, y) + 1))
```

Le `+1` évite les divisions par zéro. Les valeurs sont multipliées par
`1_000_000` et arrondies en `int32` pour garantir un calcul déterministe.

<p align="center">
  <img src="docs/principe.svg" alt="Principe du champ de vecteurs de log-ratio" width="70%">
</p>

L'ensemble de ces vecteurs forme un **champ de vecteurs de log-ratio** dont
la distribution statistique (entropie, symétries, concentration, diversité)
diffère selon l'origine de l'image. Un modèle de régression logistique simple
apprend à distinguer ces signatures.

Le pipeline de traitement est :

```text
image → gris 16 bits → vecteurs de log-ratio → features statistiques → classification
```

## 📦 Formats de fichiers intermédiaires

- `*_source_pixels.npz` : pixels décodés de l'image d'origine
- `*_gray16.png` : PNG en niveaux de gris 16 bits (`uint16` par pixel)
- `*_vectors.npz` : champ de vecteurs (`vx`, `vy` en `int32`)
- `*_distribution.npz` : triplets `vx`, `vy`, `count` pour chaque vecteur distinct
- `*_vectors.png` : visualisation de la distribution (avec léger lissage de densité)

## 🚀 Utilisation

Les scripts sont autonomes (ils ajoutent `src/` au chemin d'import) : aucune
installation du paquet n'est requise.

Analyser une image :

```bash
python3 scripts/analyze_image.py image_entree.png sortie --plot-size 1024
```

Analyser toutes les images d'un dossier :

```bash
python3 scripts/analyze_directory.py images/ resultats/ --plot-size 1024
```

Vue zoomée sur la distribution centrale :

```bash
python3 scripts/analyze_directory.py images/ resultats/ --plot-size 1024 --percentile 99.5
```

## 🔎 Détection

Classer une image ou un dossier :

```bash
python3 scripts/detect.py images/ --csv resultats/predictions.csv
```

Classer et copier les images dans deux dossiers de contrôle :

```bash
python3 scripts/detect.py images/ \
  --csv resultats/predictions.csv \
  --copy-to resultats/controle
```

Cela crée :

- `resultats/controle/image_ia_probable/`
- `resultats/controle/image_reelle_probable/`

Afficher les features statistiques :

```bash
python3 scripts/detect.py images/ --show-features
```

## 🏋️ Entraînement

Réentraîner le modèle avec des images labellisées :

```bash
python3 scripts/train_detector.py models/current_detector.json \
  --fake-dir images_ia/ \
  --real-dir images_reelles/
```

Ou avec des motifs de fichiers :

```bash
python3 scripts/train_detector.py models/current_detector.json \
  --fake-pattern 'images_ia/*.png' \
  --real-pattern 'images_reelles/*.jpg'
```

Le détecteur n'utilise ni le nom de fichier, ni l'extension, ni les métadonnées.
Il reste provisoire : il faudra le réentraîner et le valider avec un jeu plus
grand et plus varié.

## 🧪 Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Les tests couvrent la conversion en gris 16 bits, le centrage du champ de
vecteurs et les features de covariance.

## 🗂️ Modèles

- `models/current_detector.json` : modèle courant (régression logistique sur
  features statistiques des vecteurs de log-ratio)
- `models/covariance_detector.json` : variante basée uniquement sur la covariance,
  conservée pour comparaison
