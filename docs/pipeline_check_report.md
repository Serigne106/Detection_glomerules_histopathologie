# Rapport d'évaluation : demo_9images

4 images, 7 glomérules annotés. Réglages choisis par cross-fitting (2 plis) : chaque image est prédite avec des réglages choisis sans elle.

| Protocole | Dice moyen | écart-type | IC 95 % (bootstrap) | F1 instance | Précision | Rappel |
|---|---|---|---|---|---|---|
| **Honnête (cross-fitting)** | **0.454** | 0.326 | [0.147 ; 0.689] | 0.345 | 0.227 | 0.714 |
| Optimiste (réglé sur le test) | 0.571 | 0.166 | [0.433 ; 0.709] | 0.364 | 0.267 | 0.571 |

## Modes de prédiction (seuil réglé par cross-fitting)

| Mode | Dice | IC 95 % | F1 instance | s / image (CPU) | zoom auto moyen |
|---|---|---|---|---|---|
| simple | 0.571 | [0.433 ; 0.709] | 0.364 | 0.22 | 1.00 |
| tta8 | 0.475 | [0.195 ; 0.725] | 0.101 | 1.65 | 1.00 |
| tta8_ms_centered | 0.469 | [0.176 ; 0.727] | 0.118 | 5.53 | 1.00 |
| tta8_ms_down | 0.466 | [0.159 ; 0.700] | 0.444 | 3.63 | 1.00 |
| auto | 0.502 | [0.147 ; 0.832] | 0.370 | 5.12 | 1.31 |

## Erreurs (protocole honnête)

- Glomérules manqués (quasiment pas détectés) : 2
- Glomérules mal délimités (détectés mais IoU < seuil) : 0
- Glomérules fusionnés avec un voisin : 0
- Fausses détections (aucun glomérule dessous) : 17
- Erreur de comptage moyenne : 3.75 glomérule(s) par image

## Par image

| Image | Zone | Dice | Réels | Prédits | VP | FP | FN | Mode | Seuil |
|---|---|---|---|---|---|---|---|---|---|
| 4389_HE.png_1564_2416 | 1564 | 0.000 | 1 | 6 | 0 | 6 | 1 | auto | 0.65 |
| 4389_HE.png_1564_368 | 1564 | 0.468 | 3 | 4 | 2 | 2 | 1 | auto | 0.65 |
| 4389_HE.png_1564_1392 | 1564 | 0.587 | 2 | 9 | 2 | 7 | 0 | auto | 0.65 |
| 4389_HE.png_3612_1392 | 3612 | 0.762 | 1 | 3 | 1 | 2 | 0 | simple | 0.70 |

Réglage de déploiement (choisi sur toutes les images, à n'utiliser que sur de nouvelles données) : mode `simple`, seuil 0.70, post-traitement oui.
