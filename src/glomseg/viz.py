"""Visualisations : erreurs VP/FP/FN, contours numérotés, carte d'incertitude."""
from __future__ import annotations

import cv2
import numpy as np

GREEN, RED, BLUE, YELLOW = (40, 200, 70), (230, 40, 40), (40, 110, 255), (255, 200, 0)


def error_overlay(image, gt, pred, alpha=0.45):
    """Vert = vrai positif, rouge = faux positif, bleu = faux négatif."""
    color = np.zeros_like(image)
    color[gt & pred] = GREEN
    color[~gt & pred] = RED
    color[gt & ~pred] = BLUE
    any_ = gt | pred
    out = image.copy()
    out[any_] = (image[any_] * (1 - alpha) + color[any_] * alpha).astype(np.uint8)
    return out


def contour_overlay(image, labels, table=None, thickness=None):
    """Contours de chaque glomérule ; jaune si la prédiction est incertaine. Numéro au centre."""
    out = image.copy()
    t = thickness or max(2, image.shape[0] // 300)
    uncertain = {r["id"]: r.get("uncertain", False) for r in (table or [])}
    for k in range(1, int(labels.max()) + 1):
        obj = (labels == k).astype(np.uint8)
        if not obj.any():
            continue
        contours, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        col = YELLOW if uncertain.get(k) else GREEN
        cv2.drawContours(out, contours, -1, col, t)
        ys, xs = np.nonzero(obj)
        cv2.putText(out, str(k), (int(xs.mean()) - 8, int(ys.mean()) + 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4 + image.shape[0] / 1500, (255, 255, 255), t + 1, cv2.LINE_AA)
        cv2.putText(out, str(k), (int(xs.mean()) - 8, int(ys.mean()) + 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4 + image.shape[0] / 1500, (0, 0, 0), max(1, t - 1), cv2.LINE_AA)
    return out


def heatmap(values, vmax=None, colormap=cv2.COLORMAP_INFERNO):
    vmax = vmax or max(float(values.max()), 1e-6)
    v = np.clip(values / vmax, 0, 1)
    return cv2.cvtColor(cv2.applyColorMap((v * 255).astype(np.uint8), colormap), cv2.COLOR_BGR2RGB)


def label_panel(image, text):
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (255, 255, 255), -1)
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
    return out


def gallery_row(image, gt, pred, prob, std, title):
    """Image | probas | incertitude | erreurs : une ligne de la galerie d'erreurs."""
    panels = [label_panel(image, title),
              label_panel(heatmap(prob, 1.0), "probabilite"),
              label_panel(heatmap(std, 0.5), "incertitude (std TTA)"),
              label_panel(error_overlay(image, gt, pred), "VP vert / FP rouge / FN bleu")]
    return np.hstack(panels)
