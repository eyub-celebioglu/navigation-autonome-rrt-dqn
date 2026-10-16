"""
generer_graphes.py
Génère les figures pour le rapport à partir de results/data/entrainement_dqn.csv
et results/data/benchmark_rrt.csv.

Colonnes attendues dans entrainement_dqn.csv (séparateur ;) :
  episode ; score ; epsilon ; succes ; nb_pas ; index_carte ; loss

Usage : python generer_graphes.py
Les figures sont sauvegardées en PNG dans results/figures/.
"""

import csv
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# chemins relatifs a la racine du depot
RACINE = Path(__file__).resolve().parent.parent
DOSSIER_DATA = RACINE / "results" / "data"
DOSSIER_FIGURES = RACINE / "results" / "figures"

matplotlib.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

NOMS_CARTES = {
    0: "Murs simples",
    1: "Arche",
    2: "Vide",
    3: "Mur central",
    4: "Boîte centrale",
}

# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────────────────────────────────────────

def moving_average(data, window=50):
    if len(data) < window:
        window = max(1, len(data) // 2)
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid"), window

def success_rate_sliding(successes, window=100):
    rates, centers = [], []
    step = window // 2
    for i in range(0, len(successes) - window + 1, step):
        chunk = successes[i : i + window]
        rates.append(sum(chunk) / len(chunk) * 100)
        centers.append(i + window // 2)
    return centers, rates

def lire_csv(csv_path=DOSSIER_DATA / "entrainement_dqn.csv"):
    """Lit le CSV et retourne un dict de listes."""
    data = {k: [] for k in ["episode", "score", "epsilon", "succes",
                              "nb_pas", "index_carte", "loss"]}
    if not os.path.exists(csv_path):
        print(f"[SKIP] {csv_path} introuvable")
        return None

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            try:
                # ignore la ligne d'en-tête si elle existe
                int(row[0])
            except (ValueError, IndexError):
                continue
            try:
                data["episode"].append(int(row[0]))
                data["score"].append(float(row[1].replace(",", ".")))
                data["epsilon"].append(float(row[2].replace(",", ".")))
                data["succes"].append(int(row[3]) if len(row) > 3 else 0)
                data["nb_pas"].append(int(row[4]) if len(row) > 4 else 0)
                data["index_carte"].append(int(row[5]) if len(row) > 5 else -1)
                data["loss"].append(float(row[6].replace(",", ".")) if len(row) > 6 else 0.0)
            except (ValueError, IndexError):
                pass

    if not data["episode"]:
        print("[SKIP] CSV vide.")
        return None
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 – Courbe d'apprentissage (score + epsilon)
# ─────────────────────────────────────────────────────────────────────────────

def fig_learning_curve(data):
    if data is None:
        return
    episodes = data["episode"]
    scores   = data["score"]
    epsilons = data["epsilon"]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.plot(episodes, scores, color="#94c4f5", alpha=0.35, linewidth=0.8, label="Score brut")

    ma, win = moving_average(scores, 50)
    ax1.plot(episodes[win - 1:], ma, color="#1a6fbd", linewidth=2.0,
             label=f"Moyenne glissante ({win} éps.)")

    ax2.plot(episodes, epsilons, color="#e07b39", linewidth=1.5,
             linestyle="--", alpha=0.8, label="ε (exploration)")
    ax2.set_ylabel("Epsilon (ε)", color="#e07b39", fontsize=11)
    ax2.tick_params(axis="y", colors="#e07b39")
    ax2.set_ylim(0, 1.05)

    ax1.set_xlabel("Épisode", fontsize=12)
    ax1.set_ylabel("Score de l'épisode", fontsize=12)
    ax1.set_title("Courbe d'apprentissage du DQN", fontsize=14, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig(DOSSIER_FIGURES / "fig1_learning_curve.png", bbox_inches="tight")
    plt.close()
    print("✓ fig1_learning_curve.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 – Taux de succès global glissant
# ─────────────────────────────────────────────────────────────────────────────

def fig_success_rate(data):
    if data is None:
        return
    successes = data["succes"]
    if len(successes) < 100:
        print("[SKIP] fig2 — pas assez de données succès (< 100 épisodes)")
        return

    centers, rates = success_rate_sliding(successes, window=100)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(centers, rates, alpha=0.18, color="#2ca05a")
    ax.plot(centers, rates, color="#2ca05a", linewidth=2)
    ax.set_xlabel("Épisode", fontsize=12)
    ax.set_ylabel("Taux de succès (%)", fontsize=12)
    ax.set_title("Taux de succès du DQN (fenêtre 100 épisodes)", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(DOSSIER_FIGURES / "fig2_success_rate.png", bbox_inches="tight")
    plt.close()
    print("✓ fig2_success_rate.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 – Taux de succès par carte
# ─────────────────────────────────────────────────────────────────────────────

def fig_success_par_carte(data):
    if data is None:
        return

    # regroupe les succès par index de carte
    par_carte = {}
    for s, idx in zip(data["succes"], data["index_carte"]):
        if idx not in par_carte:
            par_carte[idx] = []
        par_carte[idx].append(s)

    indices  = sorted(par_carte.keys())
    noms     = [NOMS_CARTES.get(i, f"Carte {i}") for i in indices]
    taux     = [sum(par_carte[i]) / len(par_carte[i]) * 100 for i in indices]
    nb_eps   = [len(par_carte[i]) for i in indices]

    colors = ["#1a6fbd", "#e07b39", "#2ca05a", "#9b59b6", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(noms, taux,
                  color=[colors[i % len(colors)] for i in range(len(indices))],
                  alpha=0.85, width=0.5)

    for bar, val, n in zip(bars, taux, nb_eps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%\n(n={n})",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Taux de succès (%)", fontsize=12)
    ax.set_title("Taux de succès par type de carte", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_ylim(0, 115)
    plt.tight_layout()
    plt.savefig(DOSSIER_FIGURES / "fig3_success_par_carte.png", bbox_inches="tight")
    plt.close()
    print("✓ fig3_success_par_carte.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 – Courbe de loss (erreur du réseau)
# ─────────────────────────────────────────────────────────────────────────────

def fig_loss_curve(data):
    if data is None:
        return
    losses = data["loss"]
    if all(l == 0.0 for l in losses):
        print("[SKIP] fig4 — colonne loss absente ou nulle")
        return

    episodes = data["episode"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(episodes, losses, color="#c0392b", alpha=0.35, linewidth=0.8, label="Loss brute")

    ma, win = moving_average(losses, 50)
    ax.plot(episodes[win - 1:], ma, color="#922b21", linewidth=2.0,
            label=f"Moyenne glissante ({win} éps.)")

    ax.set_xlabel("Épisode", fontsize=12)
    ax.set_ylabel("Loss moyenne (SmoothL1)", fontsize=12)
    ax.set_title("Évolution de l'erreur du réseau (Loss)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(DOSSIER_FIGURES / "fig4_loss_curve.png", bbox_inches="tight")
    plt.close()
    print("✓ fig4_loss_curve.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 – Comparaison RRT / RRTSC / RRT*
# ─────────────────────────────────────────────────────────────────────────────

def fig_rrt_comparison(csv_path=DOSSIER_DATA / "benchmark_rrt.csv"):
    if not os.path.exists(csv_path):
        print(f"[SKIP] fig5 — {csv_path} introuvable (lance benchmark_rrt.py)")
        return

    algos = ["RRT", "RRTSC", "RRT*"]
    data  = {a: {"temps": [], "cout": [], "noeuds": []} for a in algos}
    iterations = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)  # skip header
        for row in reader:
            try:
                iterations.append(int(row[0]))
                for i, algo in enumerate(algos):
                    data[algo]["temps"].append(float(row[1 + i * 3].replace(",", ".")))
                    data[algo]["cout"].append(float(row[2 + i * 3].replace(",", ".")))
                    data[algo]["noeuds"].append(float(row[3 + i * 3].replace(",", ".")))
            except (ValueError, IndexError):
                pass

    if not iterations:
        print("[SKIP] fig5 — CSV RRT vide")
        return

    colors = {"RRT": "#1a6fbd", "RRTSC": "#e07b39", "RRT*": "#2ca05a"}
    titles = ["Temps de calcul (s)", "Coût du chemin (px)", "Nombre de nœuds"]
    keys   = ["temps", "cout", "noeuds"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, title, key in zip(axes, titles, keys):
        for algo in algos:
            ax.plot(iterations, data[algo][key], marker="o", markersize=4,
                    linewidth=2, label=algo, color=colors[algo])
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Itérations", fontsize=10)
        ax.legend(fontsize=9)

    fig.suptitle("Comparaison des variantes RRT (moyenne sur 10 runs)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(DOSSIER_FIGURES / "fig5_rrt_comparison.png", bbox_inches="tight")
    plt.close()
    print("✓ fig5_rrt_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Lecture du CSV…")
    DOSSIER_FIGURES.mkdir(parents=True, exist_ok=True)
    data = lire_csv()

    print("Génération des figures…\n")
    fig_learning_curve(data)
    fig_success_rate(data)
    fig_success_par_carte(data)
    fig_loss_curve(data)
    fig_rrt_comparison()

    print(f"\nTerminé. Figures dans {DOSSIER_FIGURES}")