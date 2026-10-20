"""Explorer un dataset de démonstrations PushT : chiffres clés et figures.

Exemples :
  python scripts/explorer_dataset.py --dataset lerobot/pusht_keypoints     (dataset de référence, sur le Hub)
  python scripts/explorer_dataset.py --dossier donnees/mes_demos           (vos démonstrations, sur votre disque)

Les figures sont enregistrées dans le dossier indiqué par --sortie (par défaut : rapport/images/<nom du dataset>).

L'option --marquer X Y (à répéter si besoin) dessine une croix rouge sur la carte de chaleur, au point (X, Y),
pour vérifier si un centre du T que vous voulez tester a déjà été visité pendant les démonstrations :
  python scripts/explorer_dataset.py --dataset lerobot/pusht_keypoints --marquer 90 301 --marquer 300 345
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # on enregistre les figures dans des fichiers, sans ouvrir de fenêtre
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

TAILLE_SCENE = 512
NB_CASES = 32  # la carte de chaleur découpe la scène en 32 x 32 cases de 16 unités de côté


def contour_cible():
    """Les deux rectangles qui forment le T cible (position fixe : centre (256, 256), angle 45°)."""
    angle = np.pi / 4
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    barre = np.array([[-60, 30], [60, 30], [60, 0], [-60, 0]])
    pied = np.array([[-15, 30], [-15, 120], [15, 120], [15, 30]])
    return [rectangle @ rotation.T + 256 for rectangle in (barre, pied)]


def preparer_axe(ax, titre):
    for rectangle in contour_cible():
        ax.add_patch(Polygon(rectangle, closed=True, color="lightgreen", alpha=0.6))
    ax.set_xlim(0, TAILLE_SCENE)
    ax.set_ylim(TAILLE_SCENE, 0)  # axe y vers le bas, comme dans la fenêtre de simulation
    ax.set_aspect("equal")
    ax.set_title(titre)


def dessiner_carte_de_chaleur(ax, positions, titre, marques):
    """Compte combien de pas de temps ont été passés dans chaque case de la scène et l'affiche en couleur."""
    comptes, _, _ = np.histogram2d(
        positions[:, 0], positions[:, 1], bins=NB_CASES, range=[[0, TAILLE_SCENE], [0, TAILLE_SCENE]]
    )
    # Les cases jamais visitées sont masquées : elles restent blanches.
    comptes = np.ma.masked_equal(comptes.T, 0)
    image = ax.imshow(
        comptes,
        extent=(0, TAILLE_SCENE, TAILLE_SCENE, 0),
        cmap="viridis",
        norm=LogNorm(vmin=1, vmax=max(comptes.max(), 2)),
    )
    for rectangle in contour_cible():
        ax.add_patch(Polygon(rectangle, closed=True, fill=False, edgecolor="limegreen", lw=2))
    for x, y in marques:
        ax.plot(x, y, "x", color="red", ms=12, mew=3)
    ax.set_xlim(0, TAILLE_SCENE)
    ax.set_ylim(TAILLE_SCENE, 0)
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.set_title(titre)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="identifiant d'un dataset du Hub Hugging Face (ex : lerobot/pusht_keypoints)")
    source.add_argument("--dossier", help="dossier local d'un dataset (ex : donnees/mes_demos)")
    parser.add_argument("--nb-trajectoires", type=int, default=20, help="nombre de trajectoires à dessiner")
    parser.add_argument("--sortie", default=None, help="dossier où enregistrer les figures")
    parser.add_argument(
        "--marquer",
        type=float,
        nargs=2,
        action="append",
        default=[],
        metavar=("X", "Y"),
        help="point à marquer d'une croix rouge sur la carte de chaleur (option répétable)",
    )
    args = parser.parse_args()

    if args.dossier:
        nom = Path(args.dossier).name
        dataset = LeRobotDataset(f"local/{nom}", root=args.dossier)
    else:
        nom = args.dataset.split("/")[-1]
        dataset = LeRobotDataset(args.dataset)
    sortie = Path(args.sortie or f"rapport/images/{nom}")
    sortie.mkdir(parents=True, exist_ok=True)

    # ---- Chiffres clés -------------------------------------------------------------------------
    print(f"\n=== Dataset : {args.dataset or args.dossier} ===")
    print(f"Nombre d'épisodes (démonstrations) : {dataset.num_episodes}")
    print(f"Nombre total de pas de temps (frames) : {dataset.num_frames}")
    print(f"Fréquence d'enregistrement : {dataset.fps} pas par seconde")
    print("\nContenu de chaque pas de temps (nom : taille) :")
    for cle, description in dataset.features.items():
        print(f"  - {cle} : {tuple(description['shape'])}  ({description['dtype']})")

    donnees = dataset.hf_dataset.with_format("numpy")
    episodes = np.asarray(donnees["episode_index"]).reshape(-1)
    etats = np.stack(donnees["observation.state"])  # position de l'agent
    actions = np.stack(donnees["action"])  # position visée
    coins_t = np.stack(donnees["observation.environment_state"]).reshape(-1, 8, 2)  # 8 coins du T

    liste_episodes = np.unique(episodes)
    durees = np.array([np.sum(episodes == e) for e in liste_episodes]) / dataset.fps
    print(f"\nDurée des démonstrations : moyenne {durees.mean():.1f} s, min {durees.min():.1f} s, max {durees.max():.1f} s")
    ecart = np.linalg.norm(actions - etats, axis=1)
    print(f"Distance moyenne entre l'action (position visée) et la position de l'agent : {ecart.mean():.1f} unités")

    # ---- Figure 1 : trajectoires de l'agent ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    preparer_axe(ax, f"Trajectoires de l'agent ({min(args.nb_trajectoires, len(liste_episodes))} épisodes)")
    for e in liste_episodes[: args.nb_trajectoires]:
        trajectoire = etats[episodes == e]
        ligne = ax.plot(trajectoire[:, 0], trajectoire[:, 1], lw=1, alpha=0.8)[0]
        ax.plot(*trajectoire[0], "o", color=ligne.get_color(), ms=5)
    ax.text(5, 505, "rond = point de départ, zone verte = cible", fontsize=8)
    fig.savefig(sortie / "trajectoires.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2 : positions de départ du T et de l'agent -------------------------------------
    premiers = np.array([np.flatnonzero(episodes == e)[0] for e in liste_episodes])
    fig, ax = plt.subplots(figsize=(6, 6))
    preparer_axe(ax, f"Positions de départ ({len(liste_episodes)} épisodes)")
    centres_t = coins_t[premiers].mean(axis=1)
    ax.scatter(centres_t[:, 0], centres_t[:, 1], c="dimgray", marker="s", s=20, label="centre du T")
    ax.scatter(etats[premiers, 0], etats[premiers, 1], c="royalblue", s=20, label="agent")
    ax.legend(loc="upper right")
    fig.savefig(sortie / "positions_depart.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 3 : durée des démonstrations ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(durees, bins=20, color="slategray")
    ax.set_xlabel("durée de la démonstration (s)")
    ax.set_ylabel("nombre d'épisodes")
    ax.set_title("Durée des démonstrations")
    fig.savefig(sortie / "durees.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 4 : action et état au cours d'un épisode ---------------------------------------
    masque = episodes == liste_episodes[0]
    temps = np.arange(masque.sum()) / dataset.fps
    fig, axes = plt.subplots(2, 1, figsize=(7, 4.5), sharex=True)
    for i, (axe, nom_coord) in enumerate(zip(axes, ["x", "y"], strict=True)):
        axe.plot(temps, etats[masque, i], label="état : position de l'agent")
        axe.plot(temps, actions[masque, i], "--", label="action : position visée")
        axe.set_ylabel(nom_coord)
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"Épisode {liste_episodes[0]} : état et action au cours du temps")
    axes[1].set_xlabel("temps (s)")
    fig.savefig(sortie / "etat_et_action.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 5 : carte de chaleur des positions visitées ------------------------------------
    # Le centre du T est la moyenne de ses 8 coins (c'est aussi son centre de gravité).
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), layout="constrained")
    dessiner_carte_de_chaleur(axes[0], coins_t.mean(axis=1), "Centre du T", args.marquer)
    image = dessiner_carte_de_chaleur(axes[1], etats, "Agent", [])
    fig.colorbar(image, ax=axes, shrink=0.8, label="nombre de pas de temps passés dans la case")
    fig.suptitle(f"Positions visitées pendant les {len(liste_episodes)} démonstrations (blanc = jamais visité)")
    fig.savefig(sortie / "carte_de_chaleur.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    print(f"\nFigures enregistrées dans le dossier '{sortie}' :")
    for fichier in sorted(sortie.glob("*.png")):
        print(f"  - {fichier}")


if __name__ == "__main__":
    main()
