"""Comparer le comportement de plusieurs politiques à partir d'une même situation de départ.

Pour chaque politique, le script produit deux figures :
  - predictions.png : on demande plusieurs fois à la politique quelles actions elle prévoit, pour la même
    observation de départ. Chaque ligne montre la suite de positions visées (le "paquet d'actions").
  - episodes.png : on joue plusieurs épisodes complets à partir de la même situation de départ,
    et on trace le chemin suivi par l'agent pendant les premières secondes.

Exemple :
  python scripts/comparer_politiques.py --politiques modeles/act_pusht_206demos_long modeles/diffusion_pusht_keypoints

Les épisodes avec Diffusion Policy sont lents sur CPU (1 à 3 minutes) : soyez patient(e).
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # on enregistre les figures dans des fichiers, sans ouvrir de fenêtre
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from lerobot.envs.utils import preprocess_observation  # noqa: E402
from outils_pusht import NB_PAS_MAX, TACHE, creer_env  # noqa: E402
from tester_politique import DECALAGE_CENTRE_T, charger_politique  # noqa: E402

TAILLE_IMAGE = 512  # l'image de la scène est rendue à l'échelle 1 (1 pixel = 1 unité)


def preparer_lot(obs, pretraitement, nb):
    """Met une ou plusieurs observations au format attendu par la politique (lot de taille nb)."""
    lot = preprocess_observation(obs)
    lot["task"] = [TACHE] * nb
    return pretraitement(lot)


def tirer_paquets(politique, pretraitement, posttraitement, obs, nb_tirages):
    """Demande nb_tirages fois à la politique le paquet d'actions qu'elle prévoit pour la même observation."""
    lot = preparer_lot(obs, pretraitement, 1)
    lot = {cle: valeur.repeat(nb_tirages, 1) for cle, valeur in lot.items() if cle.startswith("observation")}
    with torch.inference_mode():
        if politique.config.type == "diffusion":
            # Diffusion Policy regarde les n_obs_steps dernières observations. Au début d'un épisode,
            # elle utilise plusieurs copies de la première observation : on fait de même.
            n_obs = politique.config.n_obs_steps
            lot = {cle: valeur.unsqueeze(1).repeat(1, n_obs, 1) for cle, valeur in lot.items()}
            conditionnement = politique.diffusion._prepare_global_conditioning(lot)
            actions = politique.diffusion.conditional_sample(nb_tirages, global_cond=conditionnement)
            actions = actions[:, n_obs - 1 :]  # on garde les actions à partir de l'instant présent
        else:
            actions = politique.predict_action_chunk(lot)
    return posttraitement(actions).cpu().numpy()


def jouer_episodes(politique, pretraitement, posttraitement, options, nb_episodes):
    """Joue nb_episodes épisodes en parallèle, tous à partir de la même situation de départ."""
    envs = [creer_env(TAILLE_IMAGE) for _ in range(nb_episodes)]
    observations = [env.reset(options=options)[0] for env in envs]
    trajectoires = [[obs["agent_pos"].copy()] for obs in observations]
    recouvrements_max = np.zeros(nb_episodes)
    reussis = np.zeros(nb_episodes, dtype=bool)
    politique.reset()
    for _ in range(NB_PAS_MAX):
        lot_obs = {cle: np.stack([obs[cle] for obs in observations]) for cle in observations[0]}
        lot = preparer_lot(lot_obs, pretraitement, nb_episodes)
        with torch.inference_mode():
            actions = posttraitement(politique.select_action(lot)).cpu().numpy()
        for i, env in enumerate(envs):
            if reussis[i]:
                continue  # cet épisode est terminé : on ne le fait plus avancer
            observations[i], _, termine, _, info = env.step(actions[i])
            trajectoires[i].append(observations[i]["agent_pos"].copy())
            recouvrements_max[i] = max(recouvrements_max[i], info["coverage"])
            reussis[i] = termine
        if reussis.all():
            break
    for env in envs:
        env.close()
    return [np.array(t) for t in trajectoires], recouvrements_max, reussis


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--politiques", nargs="+", required=True, help="dossiers des modèles à comparer")
    parser.add_argument(
        "--etat-initial",
        type=float,
        nargs=5,
        default=[255, 404, 271, 160, 2.06],
        metavar=("X_AGENT", "Y_AGENT", "X_T", "Y_T", "ANGLE_T"),
        help="situation de départ : agent (x, y), centre du T (x, y), angle du T en radians",
    )
    parser.add_argument("--nb-tirages", type=int, default=20, help="nombre de paquets d'actions demandés")
    parser.add_argument("--nb-episodes", type=int, default=8, help="nombre d'épisodes joués par politique")
    parser.add_argument("--duree-trace", type=float, default=4.0, help="durée (s) des chemins tracés")
    parser.add_argument("--graine", type=int, default=0, help="graine aléatoire (pour reproduire les résultats)")
    parser.add_argument("--sortie", default="rapport/images/comparaison", help="dossier des figures")
    args = parser.parse_args()

    x_agent, y_agent, x_t, y_t, angle_t = args.etat_initial
    options = {"reset_to_state": [x_agent, y_agent, x_t, y_t - DECALAGE_CENTRE_T, angle_t]}
    env = creer_env(TAILLE_IMAGE)
    obs, _ = env.reset(options=options)
    image_depart = env.render()
    env.close()

    sortie = Path(args.sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    nb = len(args.politiques)
    fig_pred, axes_pred = plt.subplots(1, nb, figsize=(5 * nb, 5.3), squeeze=False)
    fig_ep, axes_ep = plt.subplots(1, nb, figsize=(5 * nb, 5.3), squeeze=False)
    nb_pas_trace = int(args.duree_trace * 10)

    for colonne, chemin in enumerate(args.politiques):
        nom = Path(chemin).name
        print(f"\n=== {nom} ===")
        politique, pretraitement, posttraitement = charger_politique(chemin)

        # Figure 1 : plusieurs prédictions pour la même observation.
        torch.manual_seed(args.graine)
        debut = time.time()
        paquets = tirer_paquets(politique, pretraitement, posttraitement, obs, args.nb_tirages)
        print(f"{args.nb_tirages} paquets de {paquets.shape[1]} actions prédits en {time.time() - debut:.1f} s")
        ax = axes_pred[0, colonne]
        ax.imshow(image_depart)
        for paquet in paquets:
            ax.plot(paquet[:, 0], paquet[:, 1], color="darkorange", lw=1.5, alpha=0.6)
            ax.plot(paquet[-1, 0], paquet[-1, 1], "o", color="darkorange", ms=4)
        ax.set_title(f"{nom}\n{args.nb_tirages} prédictions pour la même observation", fontsize=10)
        ax.axis("off")

        # Figure 2 : plusieurs épisodes à partir de la même situation de départ.
        torch.manual_seed(args.graine)
        debut = time.time()
        trajectoires, recouvrements, reussis = jouer_episodes(
            politique, pretraitement, posttraitement, options, args.nb_episodes
        )
        print(f"{args.nb_episodes} épisodes joués en {time.time() - debut:.0f} s")
        print("Recouvrement maximal de chaque épisode :", " ".join(f"{r:.0%}" for r in recouvrements))
        ax = axes_ep[0, colonne]
        ax.imshow(image_depart)
        for trajectoire in trajectoires:
            ax.plot(trajectoire[:nb_pas_trace, 0], trajectoire[:nb_pas_trace, 1], lw=2, alpha=0.8)
        ax.set_title(
            f"{nom}\n{args.nb_episodes} épisodes (chemin des {args.duree_trace:.0f} premières secondes)\n"
            f"réussis : {reussis.sum()}/{args.nb_episodes} - recouvrement max moyen : {recouvrements.mean():.0%}",
            fontsize=10,
        )
        ax.axis("off")

    fig_pred.savefig(sortie / "predictions.png", dpi=90, bbox_inches="tight")
    fig_ep.savefig(sortie / "episodes.png", dpi=90, bbox_inches="tight")
    print(f"\nFigures enregistrées : {sortie / 'predictions.png'} et {sortie / 'episodes.png'}")


if __name__ == "__main__":
    main()
