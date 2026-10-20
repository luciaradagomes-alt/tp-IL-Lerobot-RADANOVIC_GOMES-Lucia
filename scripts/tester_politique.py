"""Regarder une politique entraînée jouer en direct, et la perturber.

Pendant que la politique contrôle l'agent :
  - maintenez le clic gauche : VOUS prenez le contrôle (l'agent suit la souris) ;
  - relâchez le clic : la politique reprend la main à partir de la situation où vous l'avez laissée.
Conseil : cliquez sur le disque bleu (ou juste à côté) avant de le déplacer, pour éviter les sauts brusques.

La croix rouge montre la position visée par l'action en cours (de la politique ou de vous).

Touches :
  Espace  passer à un nouvel épisode
  P       mettre en pause / reprendre
  Échap   quitter

Ce script est prévu pour les politiques ACT : la Diffusion Policy de la partie 5 est trop lente sur CPU
pour être regardée en direct (utilisez plutôt les vidéos de lerobot-eval).

Exemples :
  python scripts/tester_politique.py --politique modeles/act_pusht_206demos_long
  python scripts/tester_politique.py --politique modeles/act_pusht_206demos_long --etat-initial 100 450 380 165 0
"""

import argparse
import time

import numpy as np
import torch
from outils_pusht import BLEU, GRIS, NB_PAS_MAX, NOIR, ORANGE, ROUGE, TACHE, VERT, Fenetre, creer_env

# La simulation place le T par un point de référence situé 45 unités au-dessus de son centre :
# on convertit le centre donné par l'utilisateur en ce point de référence.
DECALAGE_CENTRE_T = 45


def charger_politique(chemin):
    """Charge une politique LeRobot (et ses étapes de pré/post-traitement) depuis un dossier."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    config = PreTrainedConfig.from_pretrained(chemin)
    config.device = "cpu"
    politique = get_policy_class(config.type).from_pretrained(chemin, config=config)
    politique.eval()
    # Le pré-traitement normalise les observations (moyenne 0, écart-type 1) ;
    # le post-traitement remet les actions prédites dans les unités de la scène.
    pretraitement, posttraitement = make_pre_post_processors(
        config, pretrained_path=chemin, preprocessor_overrides={"device_processor": {"device": "cpu"}}
    )
    return politique, pretraitement, posttraitement


def action_de_la_politique(politique, pretraitement, posttraitement, obs):
    from lerobot.envs.utils import preprocess_observation

    lot = preprocess_observation(obs)  # renomme les clés au format LeRobot et ajoute une dimension "lot"
    lot["task"] = [TACHE]
    lot = pretraitement(lot)
    with torch.inference_mode():
        action = politique.select_action(lot)
    action = posttraitement(action)
    return action[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--politique", required=True, help="dossier du modèle (celui qui contient config.json)")
    parser.add_argument("--nb-episodes", type=int, default=None, help="nombre d'épisodes (par défaut : illimité)")
    parser.add_argument("--graine", type=int, default=None, help="graine aléatoire pour les positions de départ")
    parser.add_argument(
        "--etat-initial",
        type=float,
        nargs=5,
        default=None,
        metavar=("X_AGENT", "Y_AGENT", "X_T", "Y_T", "ANGLE_T"),
        help="impose la position de départ : agent (x, y), centre du T (x, y) et angle du T en radians",
    )
    parser.add_argument("--taille-fenetre", type=int, default=600, help="taille de la fenêtre en pixels")
    args = parser.parse_args()

    print(f"Chargement de la politique '{args.politique}'...")
    politique, pretraitement, posttraitement = charger_politique(args.politique)

    env = creer_env(args.taille_fenetre)
    fenetre = Fenetre(args.taille_fenetre, f"PushT - politique : {args.politique}")
    generateur = np.random.default_rng(args.graine)
    options = None
    if args.etat_initial:
        x_agent, y_agent, x_t, y_t, angle_t = args.etat_initial
        options = {"reset_to_state": [x_agent, y_agent, x_t, y_t - DECALAGE_CENTRE_T, angle_t]}

    resultats = []  # un booléen par épisode terminé (réussi ou non)
    recouvrements_max = []
    quitter = False
    try:
        while not quitter and (args.nb_episodes is None or len(resultats) < args.nb_episodes):
            obs, _ = env.reset(seed=int(generateur.integers(0, 1_000_000)), options=options)
            politique.reset()  # vide la file d'actions déjà prédites (voir "action chunking" dans l'énoncé)
            nb_pas, recouvrement, recouvrement_max = 0, 0.0, 0.0
            humain_aux_commandes, pause, fin = False, False, None

            while fin is None:
                touches = fenetre.evenements()
                if "quitter" in touches:
                    fin, quitter = "abandon", True
                    break
                if "space" in touches:
                    fin = "abandon"
                    break
                if "p" in touches:
                    pause = not pause

                if not pause:
                    if fenetre.clic_gauche_enfonce():
                        action = fenetre.souris()
                        humain_aux_commandes = True
                    else:
                        if humain_aux_commandes:
                            # Vous venez de rendre la main : les actions prédites avant votre intervention
                            # ne sont plus valables, on vide la file d'actions de la politique.
                            humain_aux_commandes = False
                            politique.reset()
                        action = action_de_la_politique(politique, pretraitement, posttraitement, obs)

                    obs, _, termine, _, info = env.step(action)
                    nb_pas += 1
                    recouvrement = info["coverage"]
                    recouvrement_max = max(recouvrement_max, recouvrement)
                    if termine:
                        fin = "reussi"
                    elif nb_pas >= NB_PAS_MAX:
                        fin = "echec"

                if humain_aux_commandes:
                    qui = ("VOUS contrôlez l'agent (relâchez le clic pour rendre la main)", ROUGE)
                elif pause:
                    qui = ("PAUSE (P pour reprendre)", ORANGE)
                else:
                    qui = ("La POLITIQUE contrôle l'agent (clic gauche maintenu pour intervenir)", BLEU)
                reussis = sum(resultats)
                lignes = [
                    (
                        f"Épisode {len(resultats) + 1}  -  temps : {nb_pas / 10:4.1f} s / 30 s  -  "
                        f"recouvrement : {recouvrement:.0%}  -  réussis : {reussis}/{len(resultats)}",
                        NOIR,
                    ),
                    qui,
                    ("Espace : épisode suivant   P : pause   Échap : quitter", GRIS),
                ]
                fenetre.afficher(env.render(), lignes, couleur_cadre=ROUGE if humain_aux_commandes else None)
                fenetre.attendre_prochain_pas()

            if fin in ("reussi", "echec"):
                resultats.append(fin == "reussi")
                recouvrements_max.append(recouvrement_max)
                message, couleur = ("Réussi !", VERT) if fin == "reussi" else ("Échec (30 s écoulées)", ORANGE)
                debut = time.time()
                while time.time() - debut < 1.2:
                    fenetre.evenements()
                    fenetre.afficher(env.render(), [], message=message, couleur_message=couleur)
                    fenetre.attendre_prochain_pas()
    finally:
        fenetre.fermer()
        env.close()

    if resultats:
        print(f"\nÉpisodes terminés : {len(resultats)}")
        print(f"Taux de réussite : {np.mean(resultats):.0%}")
        print(f"Recouvrement maximal moyen : {np.mean(recouvrements_max):.0%}")


if __name__ == "__main__":
    main()
