"""Enregistrer ses propres démonstrations de la tâche PushT, à la souris.

Principe :
  - une fenêtre affiche la scène : le disque bleu (l'agent), le T gris et la cible verte ;
  - amenez la souris sur le disque bleu : l'enregistrement démarre et l'agent suit la souris ;
  - poussez le T jusqu'à ce qu'il recouvre la cible verte (au moins 95 % de recouvrement) ;
  - dès que c'est réussi, la démonstration est sauvegardée et un nouvel épisode commence.

Touches :
  R       recommencer l'épisode en cours (la démonstration en cours n'est pas gardée)
  Échap   quitter (les démonstrations déjà réussies sont gardées)

Exemples :
  python scripts/enregistrer_demos.py --entrainement           (pour s'entraîner, rien n'est sauvegardé)
  python scripts/enregistrer_demos.py --nb-demos 20            (enregistre 20 démonstrations réussies)
"""

import argparse
import time

import numpy as np
from outils_pusht import (
    BLEU,
    GRIS,
    NB_PAS_MAX,
    NOIR,
    ORANGE,
    RAYON_ATTRAPE,
    ROUGE,
    VERT,
    Fenetre,
    creer_env,
    obs_vers_frame,
    ouvrir_dataset,
)


def jouer_episode(env, fenetre, dataset, graine, numero, total):
    """Joue un épisode. Renvoie "reussi", "recommencer", "temps_ecoule" ou "quitter"."""
    obs, _ = env.reset(seed=graine)
    enregistrement = False
    nb_pas = 0
    recouvrement = 0.0

    while True:
        touches = fenetre.evenements()
        if "quitter" in touches:
            return "quitter"
        if "r" in touches:
            return "recommencer"

        souris = fenetre.souris()
        if not enregistrement and np.linalg.norm(souris - obs["agent_pos"]) < RAYON_ATTRAPE:
            enregistrement = True  # la souris a "attrapé" l'agent : on commence à enregistrer

        if enregistrement:
            # On enregistre le couple (observation actuelle, action choisie), puis on applique l'action.
            if dataset is not None:
                dataset.add_frame(obs_vers_frame(obs, souris))
            obs, _, termine, _, info = env.step(souris)
            nb_pas += 1
            recouvrement = info["coverage"]
            if termine:
                return "reussi"
            if nb_pas >= NB_PAS_MAX:
                return "temps_ecoule"

        if enregistrement:
            etat = (f"Enregistrement en cours - temps restant : {(NB_PAS_MAX - nb_pas) / 10:4.1f} s", ROUGE)
        else:
            etat = ("Placez la souris sur le disque bleu pour commencer", BLEU)
        lignes = [
            (f"Démonstration {numero}/{total}   -   recouvrement de la cible : {recouvrement:.0%} (objectif 95 %)", NOIR),
            etat,
            ("R : recommencer l'épisode     Échap : quitter", GRIS),
        ]
        fenetre.afficher(env.render(), lignes, couleur_cadre=ROUGE if enregistrement else None)
        fenetre.attendre_prochain_pas()


def afficher_message(env, fenetre, message, couleur, duree=1.2):
    """Affiche un grand message au centre de la fenêtre pendant `duree` secondes."""
    debut = time.time()
    while time.time() - debut < duree:
        fenetre.evenements()
        fenetre.afficher(env.render(), [], message=message, couleur_message=couleur)
        fenetre.attendre_prochain_pas()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dossier", default="donnees/mes_demos", help="dossier où sauvegarder les démonstrations")
    parser.add_argument("--nb-demos", type=int, default=20, help="nombre de démonstrations réussies à enregistrer")
    parser.add_argument("--entrainement", action="store_true", help="mode entraînement : rien n'est sauvegardé")
    parser.add_argument("--graine", type=int, default=None, help="graine aléatoire pour les positions de départ")
    parser.add_argument("--taille-fenetre", type=int, default=600, help="taille de la fenêtre en pixels")
    args = parser.parse_args()

    dataset = None if args.entrainement else ouvrir_dataset(args.dossier)
    deja_presentes = dataset.num_episodes if dataset is not None else 0
    if deja_presentes:
        print(f"Le dataset '{args.dossier}' contient déjà {deja_presentes} démonstrations : on en ajoute.")

    env = creer_env(args.taille_fenetre)
    titre = "PushT - entraînement" if args.entrainement else "PushT - enregistrement de démonstrations"
    fenetre = Fenetre(args.taille_fenetre, titre)
    generateur = np.random.default_rng(args.graine)

    nb_reussies = 0
    nb_ratees = 0
    try:
        while nb_reussies < args.nb_demos:
            graine = int(generateur.integers(0, 1_000_000))
            resultat = jouer_episode(env, fenetre, dataset, graine, nb_reussies + 1, args.nb_demos)

            if resultat == "reussi":
                if dataset is not None:
                    dataset.save_episode()
                nb_reussies += 1
                afficher_message(env, fenetre, "Réussi !", VERT)
            else:
                # Démonstration incomplète : on jette ce qui a été enregistré pour cet épisode.
                if dataset is not None and dataset.has_pending_frames():
                    dataset.clear_episode_buffer()
                if resultat == "quitter":
                    break
                if resultat == "temps_ecoule":
                    nb_ratees += 1
                    afficher_message(env, fenetre, "Temps écoulé (30 s)", ORANGE)
    finally:
        if dataset is not None:
            dataset.finalize()  # indispensable : écrit les fichiers du dataset sur le disque
        fenetre.fermer()
        env.close()

    print(f"\nDémonstrations réussies pendant cette session : {nb_reussies} (temps écoulé : {nb_ratees} fois)")
    if dataset is not None:
        print(f"Total dans le dataset '{args.dossier}' : {deja_presentes + nb_reussies} démonstrations")


if __name__ == "__main__":
    main()
