"""Vérifier que tout est prêt pour le TP (à lancer AVANT la séance).

  python scripts/verifier_installation.py

Le script teste les bibliothèques, télécharge le dataset de référence, et ouvre une fenêtre
de simulation pendant quelques secondes. Chaque étape affiche [OK] ou [ÉCHEC].
"""

import platform
import sys
import time
import traceback

VERSION_LEROBOT_ATTENDUE = "0.6.1"
nb_echecs = 0


def etape(titre, fonction):
    global nb_echecs
    print(f"- {titre} ... ", end="", flush=True)
    try:
        detail = fonction()
        print(f"[OK] {detail or ''}")
    except (
        Exception
    ) as erreur:  # on veut afficher toutes les erreurs, quelles qu'elles soient
        nb_echecs += 1
        print(f"[ÉCHEC] {erreur}")
        traceback.print_exc(limit=1)


def tester_python():
    if sys.version_info < (3, 12):
        raise RuntimeError(
            f"Python {platform.python_version()} trouvé, il faut Python 3.12 (environnement activé ?)"
        )
    return f"Python {platform.python_version()} sur {platform.system()} {platform.machine()}"


def tester_torch():
    import torch

    if torch.cuda.is_available():
        materiel = f"GPU NVIDIA disponible : {torch.cuda.get_device_name(0)}"
    elif torch.backends.mps.is_available():
        materiel = "puce Apple (MPS) disponible"
    else:
        materiel = "CPU uniquement (c'est suffisant pour ce TP)"
    return f"PyTorch {torch.__version__} - {materiel}"


def tester_lerobot():
    import lerobot

    if lerobot.__version__ != VERSION_LEROBOT_ATTENDUE:
        raise RuntimeError(
            f"version {lerobot.__version__} installée, le TP a été préparé avec la version {VERSION_LEROBOT_ATTENDUE}"
        )
    return f"LeRobot {lerobot.__version__}"


def tester_autres():
    import diffusers  # noqa: F401
    import gym_pusht  # noqa: F401
    import matplotlib  # noqa: F401
    import pygame  # noqa: F401
    import pymunk  # noqa: F401

    return "gym_pusht, pygame, pymunk, matplotlib, diffusers"


def tester_dataset():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("lerobot/pusht_keypoints")
    dataset[0]
    return f"{dataset.num_episodes} démonstrations téléchargées"


def tester_video():
    import tempfile
    from pathlib import Path

    import numpy as np
    from lerobot.utils.io_utils import write_video

    with tempfile.TemporaryDirectory() as dossier:
        chemin = Path(dossier) / "test.mp4"
        write_video(chemin, [np.zeros((64, 64, 3), dtype=np.uint8)] * 10, fps=10)
        if not chemin.exists():
            raise RuntimeError("la vidéo n'a pas été créée")
    return "écriture de vidéos MP4 possible"


def tester_modeles():
    from pathlib import Path

    dossier = Path(__file__).parent.parent / "modeles"
    for nom in ["diffusion_pusht_keypoints", "act_pusht_keypoints_200k"]:
        if not (dossier / nom / "model.safetensors").exists():
            raise RuntimeError(
                f"modèle {nom} absent : lancez d'abord  python scripts/recuperer_modeles.py"
            )
    return (
        "diffusion_pusht_keypoints et act_pusht_keypoints_200k présents dans modeles/"
    )


def tester_fenetre():
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from outils_pusht import NOIR, Fenetre, creer_env

    env = creer_env(500)
    fenetre = Fenetre(500, "Test de la simulation PushT")
    obs, _ = env.reset(seed=0)
    debut = time.time()
    while time.time() - debut < 5:
        if "quitter" in fenetre.evenements():
            break
        obs, _, _, _, _ = env.step(fenetre.souris())
        lignes = [
            ("Test : l'agent bleu suit votre souris.", NOIR),
            ("La fenêtre se ferme seule après 5 s.", NOIR),
        ]
        fenetre.afficher(env.render(), lignes)
        fenetre.attendre_prochain_pas()
    fenetre.fermer()
    env.close()
    return "fenêtre de simulation affichée"


if __name__ == "__main__":
    print("\nVérification de l'installation pour le TP d'apprentissage par imitation\n")
    etape("Version de Python", tester_python)
    etape("PyTorch", tester_torch)
    etape("LeRobot", tester_lerobot)
    etape("Simulation et figures", tester_autres)
    etape("Téléchargement du dataset lerobot/pusht_keypoints", tester_dataset)
    etape("Écriture de vidéos", tester_video)
    etape("Modèles de la partie 5", tester_modeles)
    etape("Fenêtre de simulation (une fenêtre va s'ouvrir 5 secondes)", tester_fenetre)
    if nb_echecs == 0:
        print("\nTout est prêt pour le TP ! Amusez-vous bien\n")
    else:
        print(
            f"\n{nb_echecs} étape(s) en échec : relisez la partie 0 de l'énoncé, ou postez le message d'erreur.\n"
        )
        sys.exit(1)
