"""Outils communs aux scripts du TP.

Ce fichier n'est pas à lancer directement : il est importé par les autres scripts.
Il regroupe :
  - la création de l'environnement de simulation PushT ;
  - une petite fenêtre pygame pour afficher la scène et lire la souris ;
  - la description des données enregistrées (format LeRobotDataset).
"""

import warnings

# pygame affiche un avertissement sans importance au démarrage : on le masque.
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

import gymnasium as gym  # noqa: E402
import gym_pusht  # noqa: E402, F401  (cet import enregistre l'environnement "gym_pusht/PushT-v0")
import numpy as np  # noqa: E402
import pygame  # noqa: E402

# La scène PushT est un carré de 512 x 512 unités (1 unité = 1 pixel à l'échelle 1).
TAILLE_SCENE = 512
# Fréquence de contrôle : l'agent reçoit 10 actions par seconde (10 Hz).
FPS = 10
# Durée maximale d'un épisode : 300 pas, soit 30 secondes (comme dans l'évaluation de LeRobot).
NB_PAS_MAX = 300
# Phrase décrivant la tâche, stockée avec chaque image du dataset.
TACHE = "Pousser le T sur la cible verte"
# Distance (en unités de la scène) en dessous de laquelle on considère que la souris "attrape" l'agent.
RAYON_ATTRAPE = 30

# Description des données enregistrées à chaque pas de temps.
# Ce sont exactement les mêmes noms et tailles que dans le dataset "lerobot/pusht_keypoints".
FEATURES = {
    # Position (x, y) de l'agent (le disque bleu).
    "observation.state": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
    # Positions (x, y) des 8 coins du T, mises à plat : 8 x 2 = 16 nombres.
    "observation.environment_state": {"dtype": "float32", "shape": (16,), "names": None},
    # Action = position (x, y) visée par l'agent (ici : la position de la souris).
    "action": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
}

HAUTEUR_BANDEAU = 80  # hauteur (en pixels) de la zone de texte sous la scène

BLANC = (255, 255, 255)
NOIR = (20, 20, 20)
GRIS = (90, 90, 90)
VERT = (30, 150, 60)
ROUGE = (200, 40, 40)
BLEU = (40, 90, 200)
ORANGE = (220, 120, 0)


def creer_env(taille_fenetre=600):
    """Crée l'environnement PushT avec des observations "positions" (sans image)."""
    return gym.make(
        "gym_pusht/PushT-v0",
        obs_type="environment_state_agent_pos",
        render_mode="rgb_array",
        visualization_width=taille_fenetre,
        visualization_height=taille_fenetre,
        max_episode_steps=100_000,  # on gère nous-mêmes la durée maximale
        disable_env_checker=True,
    )


def obs_vers_frame(obs, action):
    """Transforme une observation de l'environnement + une action en une ligne du dataset."""
    return {
        "observation.state": np.asarray(obs["agent_pos"], dtype=np.float32),
        "observation.environment_state": np.asarray(obs["environment_state"], dtype=np.float32),
        "action": np.asarray(action, dtype=np.float32),
        "task": TACHE,
    }


def ouvrir_dataset(dossier):
    """Ouvre le dataset situé dans `dossier` pour y ajouter des épisodes (le crée s'il n'existe pas)."""
    from pathlib import Path

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dossier = Path(dossier)
    # LeRobot demande un identifiant de la forme "utilisateur/nom" ; ici le dataset reste sur votre disque.
    repo_id = f"local/{dossier.name}"
    if (dossier / "meta" / "info.json").exists():
        return LeRobotDataset.resume(repo_id=repo_id, root=dossier)
    if dossier.exists() and any(dossier.iterdir()):
        raise SystemExit(
            f"Le dossier '{dossier}' existe mais ne contient pas de dataset valide. "
            "Supprimez-le ou choisissez un autre nom avec --dossier."
        )
    if dossier.exists():
        dossier.rmdir()
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=FEATURES,
        root=dossier,
        robot_type="pusht",
        use_videos=False,
    )


class Fenetre:
    """Fenêtre pygame : affiche la scène, un bandeau de texte, et lit la souris / le clavier."""

    def __init__(self, taille, titre):
        pygame.init()
        self.taille = taille
        self.ecran = pygame.display.set_mode((taille, taille + HAUTEUR_BANDEAU))
        pygame.display.set_caption(titre)
        self.police = pygame.font.Font(None, 26)
        self.grande_police = pygame.font.Font(None, 64)
        self.horloge = pygame.time.Clock()

    def souris(self):
        """Position de la souris convertie en coordonnées de la scène (entre 0 et 512)."""
        x, y = pygame.mouse.get_pos()
        position = np.array([x, y], dtype=np.float64) * TAILLE_SCENE / self.taille
        return np.clip(position, 0, TAILLE_SCENE)

    @staticmethod
    def clic_gauche_enfonce():
        return pygame.mouse.get_pressed()[0]

    @staticmethod
    def evenements():
        """Renvoie la liste des touches pressées depuis le dernier appel ("quitter" si on ferme la fenêtre)."""
        touches = []
        for evenement in pygame.event.get():
            if evenement.type == pygame.QUIT:
                touches.append("quitter")
            elif evenement.type == pygame.KEYDOWN:
                if evenement.key == pygame.K_ESCAPE:
                    touches.append("quitter")
                else:
                    touches.append(pygame.key.name(evenement.key))
        return touches

    def afficher(self, image, lignes, message=None, couleur_message=VERT, couleur_cadre=None):
        """Affiche l'image de la scène, des lignes de texte dessous et éventuellement un grand message."""
        self.ecran.fill(BLANC)
        surface = pygame.surfarray.make_surface(np.ascontiguousarray(image.swapaxes(0, 1)))
        self.ecran.blit(surface, (0, 0))
        if couleur_cadre is not None:
            pygame.draw.rect(self.ecran, couleur_cadre, (0, 0, self.taille, self.taille), width=6)
        if message is not None:
            rendu = self.grande_police.render(message, True, couleur_message)
            self.ecran.blit(rendu, rendu.get_rect(center=(self.taille // 2, self.taille // 2)))
        y = self.taille + 8
        for texte, couleur in lignes:
            self.ecran.blit(self.police.render(texte, True, couleur), (10, y))
            y += 24
        pygame.display.flip()

    def attendre_prochain_pas(self):
        """Attend ce qu'il faut pour respecter la fréquence de 10 pas par seconde."""
        self.horloge.tick(FPS)

    @staticmethod
    def fermer():
        pygame.quit()
