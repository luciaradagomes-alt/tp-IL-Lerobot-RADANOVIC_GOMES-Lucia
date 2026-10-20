"""Télécharger les deux politiques de la partie 5 (à lancer AVANT la séance).

  python scripts/recuperer_modeles.py

1. Diffusion Policy (environ 1 Go) : le modèle officiel lerobot/diffusion_pusht_keypoints a été publié en 2024
   avec une ancienne version de LeRobot, dont le format de fichiers n'est plus lu directement par la
   version 0.6.1. Le script :
     - télécharge ses poids depuis le Hub Hugging Face ;
     - recrée sa configuration au format actuel (mêmes réglages que lors de son entraînement) ;
     - sépare les étapes de normalisation (minimum et maximum de chaque donnée), qui étaient stockées dans le
       fichier des poids, comme le fait maintenant LeRobot.
   Résultat : modeles/diffusion_pusht_keypoints/

2. ACT entraîné dans les mêmes conditions (environ 150 Mo), téléchargé depuis la page "Releases" du dépôt GitHub
   du TP. Résultat : modeles/act_pusht_keypoints_200k/
"""

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

DEPOT_DIFFUSION = "lerobot/diffusion_pusht_keypoints"
VERSION_DIFFUSION = "58570fc39828d28efa5457aa297a52be27ac3a10"  # version figée du dépôt, la même pour tous
URL_ACT = "https://github.com/quentinRolld/tp-IL-Lerobot/releases/download/modeles-v1/act_pusht_keypoints_200k.zip"
SHA256_ACT = "51b1276e89bed39dba58261cc11c5d00593edd9d382f295194e478eb6f70e2d6"  # empreinte de l'archive, pour vérifier qu'elle n'est pas abîmée


def recuperer_diffusion(sortie):
    from huggingface_hub import hf_hub_download
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from safetensors.torch import load_file

    print(f"Téléchargement des poids de {DEPOT_DIFFUSION} (environ 1 Go, cela peut prendre quelques minutes)...")
    poids = load_file(hf_hub_download(DEPOT_DIFFUSION, "model.safetensors", revision=VERSION_DIFFUSION))

    # Configuration identique à celle utilisée pour l'entraînement du modèle officiel.
    config = DiffusionConfig(
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (2,)),
            "observation.environment_state": PolicyFeature(FeatureType.ENV, (16,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
        n_obs_steps=2,  # la politique regarde les 2 dernières observations
        horizon=16,  # elle prédit un paquet de 16 actions...
        n_action_steps=8,  # ...dont elle exécute 8 avant de prédire à nouveau
        down_dims=(512, 1024, 2048),
        kernel_size=5,
        n_groups=8,
        diffusion_step_embed_dim=128,
        use_film_scale_modulation=True,
        noise_scheduler_type="DDIM",
        num_train_timesteps=100,
        num_inference_steps=10,  # nombre d'étapes de débruitage pour produire un paquet d'actions
        beta_schedule="squaredcos_cap_v2",
        beta_start=0.0001,
        beta_end=0.02,
        prediction_type="epsilon",
        clip_sample=True,
        clip_sample_range=1.0,
        normalization_mapping={
            "STATE": NormalizationMode.MIN_MAX,
            "ENV": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        },
        device="cpu",
        push_to_hub=False,
    )
    politique = DiffusionPolicy(config)
    anciennes_cles_normalisation = ("normalize", "unnormalize")
    politique.load_state_dict(
        {cle: valeur for cle, valeur in poids.items() if not cle.startswith(anciennes_cles_normalisation)},
        strict=True,
    )

    # Les min et max de chaque donnée étaient rangés dans les poids : on les récupère.
    statistiques = {}
    for cle in ["observation.state", "observation.environment_state"]:
        nom = cle.replace(".", "_")
        statistiques[cle] = {
            "min": poids[f"normalize_inputs.buffer_{nom}.min"].numpy(),
            "max": poids[f"normalize_inputs.buffer_{nom}.max"].numpy(),
        }
    statistiques["action"] = {
        "min": poids["normalize_targets.buffer_action.min"].numpy(),
        "max": poids["normalize_targets.buffer_action.max"].numpy(),
    }
    pretraitement, posttraitement = make_pre_post_processors(config, dataset_stats=statistiques)

    politique.save_pretrained(sortie)
    pretraitement.save_pretrained(sortie)
    posttraitement.save_pretrained(sortie)
    nb_parametres = sum(p.numel() for p in politique.parameters()) / 1e6
    print(f"Diffusion Policy ({nb_parametres:.0f} millions de paramètres) enregistrée dans '{sortie}'.")


def recuperer_act(sortie, url):
    print("Téléchargement du modèle ACT (environ 150 Mo)...")
    with tempfile.TemporaryDirectory() as dossier_temporaire:
        archive = Path(dossier_temporaire) / "act.zip"
        with urllib.request.urlopen(url) as reponse, open(archive, "wb") as fichier:
            shutil.copyfileobj(reponse, fichier)
        if SHA256_ACT is not None:
            empreinte = hashlib.sha256(archive.read_bytes()).hexdigest()
            if empreinte != SHA256_ACT:
                raise SystemExit("L'archive téléchargée est abîmée : relancez le script.")
        with zipfile.ZipFile(archive) as zip_act:
            zip_act.extractall(sortie.parent)
    print(f"ACT enregistré dans '{sortie}'.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dossier", default="modeles", help="dossier où ranger les modèles")
    parser.add_argument("--url-act", default=URL_ACT, help=argparse.SUPPRESS)
    args = parser.parse_args()

    sortie_diffusion = Path(args.dossier) / "diffusion_pusht_keypoints"
    if (sortie_diffusion / "model.safetensors").exists():
        print(f"Diffusion Policy déjà présente dans '{sortie_diffusion}'.")
    else:
        recuperer_diffusion(sortie_diffusion)

    sortie_act = Path(args.dossier) / "act_pusht_keypoints_200k"
    if (sortie_act / "model.safetensors").exists():
        print(f"ACT déjà présent dans '{sortie_act}'.")
    else:
        recuperer_act(sortie_act, args.url_act)


if __name__ == "__main__":
    main()
