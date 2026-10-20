# Modèles de référence

Ces politiques ACT ont été entraînées avec LeRobot 0.6.1 sur les N premières démonstrations du dataset
[`lerobot/pusht_keypoints`](https://huggingface.co/datasets/lerobot/pusht_keypoints) (option
`--dataset.episodes`), avec les mêmes réglages que la commande d'entraînement de la partie 3.2 de l'énoncé
(`chunk_size=20`, `n_action_steps=10`, `dim_model=256`, `n_heads=4`, `dim_feedforward=1024`,
`n_encoder_layers=2`, `n_vae_encoder_layers=2`, normalisation `MEAN_STD` des entrées `STATE` et `ENV`
et des actions, `optimizer_lr=1e-4`, `batch_size=64`, graine 1000).

| Modèle | Démonstrations | Itérations |
|---|---|---|
| `act_pusht_10demos` | épisodes 0 à 9 | 5 000 |
| `act_pusht_20demos` | épisodes 0 à 19 | 5 000 |
| `act_pusht_50demos` | épisodes 0 à 49 | 5 000 |
| `act_pusht_206demos` | les 206 épisodes | 5 000 |
| `act_pusht_206demos_long` | les 206 épisodes | 30 000 |

Chaque dossier contient :

- `config.json` : la configuration de la politique (type, taille du réseau, entrées et sorties) ;
- `model.safetensors` : les poids du réseau ;
- `policy_preprocessor*` et `policy_postprocessor*` : les étapes de normalisation des observations et des actions
  (moyennes et écarts-types calculés sur les données d'entraînement).

## Modèles de la partie 5 (téléchargés par `scripts/recuperer_modeles.py`)

Ces deux modèles sont trop lourds pour être rangés dans le dépôt Git. Ils ont été entraînés dans les mêmes
conditions : les 206 démonstrations de `lerobot/pusht_keypoints`, 200 000 itérations, lots de 64, taux
d'apprentissage de départ 1e-4, paquets de 16 actions dont 8 exécutées.

| Modèle | Origine |
|---|---|
| `diffusion_pusht_keypoints` | modèle officiel [`lerobot/diffusion_pusht_keypoints`](https://huggingface.co/lerobot/diffusion_pusht_keypoints), converti au format de LeRobot 0.6.1 |
| `act_pusht_keypoints_200k` | ACT de taille d'origine (`dim_model=512`, 4 couches d'encodeur), entraîné avec LeRobot 0.6.1 ; publié dans la page *Releases* du dépôt |
