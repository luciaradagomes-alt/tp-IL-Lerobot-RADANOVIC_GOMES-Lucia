# TP – Apprentissage par imitation avec LeRobot

<p align="center">
  <img src="images/pusht_diffusion.gif" width="300" alt="Une politique apprise par imitation résout la tâche PushT">
</p>

**Sorbonne Université – Cours *AIHM***

*[English version](README.en.md)*

Durée de la séance : **4 heures**. La **partie 0 (installation) est à faire avant la séance**.

---

## Sommaire

- [Objectifs et déroulé](#objectifs-et-déroulé)
- [Partie 0 – Préparer sa machine (avant la séance)](#partie-0--préparer-sa-machine-avant-la-séance)
- [Partie 1 – Découvrir la tâche et les démonstrations](#partie-1--découvrir-la-tâche-et-les-démonstrations-25-min)
- [Partie 2 – Enregistrer ses propres démonstrations](#partie-2--enregistrer-ses-propres-démonstrations-30-min)
- [Partie 3 – Entraîner et évaluer une politique ACT](#partie-3--entraîner-et-évaluer-une-politique-act-55-min)
- [Partie 4 – Les limites de l'apprentissage par imitation](#partie-4--les-limites-de-lapprentissage-par-imitation-60-min)
- [Partie 5 – Une autre famille de politiques : Diffusion Policy](#partie-5--une-autre-famille-de-politiques--diffusion-policy-45-min)
- [Rendre son rapport avec GitHub](#rendre-son-rapport-avec-github)

---

## Objectifs et déroulé

L'**apprentissage par imitation** (Imitation Learning, **IL**) est une approche d’apprentissage automatique qui consiste à estimer une politique de contrôle à partir de démonstrations d’une tâche réalisées par un expert. Chaque démonstration comprend généralement une séquence d’observations de l’environnement et les actions correspondantes. Ces données constituent un jeu d’entraînement destiné à l’apprentissage d’une politique capable de reproduire le comportement observé.

Le **clonage comportemental** (Behavior Cloning, **BC**) est une méthode d’apprentissage par imitation qui formalise ce problème comme un apprentissage supervisé. À partir d’un ensemble de paires observation-action extraites des démonstrations, un modèle est entraîné à prédire l’action associée à une observation donnée. La **politique** (policy) obtenue est une fonction qui associe une observation à une action, ou à une distribution de probabilité sur les actions possibles. Dans le cas d’un agent robotique, cette politique peut ensuite être utilisée pour générer les commandes de contrôle à partir des observations disponibles.

Ce TP se fait **en simulation**, sur votre ordinateur, avec la bibliothèque **[LeRobot](https://huggingface.co/docs/lerobot)** de Hugging Face
(une bibliothèque Python libre qui regroupe des outils pour enregistrer des démonstrations, entraîner et évaluer des politiques de robots).

Objectifs du TP :

1. comprendre ce qu'est une démonstration, un dataset, une politique, et comment on les manipule avec LeRobot ;
2. enregistrer vos propres démonstrations et entraîner une politique dessus ;
3. évaluer une politique de façon rigoureuse (taux de réussite sur des situations identiques pour tous les modèles) ;
4. expliquer les principales **limites** de l'apprentissage par imitation (quantité et qualité des données,
   accumulation des erreurs, situations jamais vues) ;
5. comparer deux familles de politiques (ACT et Diffusion Policy) et comprendre d'où vient leur différence de
   performance sur cette tâche.

> **Le rapport.** Vous rédigez votre rapport **au fur et à mesure**, en Markdown (un format de texte très simple
> que GitHub affiche joliment ; c'est l'occasion d'apprendre sa syntaxe). Un modèle à compléter vous attend dans
> [rapport/RAPPORT.md](rapport/RAPPORT.md). Les questions à traiter sont signalées par **Q1**, **Q2**, etc.
> La façon de me le rendre est expliquée dans la section [Rendre son rapport avec GitHub](#rendre-son-rapport-avec-github).

### La tâche : PushT

![La tâche PushT](images/pusht.png)

PushT est une tâche de référence en apprentissage par imitation. Vu de dessus :

- le **disque bleu** est l'« agent » : c'est l'extrémité d'un robot, que l'on déplace ;
- le **T gris** est un objet posé sur une table : on ne peut le déplacer qu'en le poussant avec l'agent ;
- la **zone verte** est la cible : il faut amener le T exactement dessus.

L'épisode est **réussi** quand le T recouvre au moins **95 %** de la zone verte. Il est **raté** au bout de
**30 secondes**. La position de départ du T et de l'agent est tirée au hasard à chaque épisode.

À chaque pas de temps (10 fois par seconde) :

- l'**observation** contient la position (x, y) de l'agent, et la position (x, y) des 8 coins du T. Soit 2 + 16 = 18 nombres ;
- l'**action** est la position (x, y) **visée** par l'agent : l'agent se déplace vers ce point
  (quand vous jouez, c'est la position de votre souris).

> **Pourquoi des positions et pas des images ?** En règle générale, pour obtenir les meilleures résultats avec les méthodes SOTA, un robot recevrait les observations de la scène sous forme d'images RGB.
> Mais entraîner un réseau sur des images demande un GPU et des heures de calcul.
> Pour que le TP tienne sur un simple ordinateur portable, la politique reçoit directement les positions exactes des
> objets, comme si un système parfait de suivi d'objets les lui donnait. Les principes de l'apprentissage par
> imitation restent exactement les mêmes.

---

## Partie 0 – Préparer sa machine (avant la séance)

> ⚠️ **À faire avant la séance.** Comptez 45 min à 1 h et environ 3 Go d'espace disque. Il faut une connexion
> internet. Si vous êtes bloqué(e), vous pouvez ouvrir une issue sur git, ou vraiment au pire m'envoyer un mail (quentin.rolland2@cea.fr) **avant** la séance.
> Contact supplémentaire : Silvia Tulli – tulli@isir.upmc.fr – GitHub [`Silviatulli`](https://github.com/Silviatulli).

Ces instructions suivent la [documentation officielle d'installation de LeRobot](https://huggingface.co/docs/lerobot/installation),
adaptée à nos besoins (simulation uniquement, sans GPU). Le TP a été préparé avec **LeRobot 0.6.1** et
**Python 3.12** : utilisez exactement ces versions.

Un mot sur le vocabulaire (au cas où, on sait jamais...):

- **conda** est un gestionnaire d'**environnements Python** : un environnement est un dossier isolé qui contient
  une version de Python et des bibliothèques, sans interférer avec le reste de votre ordinateur. Nous utiliserons
  **Miniforge**, une version légère et gratuite de conda ;
- **pip** est l'outil qui installe les bibliothèques Python ;
- **Git** est un logiciel qui garde l'historique des versions de vos fichiers ; **GitHub** est un site web qui
  héberge des dépôts Git (un **dépôt**, ou *repository*, est un dossier de projet suivi par Git) ;
- **CPU** désigne le processeur principal de votre ordinateur ; **GPU** désigne une carte graphique, très efficace
  pour entraîner des réseaux de neurones. **Ce TP est conçu pour fonctionner avec un CPU seul.**

### 0.1 Créer un compte GitHub

Si vous n'en avez pas, créez un compte gratuit sur [github.com](https://github.com/signup). Il servira à me rendre
votre rapport (normalement c'est déjà fait depuis le premier TP).

### 0.2 Installer Git et Miniforge

Ouvrez la section qui correspond à votre système.

<details>
<summary><b>Windows</b></summary>

1. **Git** : téléchargez et lancez l'installateur depuis [git-scm.com/download/win](https://git-scm.com/download/win).
   Gardez toutes les options par défaut.
2. **Miniforge** : téléchargez `Miniforge3-Windows-x86_64.exe` depuis
   [conda-forge.org/download](https://conda-forge.org/download/) et lancez-le. Choisissez l'installation
   « Just Me » et gardez les autres options par défaut.
3. Dans le menu Démarrer, ouvrez **Miniforge Prompt**. **Sous Windows, toutes les commandes du TP se tapent dans
   cette fenêtre** (et pas dans l'invite de commandes classique ni dans PowerShell).

</details>

<details>
<summary><b>macOS (Mac avec puce Apple M1, M2, M3, M4… ou Mac Intel)</b></summary>

Pour savoir quelle puce vous avez : menu Pomme  > *À propos de ce Mac* (ligne « Puce » ou « Processeur »).

1. Ouvrez l'application **Terminal** (dans *Applications > Utilitaires*).
2. **Git** : tapez la commande suivante, puis acceptez l'installation des « outils de ligne de commande » :
   ```bash
   xcode-select --install
   ```
   (Si un message indique qu'ils sont déjà installés, tout va bien.)
3. **Miniforge** : tapez ces deux commandes, puis répondez `yes` aux questions (licence, et initialisation de conda) :
   ```bash
   curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
   bash Miniforge3-$(uname)-$(uname -m).sh
   ```
4. **Fermez le terminal et rouvrez-en un nouveau.** Le début de la ligne doit maintenant afficher `(base)`.

</details>

<details>
<summary><b>Linux</b></summary>

1. **Git** : `sudo apt install git` (Ubuntu, Debian) ou `sudo dnf install git` (Fedora).
2. **Miniforge** : tapez ces deux commandes, puis répondez `yes` aux questions (licence, et initialisation de conda) :
   ```bash
   wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
   bash Miniforge3-$(uname)-$(uname -m).sh
   ```
3. **Fermez le terminal et rouvrez-en un nouveau.** Le début de la ligne doit maintenant afficher `(base)`.

</details>

### 0.3 Récupérer le TP

Dans le terminal (Miniforge Prompt sous Windows), placez-vous dans le dossier où vous voulez travailler, puis
**clonez** le dépôt :

```bash
git clone https://github.com/quentinRolld/tp-IL-Lerobot.git
cd tp-IL-Lerobot
```

La commande `cd` (*change directory*) permet d'entrer dans un dossier. **Toutes les commandes du TP se lancent
depuis ce dossier `tp-IL-Lerobot`.**

### 0.4 Créer l'environnement Python et installer LeRobot

Commencez par créer et activer un environnement conda nommé `lerobot`, avec Python 3.12 (commandes identiques
sur tous les systèmes) :

```bash
conda create -y -n lerobot python=3.12
conda activate lerobot
conda install -y -c conda-forge ffmpeg
```

Le début de la ligne du terminal doit maintenant afficher `(lerobot)`. `ffmpeg` est un outil de lecture et
d'écriture de vidéos.

Installez ensuite **PyTorch**. La commande dépend
de votre système :

<details>
<summary><b>Windows</b> ou <b>Linux</b> (sans GPU NVIDIA)</summary>

```bash
pip install "torch==2.11.0" "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cpu
```

Cette version « CPU » est beaucoup plus légère (environ 200 Mo au lieu de plusieurs Go).

</details>

<details>
<summary><b>Windows</b> ou <b>Linux</b> avec un GPU NVIDIA (facultatif)</summary>

Si vous avez une carte graphique NVIDIA récente avec ses pilotes à jour, vous pouvez installer la version GPU :

```bash
pip install "torch==2.11.0" "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cu128
```

Ce n'est pas nécessaire pour le TP : en cas de problème, utilisez la version CPU ci-dessus.

</details>

<details>
<summary><b>macOS avec puce Apple</b> (M1, M2, M3, M4…)</summary>

```bash
pip install "torch==2.11.0" "torchvision==0.26.0"
```

</details>

<details>
<summary><b>macOS avec processeur Intel</b></summary>

PyTorch ne publie plus de version pip pour les Mac Intel. On l'installe donc avec conda :

```bash
conda install -y -c conda-forge "pytorch=2.11" "torchvision=0.26"
```

Je n'ai pas pu tester cette configuration sur toutes les machines : si l'installation ou le script de vérification échoue, prévenez-moi avant la séance (vous pourrez travailler en binôme).

</details>

Enfin, installez **LeRobot** avec les compléments nécessaires au TP (commande identique sur tous les systèmes) :

```bash
pip install "lerobot[pusht,training,diffusion]==0.6.1" matplotlib
```

- `pusht` ajoute la simulation PushT ;
- `training` ajoute les outils d'entraînement ;
- `diffusion` ajoute ce qu'il faut pour utiliser la politique *Diffusion Policy* de la partie 5 ;

### 0.5 Télécharger les modèles de la partie 5

La partie 5 compare deux politiques pré-entraînées qui pèsent au total un peu plus de 1 Go : téléchargez-les
**avant la séance**.

```bash
python scripts/recuperer_modeles.py
```

Le script range les deux modèles dans le dossier `modeles/` :

- `diffusion_pusht_keypoints/` : une *Diffusion Policy* publiée par l'équipe LeRobot sur le Hub Hugging Face.
  Elle a été publiée avec une ancienne version de LeRobot : le script la convertit au format de la version 0.6.1 ;
- `act_pusht_keypoints_200k/` : une politique ACT que j'ai entraînée dans les mêmes conditions, téléchargée depuis
  la page *Releases* du dépôt GitHub du TP.

### 0.6 Vérifier l'installation

```bash
python scripts/verifier_installation.py
```

Ce script teste chaque élément, télécharge le dataset de démonstrations (quelques Mo) et ouvre une fenêtre de
simulation pendant 5 secondes (l'agent bleu suit votre souris). Toutes les lignes doivent afficher `[OK]` et le
script doit se terminer par « Tout est prêt pour le TP ! Amusez-vous bien ».

### 0.7 À chaque nouvelle session de travail

Quand vous ouvrez un nouveau terminal, il faut réactiver l'environnement et revenir dans le dossier du TP :

```bash
conda activate lerobot
cd chemin/vers/tp-IL-Lerobot
```

(remplacez `chemin/vers` par l'endroit où vous avez cloné le dépôt).

---

## Partie 1 – Découvrir la tâche et les démonstrations (25 min)

### 1.1 Jouer soi-même

Lancez le script d'enregistrement en **mode entraînement** (rien n'est sauvegardé) :

```bash
python scripts/enregistrer_demos.py --entrainement
```

Placez la souris sur le disque bleu : il se met à suivre la souris. Poussez le T sur la zone verte. Le bandeau en
bas indique le pourcentage de recouvrement de la cible. `R` recommence l'épisode, `Échap` quitte.
Faites-le 3 ou 4 fois pour prendre la main.

### 1.2 Le dataset de démonstrations de référence

LeRobot fournit un **dataset** de 206 démonstrations de PushT réalisées par des humains, publié
sur le **Hub Hugging Face**.

1. Regardez quelques démonstrations en vidéo avec le visualiseur en ligne :
   [lerobot/pusht dans le visualiseur](https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Flerobot%2Fpusht%2Fepisode_0).
   (Ce dataset `lerobot/pusht` contient des images ; nous utiliserons sa version `lerobot/pusht_keypoints`,
   qui contient les mêmes démonstrations mais avec les positions des coins du T à la place des images.)
2. Lancez le script d'exploration sur le dataset de référence :
   ```bash
   python scripts/explorer_dataset.py --dataset lerobot/pusht_keypoints
   ```
   Il affiche des chiffres clés et enregistre cinq figures dans `rapport/images/pusht_keypoints/` :
   - `trajectoires.png` : le chemin suivi par l'agent dans quelques démonstrations ;
   - `positions_depart.png` : les positions de départ du T et de l'agent dans toutes les démonstrations ;
   - `durees.png` : la durée des démonstrations ;
   - `etat_et_action.png` : la position de l'agent (l'**état**) et la position visée (l'**action**) au cours d'une démonstration ;
   - `carte_de_chaleur.png` : les zones de la scène où se sont trouvés le T et l'agent pendant toutes les démonstrations
     (elle servira en partie 4.4).

Dans LeRobot, un dataset est une suite de **pas de temps** (*frames*), regroupés en **épisodes** (un épisode = une
démonstration, du début à la fin). Chaque pas de temps contient :

| Nom dans LeRobot | Taille | Signification |
|---|---|---|
| `observation.state` | 2 | position (x, y) de l'agent |
| `observation.environment_state` | 16 | positions (x, y) des 8 coins du T |
| `action` | 2 | position (x, y) visée par l'agent à ce pas de temps |
| `episode_index`, `frame_index`, `timestamp`… | 1 | numéro de l'épisode, numéro du pas dans l'épisode, temps en secondes… |

**Q1.** Combien d'épisodes et de pas de temps contient le dataset ? Quelle est la durée moyenne d'une démonstration ?
Insérez dans votre rapport la figure `etat_et_action.png` (voir le modèle de rapport pour la syntaxe Markdown d'une image).

**Q2.** Sur la figure `etat_et_action.png`, l'action et l'état sont proches mais décalés. Lequel est « en avance »
sur l'autre ? Pourquoi est-ce logique ?

**Q3.** Formulez l'apprentissage par imitation de cette tâche comme un problème d'apprentissage supervisé :
quelle est l'entrée du réseau, quelle est la sortie attendue, et d'où viennent les exemples ?

**Q4.** Regardez `positions_depart.png`. Les positions de départ couvrent-elles toute la scène ?

---

## Partie 2 – Enregistrer ses propres démonstrations (30 min)

Vous allez maintenant devenir l'expert et enregistrer **20 démonstrations réussies** :

```bash
python scripts/enregistrer_demos.py --nb-demos 20
```

- l'enregistrement d'un épisode démarre quand la souris touche le disque bleu (un cadre rouge apparaît) ;
- dès que le T recouvre 95 % de la cible, la démonstration est **sauvegardée automatiquement** et l'épisode suivant commence ;
- au bout de 30 s sans réussite, l'épisode est abandonné et **n'est pas sauvegardé** ;
- `R` recommence l'épisode en cours (il n'est pas sauvegardé), `Échap` quitte (les démonstrations déjà réussies sont gardées).

Vos démonstrations sont enregistrées au format LeRobot dans le dossier `donnees/mes_demos/`. Si vous quittez avant
la fin, relancez la même commande : les nouvelles démonstrations s'ajoutent aux anciennes.

> **Conseils pour de bonnes démonstrations.** Le réseau va imiter **tout** ce que vous faites, y compris vos
> hésitations. Essayez d'être régulier(ère) : des gestes calmes, une stratégie cohérente d'une démonstration à l'autre,
> sans pause inutile.

Une fois terminé, explorez vos propres données :

```bash
python scripts/explorer_dataset.py --dossier donnees/mes_demos
```

Les figures sont enregistrées dans `rapport/images/mes_demos/`.

**Q5.** Comparez vos démonstrations à celles du dataset de référence (durée moyenne, forme des trajectoires), j'attends un tableau de comparaison de statistiques avec les données pertinentes.
Insérez le tableau et votre figure `trajectoires.png` dans le rapport.

**Q6.** Combien de temps vous a pris l'enregistrement de 20 démonstrations, et combien de tentatives avez-vous ratées ? Qu'en concluez-vous sur le coût de la collecte de données pour un vrai robot, où il en faudrait des centaines ?

---

## Partie 3 – Entraîner et évaluer une politique ACT (55 min)

### 3.1 La politique ACT en bref

Nous utilisons **ACT** (*Action Chunking with Transformers*, [Zhao et al., 2023](https://arxiv.org/abs/2304.13705)),
une politique très utilisée en robotique, et disponible dans LeRobot. Ses deux idées principales :

- **Prédire des séquences d'actions** (*action chunking*). Au lieu de prédire une seule action à la fois, le réseau
  prédit d'un coup les `chunk_size` prochaines actions (ici 20 actions, soit 2 secondes). La politique en exécute
  ensuite `n_action_steps` (ici 10, soit 1 seconde) avant de refaire une prédiction à partir de la nouvelle observation. 
- **Un réseau de type Transformer, entraîné comme un CVAE.** 
  Un **CVAE** (*Conditional Variational AutoEncoder*) ajoute une petite variable cachée, appelée **variable latente**, qui résume le « style » de la démonstration (par exemple : contourner
  le T par la gauche ou par la droite). Pendant l'entraînement, un second réseau (l'encodeur) calcule cette variable à partir des vraies actions de l'expert ; au moment d'utiliser la politique, on la fixe à zéro (le style « moyen »).

La fonction de **perte** (*loss*, la quantité que l'entraînement cherche à rendre la plus petite possible) est la somme de deux termes :

- `l1_loss` : l'écart moyen (en valeur absolue) entre les actions prédites et celles de l'expert ;
- `kld_loss` : la **divergence de Kullback-Leibler** (KL), qui mesure à quel point la variable latente s'écarte d'une loi normale ; elle empêche cette variable de « tricher » en recopiant les réponses.

Pour que l'entraînement tienne sur un CPU, nous utilisons une version d'ACT environ **10 fois plus petite** que l'originale (4 millions de paramètres au lieu de 40 millions).

### 3.2 Lancer l'entraînement

L'entraînement dure **environ 15 à 25 minutes** sur un ordi récent. **Lancez-le tout de suite**, puis lisez la
suite et répondez aux questions pendant qu'il tourne.

Sur **macOS et Linux** :

```bash
lerobot-train \
  --dataset.repo_id=local/mes_demos \
  --dataset.root=donnees/mes_demos \
  --policy.type=act \
  --policy.device=cpu \
  --policy.push_to_hub=false \
  --policy.chunk_size=20 \
  --policy.n_action_steps=10 \
  --policy.dim_model=256 \
  --policy.n_heads=4 \
  --policy.dim_feedforward=1024 \
  --policy.n_encoder_layers=2 \
  --policy.n_vae_encoder_layers=2 \
  "--policy.normalization_mapping={STATE: MEAN_STD, ENV: MEAN_STD, ACTION: MEAN_STD}" \
  --policy.optimizer_lr=1e-4 \
  --steps=5000 \
  --batch_size=64 \
  --save_freq=1000 \
  --log_freq=200 \
  --num_workers=2 \
  --wandb.enable=false \
  --output_dir=outputs/train/act_mes_demos
```

<details>
<summary>Sur <b>Windows</b> (Miniforge Prompt) : même commande, avec <code>^</code> à la place de <code>\</code></summary>

```bat
lerobot-train ^
  --dataset.repo_id=local/mes_demos ^
  --dataset.root=donnees/mes_demos ^
  --policy.type=act ^
  --policy.device=cpu ^
  --policy.push_to_hub=false ^
  --policy.chunk_size=20 ^
  --policy.n_action_steps=10 ^
  --policy.dim_model=256 ^
  --policy.n_heads=4 ^
  --policy.dim_feedforward=1024 ^
  --policy.n_encoder_layers=2 ^
  --policy.n_vae_encoder_layers=2 ^
  "--policy.normalization_mapping={STATE: MEAN_STD, ENV: MEAN_STD, ACTION: MEAN_STD}" ^
  --policy.optimizer_lr=1e-4 ^
  --steps=5000 ^
  --batch_size=64 ^
  --save_freq=1000 ^
  --log_freq=200 ^
  --num_workers=2 ^
  --wandb.enable=false ^
  --output_dir=outputs/train/act_mes_demos
```

</details>

Signification des options :

| Option | Signification |
|---|---|
| `--dataset.repo_id`, `--dataset.root` | le nom du dataset et le dossier où il se trouve sur votre disque |
| `--policy.type=act` | le type de politique à entraîner |
| `--policy.device=cpu` | entraîner sur le processeur (CPU). Avec un GPU NVIDIA : `cuda` ; sur Mac à puce Apple, vous pouvez essayer `mps` |
| `--policy.push_to_hub=false` | ne pas envoyer le modèle sur le Hub Hugging Face |
| `--policy.chunk_size=20` | nombre d'actions prédites d'un coup (20 actions = 2 s) |
| `--policy.n_action_steps=10` | nombre d'actions exécutées avant de refaire une prédiction (10 actions = 1 s) |
| `--policy.dim_model`, `n_heads`, `dim_feedforward`, `n_encoder_layers`, `n_vae_encoder_layers` | la taille du réseau (ici réduite pour le CPU) |
| `--policy.normalization_mapping` | la façon de **normaliser** chaque type de donnée avant de la donner au réseau. `MEAN_STD` : on retire la moyenne et on divise par l'écart-type (calculés sur le dataset), pour que toutes les entrées soient de l'ordre de 1. `STATE` désigne la position de l'agent, `ENV` les coordonnées des coins du T, `ACTION` les actions. Par défaut, LeRobot ne normalise pas les entrées `ENV` pour ACT : on le demande explicitement |
| `--policy.optimizer_lr=1e-4` | le **taux d'apprentissage** (*learning rate*) : la taille des pas faits pour corriger les poids du réseau à chaque itération |
| `--steps=5000` | le nombre d'**itérations** (*steps*) : à chaque itération, le réseau voit un lot d'exemples et corrige ses poids |
| `--batch_size=64` | la taille d'un **lot** (*batch*) : le nombre d'exemples (pas de temps) utilisés à chaque itération |
| `--save_freq=1000` | enregistrer une sauvegarde du modèle (un ***checkpoint***) toutes les 1000 itérations |
| `--log_freq=200` | afficher la perte toutes les 200 itérations |
| `--num_workers=2` | nombre de processus qui préparent les données en parallèle |
| `--wandb.enable=false` | désactiver Weights & Biases (un service en ligne de suivi des entraînements) |
| `--output_dir` | le dossier où sont rangés les résultats |

Toutes les 200 itérations, une ligne comme celle-ci s'affiche :

```
step:1K smpl:64K ep:493 epch:24.63 loss:0.297 grdn:6.303 lr:1.0e-04 updt_s:0.264 data_s:0.009 smp/s:235 l1_loss:0.287 kld_loss:0.001
```

`step` est le numéro de l'itération, `smpl` le nombre d'exemples vus, `epch` le nombre d'**époques** (une époque =
le réseau a vu, en moyenne, une fois chaque exemple du dataset), `loss` la perte totale, `grdn` la taille du
gradient (la direction de correction des poids), `lr` le taux d'apprentissage, `updt_s` le temps de calcul d'une
itération en secondes. Notez quelques valeurs de `loss`, `l1_loss` et `kld_loss` au fil de l'entraînement
pour la question Q9.

Les modèles sont rangés dans `outputs/train/act_mes_demos/checkpoints/` : un sous-dossier par sauvegarde
(`001000`, `002000`, …) et `last` pour la dernière.

**Q7.** Avec 20 démonstrations, combien de pas de temps contient votre dataset ? Avec `--batch_size=64` et
`--steps=5000`, combien de fois le réseau a-t-il vu chaque exemple en moyenne (c'est-à-dire combien d'époques) ?
Comparez avec le nombre d'époques indiqué à la fin de l'entraînement.

**Q8.** Avec `chunk_size=20` et `n_action_steps=10`, combien de fois par seconde la politique fait-elle une prédiction ? Quel avantage et quel inconvénient voyez-vous à exécuter plusieurs actions
« les yeux fermés » (dans le sens, exécuter une séquence d'actions d'un coup  au lieu d'une action à la fois) ?

**Q9.** Comment évoluent `l1_loss` et `kld_loss` pendant l'entraînement ? Une perte très faible à la fin
garantit-elle que la politique réussira la tâche ? (Vous pourrez répondre complètement après la partie 3.3.)
Pour aller plus loin : `kld_loss` devient presque nul ; d'après la description de la partie 3.1, que peut-on en
déduire sur l'utilisation que fait le réseau de sa variable latente ?

### 3.3 Évaluer la politique

En attendant la fin de l'entraînement, vous pouvez déjà regarder jouer un modèle que j'ai entraîné plus longtemps en avance sur les 206 démonstrations de référence (voir partie 4) :

```bash
python scripts/tester_politique.py --politique modeles/act_pusht_206demos_long
```

Une fois **votre** entraînement terminé, évaluez votre politique sur **50 épisodes** :

```bash
lerobot-eval --policy.path=outputs/train/act_mes_demos/checkpoints/last/pretrained_model --env.type=pusht --env.obs_type=environment_state_agent_pos --eval.n_episodes=50 --eval.batch_size=10 --eval.use_async_envs=false --policy.device=cpu --output_dir=outputs/eval/act_mes_demos
```

| Option | Signification |
|---|---|
| `--policy.path` | le dossier du modèle à évaluer |
| `--env.type=pusht`, `--env.obs_type=environment_state_agent_pos` | la simulation à utiliser, avec des observations sous forme de positions |
| `--eval.n_episodes=50` | nombre d'épisodes joués |
| `--eval.batch_size=10` | nombre de simulations jouées en même temps |
| `--eval.use_async_envs=false` | jouer les simulations dans un seul processus (nécessaire avec PushT dans cette version de LeRobot) |
| `--output_dir` | dossier où sont écrits les résultats et les vidéos |

Cette commande ne fonctionne qu'en une seule ligne : elle est identique sur tous les systèmes.

L'évaluation prend 1 à 2 minutes. Elle affiche puis enregistre dans `outputs/eval/act_mes_demos/eval_info.json` :

- `pc_success` : le **taux de réussite**, en % des épisodes ;
- `avg_max_reward` : le **score moyen**. Pour chaque épisode, on garde le meilleur recouvrement atteint divisé par
  95 % (donc 1 = réussite), puis on fait la moyenne sur les épisodes. Ce score montre les progrès même quand la
  politique ne réussit pas complètement.

> **Ne soyez pas surpris(e) par des taux de réussite faibles.** Le critère de réussite de PushT (95 % de
> recouvrement) est très exigeant : il faut placer le T presque au pixel près. Avec quelques minutes d'entraînement
> sur CPU, un taux de réussite proche de 0 % est normal. Comparez surtout les **scores** et le pourcentage
> d'épisodes ayant atteint au moins 80 % de recouvrement (donné par le script `tableau_resultats.py`, partie 4).

Les positions de départ des 50 épisodes sont tirées avec une **graine aléatoire** fixe (un nombre qui détermine
la suite de tirages « au hasard ») : tous les modèles évalués avec cette commande rencontrent exactement les mêmes 50 situations, ce qui rend la comparaison équitable (c'est hyper important).

Les vidéos des 10 premiers épisodes sont dans `outputs/eval/act_mes_demos/videos/pusht_0/`. Pour en mettre une dans
votre rapport, convertissez-la en GIF animé (GitHub affiche les GIF directement dans un fichier Markdown) :

```bash
python scripts/video_vers_gif.py outputs/eval/act_mes_demos/videos/pusht_0/eval_episode_0.mp4 rapport/images/mon_modele.gif
```

Regardez enfin votre politique interagir en direct :

```bash
python scripts/tester_politique.py --politique outputs/train/act_mes_demos/checkpoints/last/pretrained_model
```

La croix rouge montre le point visé par l'action en cours. `Espace` passe à l'épisode suivant, `P` met en pause,
`Échap` quitte.

**Q10.** Donnez le taux de réussite et le score moyen de votre politique. Insérez un GIF d'un épisode réussi
(ou à défaut, d'un épisode raté) et décrivez le comportement de la politique : imite-t-elle votre façon d'agir ?

**Q11.** (Facultatif, si vous avez le temps.) Évaluez aussi le checkpoint `001000` (remplacez `last` par `001000`
dans `--policy.path`, et changez `--output_dir`). Le modèle s'améliore-t-il entre 1000 et 5000 itérations ?

---

## Partie 4 – Les limites de l'apprentissage par imitation (60 min)

Le dossier [modeles/](modeles/) contient des politiques ACT que j'ai entraînées sur les **N premières
démonstrations** du dataset de référence `lerobot/pusht_keypoints`. Les quatre premières ont **exactement les
mêmes réglages que vous** (même taille de réseau, 5000 itérations). La dernière a été entraînée 6 fois plus
longtemps, sur GPU :

| Modèle | Démonstrations utilisées | Itérations |
|---|---|---|
| `modeles/act_pusht_10demos` | 10 | 5 000 |
| `modeles/act_pusht_20demos` | 20 | 5 000 |
| `modeles/act_pusht_50demos` | 50 | 5 000 |
| `modeles/act_pusht_206demos` | 206 (tout le dataset) | 5 000 |
| `modeles/act_pusht_206demos_long` | 206 (tout le dataset) | 30 000 |

### 4.1 La quantité de données

Évaluez les cinq modèles avec la même commande qu'en partie 3.3, en changeant `--policy.path` et `--output_dir`.
Par exemple, pour le premier :

```bash
lerobot-eval --policy.path=modeles/act_pusht_10demos --env.type=pusht --env.obs_type=environment_state_agent_pos --eval.n_episodes=50 --eval.batch_size=10 --eval.use_async_envs=false --policy.device=cpu --output_dir=outputs/eval/act_pusht_10demos
```

Puis rassemblez tous vos résultats dans un tableau Markdown, prêt à copier dans votre rapport :

```bash
python scripts/tableau_resultats.py outputs/eval/act_pusht_10demos outputs/eval/act_pusht_20demos outputs/eval/act_pusht_50demos outputs/eval/act_pusht_206demos outputs/eval/act_pusht_206demos_long outputs/eval/act_mes_demos
```

**Q12.** Copiez le tableau dans votre rapport. Comment évoluent le taux de réussite, le recouvrement ≥ 80 % et le
score moyen avec le nombre de démonstrations ? Et avec la durée d'entraînement (comparez `act_pusht_206demos` et
`act_pusht_206demos_long`) ? Même le meilleur modèle est loin d'une politique fiable : à votre avis, que faudrait-il
changer (données, modèle, entraînement) pour s'en approcher ?

### 4.2 La qualité et la cohérence des démonstrations

**Q13.** Comparez **votre** modèle (20 démonstrations) au modèle `act_pusht_20demos` (20 démonstrations de
référence). Lequel est le meilleur ? Proposez des explications (régularité des gestes, durée des démonstrations,
stratégie, diversité des positions de départ…). Appuyez-vous sur les figures de `explorer_dataset.py`.

**Q14.** Si vous deviez enregistrer 20 démonstrations supplémentaires pour améliorer votre modèle, comment vous y prendriez-vous (façon de jouer, situations de départ à privilégier…) ? Justifiez à l'aide de vos figures.

### 4.3 L'accumulation des erreurs

Pendant l'entraînement, la politique n'a vu que des situations rencontrées par l'expert. Mais à l'inférence, elle fait de petites erreurs, qui la mènent dans des situations un peu différentes, où elle se trompe un peu plus,
etc. Les erreurs s'accumulent (en anglais *compounding errors*). On parle aussi de **décalage de distribution** (***covariate shift***) : les situations rencontrées en test sortent de la distribution des données d'entraînement.

Testez la robustesse du modèle `act_pusht_206demos_long` en le **perturbant** : pendant qu'il joue, maintenez le clic
gauche pour prendre le contrôle de l'agent, emmenez-le ailleurs (ou déplacez le T), puis relâchez le clic pour lui
rendre la main.

```bash
python scripts/tester_politique.py --politique modeles/act_pusht_206demos_long
```

**Q15.** Décrivez 2 ou 3 perturbations testées par vos soins. La politique arrive-t-elle à se rattraper ? Dans quels cas échoue-t-elle ? Interprétation ?

### 4.4 Les situations jamais vues

Pour savoir quelles situations la politique n'a **jamais vues**, regardez la **carte de chaleur** des positions
visitées pendant les 206 démonstrations de référence (figure `carte_de_chaleur.png` produite par
`explorer_dataset.py` en partie 1). La scène est découpée en cases de 16 × 16 unités ; la couleur indique combien
de pas de temps le centre du T (à gauche) ou l'agent (à droite) a passé dans chaque case. Les cases **blanches**
n'ont jamais été visitées : un T placé là est **hors distribution** (en dehors de la zone couverte par les données
d'entraînement).

![Carte de chaleur des positions visitées dans lerobot/pusht_keypoints](images/carte_de_chaleur_pusht.png)

Les croix rouges correspondent aux trois centres du T testés ci-dessous : (300, 345), au cœur des données, et
(105, 93) et (393, 93), dans le blanc.

L'option `--etat-initial` de `tester_politique.py` impose la position de départ : position (x, y) de l'agent,
position (x, y) du **centre du T** (comme sur la carte), puis angle du T en radians. Rappel : la scène va de 0 à 512
dans chaque direction, et (0, 0) est le coin en haut à gauche.

Comparez une situation **semblable** aux démonstrations :

```bash
python scripts/tester_politique.py --politique modeles/act_pusht_206demos_long --etat-initial 100 100 300 345 0
```

à des situations **jamais vues** pendant l'entraînement (agent dans un coin de la scène, T près du bord
supérieur, dans une zone que le centre du T n'a jamais visitée) :

```bash
python scripts/tester_politique.py --politique modeles/act_pusht_206demos_long --etat-initial 492 492 105 93 -2.66
python scripts/tester_politique.py --politique modeles/act_pusht_206demos_long --etat-initial 20 492 393 93 2.19
```

Pour choisir vos propres situations, vous pouvez placer des croix sur la carte avec l'option `--marquer X Y`
(répétable), puis vérifier dans quelle zone elles tombent :

```bash
python scripts/explorer_dataset.py --dataset lerobot/pusht_keypoints --marquer 256 120 --marquer 450 450
```

Attention : le T mesure 120 unités de large et de haut. Si son centre est trop près d'un mur, il le chevauche et la
simulation le repousse brutalement au début de l'épisode.

**Q16.** Comment se comporte la politique dans ces différentes situations ? Interprétation ?

---

## Partie 5 – Une autre famille de politiques : Diffusion Policy (45 min)

Même le meilleur ACT de la partie 4 reste loin d'une politique fiable. Nous le comparons ici à une autre classe de
politiques, **Diffusion Policy** ([Chi et al., 2023](https://arxiv.org/abs/2303.04137)), qui repose sur les modèles
de diffusion ([Ho et al., 2020](https://arxiv.org/abs/2006.11239)).

### 5.1 Cadre formel

**Clonage comportemental.** Notons $o_t$ l'observation à l'instant $t$ et
$A_t = (a_t, a_{t+1}, \dots, a_{t+H-1})$ le chunk des $H$ actions suivantes de l'expert ($H$ est l'**horizon** de
prédiction ; ici $H = 16$, et chaque $a_t \in \mathbb{R}^2$). Les démonstrations fournissent des couples
$(o_t, A_t)$, considérés comme tirés d'une loi inconnue. Pour une observation $o$ donnée, l'expert ne réagit pas
toujours de la même façon : ses actions suivent une **loi conditionnelle** $p_E(A \mid o)$ (« loi de $A$ sachant
$o$ »). Apprendre une politique revient à approcher cette loi conditionnelle.

**ACT.** Comme vu en partie 3.1, ACT est un **auto-encodeur variationnel conditionnel** (CVAE). Pendant
l'entraînement, un encodeur $q_\phi(z \mid A, o)$ résume le chunk d'actions de l'expert en une variable latente
$z \in \mathbb{R}^{32}$, et un décodeur $f_\theta(o, z)$ reconstruit le chunk d'actions à partir de $o$ et de $z$
($\theta$ et $\phi$ désignent les poids des deux réseaux). La perte minimisée est :

$$
\mathcal{L}_{\text{ACT}}(\theta, \phi) = \mathbb{E}\Big[\, \big\| f_\theta(o, z) - A \big\|_1 \Big]
\;+\; \beta \; \mathbb{E}\Big[ D_{\text{KL}}\big(q_\phi(z \mid A, o) \,\big\|\, \mathcal{N}(0, I)\big) \Big],
\qquad z \sim q_\phi(z \mid A, o)
$$

où $\Vert\cdot\Vert_1$ est la somme des valeurs absolues (le terme `l1_loss`), $D_{\text{KL}}$ la divergence de Kullback-Leibler (le terme `kld_loss`), $\mathcal{N}(0, I)$ la loi normale centrée réduite, choisie comme **loi a priori** de $z$, et $\beta = 10$ un coefficient de pondération. À l'utilisation, l'expert n'est plus là pour
fournir $A$ : on fixe $z$ à la moyenne de la loi a priori, $z = 0$, et la politique renvoie
$\hat{A} = f_\theta(o, 0)$.

**Diffusion Policy.** Diffusion Policy est un **modèle génératif** de la loi conditionnelle $p_E(A \mid o)$,
c'est-à-dire un modèle dont on peut tirer des échantillons $A$ pour une observation $o$ donnée. Il repose sur deux
processus :

- un **processus de bruitage** (utilisé uniquement à l'entraînement) : à partir d'un chunk d'actions de l'expert
  $A^0 = A$, on fabrique des versions de plus en plus bruitées
  $A^k = \sqrt{\bar{\alpha}_k}\, A^0 + \sqrt{1 - \bar{\alpha}_k}\; \varepsilon$, avec
  $\varepsilon \sim \mathcal{N}(0, I)$ et $k = 1, \dots, K$ ($K = 100$). Les coefficients
  $\bar{\alpha}_k \in (0, 1)$ décroissent avec $k$, de sorte que $A^K$ est presque du bruit pur ;
- un réseau $\varepsilon_\theta(A^k, k, o)$ entraîné à **retrouver le bruit** $\varepsilon$ qui a été ajouté,
  en minimisant l'erreur quadratique :

$$
\mathcal{L}_{\text{DP}}(\theta) = \mathbb{E}_{(o, A),\, k,\, \varepsilon}\Big[ \big\| \varepsilon - \varepsilon_\theta(A^k, k, o) \big\|_2^2 \Big]
$$

À l'inférence, on **tire** $A^K \sim \mathcal{N}(0, I)$, puis on applique itérativement un pas de débruitage
$A^{k-1} = g\big(A^k, \varepsilon_\theta(A^k, k, o)\big)$ jusqu'à obtenir $A^0$, le chunk d'actions exécuté. Ici, la
règle d'échantillonnage DDIM ([Song et al., 2021](https://arxiv.org/abs/2010.02502)) permet de n'effectuer que
**10 pas de débruitage** au lieu de 100. Chaque pas nécessite une évaluation complète du réseau.

### 5.2 Protocole expérimental

Les deux politiques comparées ont été entraînées sur les mêmes données et avec le même budget : les 206
démonstrations de `lerobot/pusht_keypoints`, les mêmes observations (position de l'agent et des 8 coins du T),
200 000 itérations sur des lots de 64 exemples, un taux d'apprentissage initial de $10^{-4}$, un horizon
$H = 16$ dont 8 actions sont exécutées avant une nouvelle prédiction, et des entrées comme des sorties
normalisées (par minimum et maximum pour Diffusion Policy, par moyenne et écart-type pour ACT).

| Modèle | Type | Nombre de paramètres | Origine |
|---|---|---|---|
| `modeles/act_pusht_keypoints_200k` | ACT (taille d'origine) | 40 millions | entraîné par mes soins avec le protocole ci-dessus (téléchargé en partie 0) |
| `modeles/diffusion_pusht_keypoints` | Diffusion Policy | 249 millions | modèle publié par l'équipe LeRobot (téléchargé en partie 0) |

Les deux modèles diffèrent néanmoins sur trois points, à garder en tête lors de l'interprétation des résultats :
Diffusion Policy est conditionnée par les **deux** dernières observations $(o_{t-1}, o_t)$ alors qu'ACT ne
reçoit que $o_t$ ; son taux d'apprentissage décroît au cours de l'entraînement (schéma « cosinus »), alors que
celui d'ACT reste constant (réglage par défaut de LeRobot) ; enfin, elle compte six fois plus de paramètres.

### 5.3 Évaluation quantitative

Lancez d'abord l'évaluation de Diffusion Policy, **lente sur CPU** (5 à 15 minutes selon votre machine), et lisez
la suite pendant qu'elle s'exécute :

```bash
lerobot-eval --policy.path=modeles/diffusion_pusht_keypoints --env.type=pusht --env.obs_type=environment_state_agent_pos --eval.n_episodes=50 --eval.batch_size=10 --eval.use_async_envs=false --policy.device=cpu --output_dir=outputs/eval/diffusion_pusht_keypoints
```

Puis celle d'ACT (1 à 2 minutes) et le tableau comparatif :

```bash
lerobot-eval --policy.path=modeles/act_pusht_keypoints_200k --env.type=pusht --env.obs_type=environment_state_agent_pos --eval.n_episodes=50 --eval.batch_size=10 --eval.use_async_envs=false --policy.device=cpu --output_dir=outputs/eval/act_pusht_keypoints_200k
```

```bash
python scripts/tableau_resultats.py outputs/eval/act_pusht_keypoints_200k outputs/eval/diffusion_pusht_keypoints
```

**Q17.** Reportez le tableau dans votre rapport et comparez les deux politiques. Relevez la durée de chaque
évaluation. À partir de la section 5.1, exprimez le nombre d'évaluations du réseau nécessaires pour produire un
chunk d'actions avec chacune des deux politiques, et commentez les conséquences pour une commande en temps réel
d'un robot. Insérez un GIF d'un épisode de chaque politique (voir `video_vers_gif.py`, partie 3.3).

### 5.4 Analyse dans une situation donnée

Le script suivant fixe une situation initiale $o$ (l'agent sous le T, qui peut être contourné par la
gauche ou par la droite) et produit deux figures dans
`rapport/images/comparaison/` :

- `predictions.png` : pour **la même observation** $o$, chaque politique est interrogée **20 fois** ; chaque courbe
  orange représente un chunk d'actions prédit $\hat{A}$, c'est-à-dire la suite des positions visées par l'agent
  sur environ 1,5 seconde ;
- `episodes.png` : **8 épisodes complets** sont joués par chaque politique depuis cette même situation initiale ;
  on trace la trajectoire de l'agent pendant les 4 premières secondes.

```bash
python scripts/comparer_politiques.py --politiques modeles/act_pusht_keypoints_200k modeles/diffusion_pusht_keypoints
```

Le script prend quelques minutes, du fait de Diffusion Policy.

**Q18.** Insérez les deux figures dans votre rapport et décrivez `predictions.png` pour chaque politique. En vous
appuyant sur la section 5.1, expliquez pourquoi les 20 prédictions d'ACT sont confondues, et identifiez précisément
la source de la variabilité des prédictions de Diffusion Policy.

**Q19.** Combien de **modes** (groupes distincts de trajectoires) distinguez-vous dans l'ensemble des prédictions de
Diffusion Policy ? Chacun correspond-il à une stratégie admissible pour amener le T sur la cible ? Lequel de ces
modes la prédiction d'ACT suit-elle ?

**Q20.** Analyse théorique.

1. Supposons que, dans la situation étudiée, l'expert contourne le T une fois sur deux par la gauche (chunk
   $A_{\text{gauche}}$) et une fois sur deux par la droite (chunk $A_{\text{droite}}$) :
   $p_E(A \mid o) = \tfrac{1}{2}\,\delta_{A_{\text{gauche}}} + \tfrac{1}{2}\,\delta_{A_{\text{droite}}}$, où $\delta_x$
   désigne la loi qui donne la valeur $x$ avec probabilité 1. Une loi présentant plusieurs modes est dite
   **multimodale**. Un prédicteur déterministe $f$ associe un unique chunk $f(o)$ à chaque observation $o$ : quelle
   part de $p_E(A \mid o)$ peut-il reproduire ? Qu'en résulte-t-il pour la diversité des comportements de la
   politique, d'un épisode à l'autre, depuis une même situation ?
2. En principe, la variable latente $z$ du CVAE pourrait indiquer quel mode suivre. En vous appuyant sur
   l'évolution de `kld_loss` observée en Q9 (la divergence $D_{\text{KL}}$ tend vers 0) et sur le fait qu'on fixe
   $z = 0$ à l'utilisation, expliquez pourquoi ACT se comporte ici comme un prédicteur déterministe
   $o \mapsto f_\theta(o)$. Que peut-on en déduire sur l'information que $z$ transmet au décodeur ?
3. Expliquez pourquoi le processus d'échantillonnage de Diffusion Policy permet au contraire de représenter tous les
   modes de $p_E(A \mid o)$.

**Q21.** Comparez, sur `episodes.png`, la variabilité des trajectoires d'un épisode à l'autre pour les deux
politiques. Le caractère aléatoire de Diffusion Policy constitue-t-il un défaut ? Relancez ensuite le script depuis
plusieurs situations initiales de votre choix (option `--etat-initial`, même format qu'en partie 4.4). Les
prédictions de Diffusion Policy présentent-elles systématiquement plusieurs modes ?
Enfin, sur 200 situations initiales tirées au hasard, l'avantage de Diffusion Policy sur ACT est du même ordre dans les situations où elle prédit plusieurs modes (62 % de réussite pour ACT contre 82 %) et dans les autres (53 % contre 78 %). La multimodalité suffit-elle à expliquer l'écart de performance mesuré en 5.3 ?
Quelles autres différences entre les deux politiques (section 5.2) pourraient y contribuer, et quelle expérience permettrait de départager ces hypothèses ?

**Q22.** Conclusion : en vous appuyant sur l'ensemble du TP, résumez en 5 à 10 lignes ce que l'apprentissage par imitation fait bien, ses principales limites, et ce que le choix de la classe de politique peut (ou ne peut pas) corriger, en mobilisant la notion de multimodalité.

---

## Rendre son rapport avec GitHub

Le rapport est le fichier `rapport/RAPPORT.md`, accompagné de ses images dans `rapport/images/`. Pour me le rendre,
vous allez créer **votre propre dépôt GitHub privé** (seuls vous et moi pourrons le voir), y envoyer votre travail,
puis m'y inviter.

### Étape 1 – Créer votre dépôt privé sur GitHub

1. Connectez-vous sur [github.com](https://github.com), cliquez sur **+** (en haut à droite) puis **New repository**.
2. *Repository name* : `tp-IL-Lerobot-NOM-Prenom` (avec votre nom et votre prénom).
3. Cochez **Private**.
4. **Ne cochez rien d'autre** (pas de README, pas de .gitignore, pas de licence) : le dépôt doit être vide.
5. Cliquez sur **Create repository**. Gardez la page ouverte : elle affiche l'adresse de votre dépôt, de la forme
   `https://github.com/VOTRE-PSEUDO/tp-IL-Lerobot-NOM-Prenom.git`.

### Étape 2 – Relier votre dossier local à ce dépôt

Dans le terminal, depuis le dossier `tp-IL-Lerobot` (remplacez l'adresse par la vôtre) :

```bash
git remote rename origin prof
git remote add origin https://github.com/VOTRE-PSEUDO/tp-IL-Lerobot-NOM-Prenom.git
```

Votre dossier connaît maintenant deux dépôts distants (*remotes*) : `prof` (le mien, d'où vient l'énoncé) et
`origin` (le vôtre, où vous rendez votre travail). Si je corrige l'énoncé pendant le TP, vous pourrez récupérer
la mise à jour avec `git pull prof main`.

Indiquez ensuite à Git votre nom et votre adresse e-mail (une seule fois par ordinateur) :

```bash
git config --global user.name "Prénom Nom"
git config --global user.email "votre.adresse@etu.sorbonne-universite.fr"
```

### Étape 3 – Envoyer votre rapport

Chaque fois que vous voulez enregistrer une version de votre travail :

```bash
git add rapport
git commit -m "Rapport : parties 1 à 3"
git push -u origin main
```

- `git add rapport` : sélectionne le dossier `rapport` (texte et images) pour la prochaine version ;
- `git commit -m "..."` : crée une version (un ***commit***) avec un message qui la décrit ;
- `git push` : envoie vos versions sur GitHub (`-u origin main` n'est utile que la première fois ; ensuite,
  `git push` suffit).

> **Identification.** Lors du premier `git push`, GitHub vous demande de vous identifier. Sous Windows et macOS,
> une fenêtre de connexion s'ouvre en général toute seule : connectez-vous avec votre compte GitHub.
> Si Git vous demande un mot de passe dans le terminal, votre mot de passe GitHub habituel ne fonctionnera pas :
> il faut un **jeton d'accès** (*personal access token*). Créez-en un sur
> [github.com/settings/tokens](https://github.com/settings/tokens) (*Generate new token (classic)*, cochez la case
> `repo`), et collez-le à la place du mot de passe.

Les dossiers `donnees/` et `outputs/` (démonstrations, modèles, vidéos) sont volontairement ignorés par Git
(voir le fichier `.gitignore`) : ils sont trop lourds et je n'en ai pas besoin. Seul le dossier `rapport/` m'intéresse.

### Étape 4 – M'inviter sur votre dépôt

Sur la page de votre dépôt GitHub : **Settings** > **Collaborators** > **Add people**, et ajoutez les comptes
**`quentinRolld`** et **`Silviatulli`**.

Vérifiez sur la page GitHub de votre dépôt que le fichier `rapport/RAPPORT.md` s'affiche correctement, avec ses
images. **La version prise en compte est la dernière envoyée avant la date limite indiquée en cours.**

