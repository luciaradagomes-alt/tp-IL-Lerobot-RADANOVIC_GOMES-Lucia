# Lab – Imitation Learning with LeRobot

<p align="center">
  <img src="images/pusht_diffusion.gif" width="300" alt="A policy learned by imitation solves the PushT task">
</p>

**Sorbonne Université – *AIHM* course**

*[Version française](README.md)*

Session length: **4 hours**. **Part 0 (installation) must be done before the session**.

> **Note for English-speaking students.** The scripts, their on-screen messages, the folder names and the report
> template ([rapport/RAPPORT.md](rapport/RAPPORT.md)) are in French. All commands below are exactly the ones to type,
> whatever your language. You may write your report in English.

---

## Contents

- [Goals and overview](#goals-and-overview)
- [Part 0 – Setting up your machine (before the session)](#part-0--setting-up-your-machine-before-the-session)
- [Part 1 – Discovering the task and the demonstrations](#part-1--discovering-the-task-and-the-demonstrations-25-min)
- [Part 2 – Recording your own demonstrations](#part-2--recording-your-own-demonstrations-30-min)
- [Part 3 – Training and evaluating an ACT policy](#part-3--training-and-evaluating-an-act-policy-55-min)
- [Part 4 – The limits of imitation learning](#part-4--the-limits-of-imitation-learning-60-min)
- [Part 5 – Another family of policies: Diffusion Policy](#part-5--another-family-of-policies-diffusion-policy-45-min)
- [Submitting your report with GitHub](#submitting-your-report-with-github)

---

## Goals and overview

**Imitation learning** (**IL**) is a machine learning approach that estimates a control policy from demonstrations
of a task performed by an expert. Each demonstration usually consists of a sequence of observations of the
environment and the corresponding actions. These data form a training set used to learn a policy able to reproduce
the observed behaviour.

**Behavior cloning** (**BC**) is an imitation learning method that formulates this problem as supervised learning.
From a set of observation-action pairs extracted from the demonstrations, a model is trained to predict the action
associated with a given observation. The resulting **policy** is a function that maps an observation to an action,
or to a probability distribution over the possible actions. For a robotic agent, this policy can then be used to
generate control commands from the available observations.

This lab is carried out **in simulation**, on your own computer, with Hugging Face's
**[LeRobot](https://huggingface.co/docs/lerobot)** library (an open-source Python library that gathers tools to
record demonstrations, and to train and evaluate robot policies).

Goals of the lab:

1. understand what a demonstration, a dataset and a policy are, and how to handle them with LeRobot;
2. record your own demonstrations and train a policy on them;
3. evaluate a policy rigorously (success rate on the same situations for every model);
4. explain the main **limits** of imitation learning (quantity and quality of data, compounding errors, unseen
   situations);
5. compare two families of policies (ACT and Diffusion Policy) and understand where their difference in
   performance on this task comes from.

> **The report.** You write your report **as you go**, in Markdown (a very simple text format that GitHub renders
> nicely; a good opportunity to learn its syntax). A template is waiting for you in
> [rapport/RAPPORT.md](rapport/RAPPORT.md). The questions to answer are marked **Q1**, **Q2**, etc.
> How to hand it in is explained in the section [Submitting your report with GitHub](#submitting-your-report-with-github).

### The task: PushT

![The PushT task](images/pusht.png)

PushT is a benchmark task in imitation learning. Seen from above:

- the **blue disk** is the "agent": the end of a robot, which we move around;
- the **grey T** is an object lying on a table: it can only be moved by pushing it with the agent;
- the **green area** is the target: the T must be brought exactly onto it.

An episode is a **success** when the T covers at least **95 %** of the green area. It is a **failure** after
**30 seconds**. The starting positions of the T and of the agent are drawn at random for each episode.

At each time step (10 times per second):

- the **observation** contains the (x, y) position of the agent and the (x, y) positions of the 8 corners of the T,
  i.e. 2 + 16 = 18 numbers;
- the **action** is the (x, y) position **targeted** by the agent: the agent moves towards this point
  (when you play, it is the position of your mouse).

> **Why positions rather than images?** A real robot would see the scene through a camera. But training a network
> on images requires a GPU (a powerful graphics card) and hours of computation. So that the lab runs on a plain
> laptop, the policy directly receives the exact positions of the objects, as if a perfect object-tracking system
> provided them. The principles of imitation learning remain exactly the same.

---

## Part 0 – Setting up your machine (before the session)

> ⚠️ **To do before the session.** Allow 45 min to 1 h and about 3 GB of disk space. You need an internet
> connection. If you are stuck, you can open an issue on GitHub, or as a last resort send me an email
> (quentin.rolland2@cea.fr) **before** the session.

These instructions follow the [official LeRobot installation guide](https://huggingface.co/docs/lerobot/installation),
adapted to our needs (simulation only, no GPU). The lab was prepared with **LeRobot 0.6.1** and **Python 3.12**: use
exactly these versions.

A word on vocabulary (just in case):

- **conda** is a manager of **Python environments**: an environment is an isolated folder containing a version of
  Python and libraries, without interfering with the rest of your computer. We will use **Miniforge**, a light and
  free version of conda;
- **pip** is the tool that installs Python libraries;
- **Git** is software that keeps the version history of your files; **GitHub** is a website that hosts Git
  repositories (a **repository** is a project folder tracked by Git);
- **CPU** is the main processor of your computer; **GPU** is a graphics card, very efficient for training neural
  networks. **This lab is designed to run on a CPU only.**

### 0.1 Create a GitHub account

If you do not have one, create a free account on [github.com](https://github.com/signup). You will use it to hand in
your report.

### 0.2 Install Git and Miniforge

Open the section that matches your system.

<details>
<summary><b>Windows</b></summary>

1. **Git**: download and run the installer from [git-scm.com/download/win](https://git-scm.com/download/win).
   Keep all the default options.
2. **Miniforge**: download `Miniforge3-Windows-x86_64.exe` from
   [conda-forge.org/download](https://conda-forge.org/download/) and run it. Choose the "Just Me" installation and
   keep the other default options.
3. From the Start menu, open **Miniforge Prompt**. **On Windows, all the commands of the lab must be typed in this
   window** (not in the classic command prompt nor in PowerShell).

</details>

<details>
<summary><b>macOS (Mac with an Apple M1, M2, M3, M4… chip, or Intel Mac)</b></summary>

To find out which chip you have: Apple menu  > *About This Mac* ("Chip" or "Processor" line).

1. Open the **Terminal** application (in *Applications > Utilities*).
2. **Git**: type the following command, then accept the installation of the "command line developer tools":
   ```bash
   xcode-select --install
   ```
   (If a message says they are already installed, all is well.)
3. **Miniforge**: type these two commands, then answer `yes` to the questions (licence, and conda initialisation):
   ```bash
   curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
   bash Miniforge3-$(uname)-$(uname -m).sh
   ```
4. **Close the terminal and open a new one.** The beginning of the line should now show `(base)`.

</details>

<details>
<summary><b>Linux</b></summary>

1. **Git**: `sudo apt install git` (Ubuntu, Debian) or `sudo dnf install git` (Fedora).
2. **Miniforge**: type these two commands, then answer `yes` to the questions (licence, and conda initialisation):
   ```bash
   wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
   bash Miniforge3-$(uname)-$(uname -m).sh
   ```
3. **Close the terminal and open a new one.** The beginning of the line should now show `(base)`.

</details>

### 0.3 Get the lab

In the terminal (Miniforge Prompt on Windows), go to the folder where you want to work, then **clone** the
repository (that is, download a copy of it tracked by Git):

```bash
git clone https://github.com/quentinRolld/tp-IL-Lerobot.git
cd tp-IL-Lerobot
```

The `cd` command (*change directory*) moves into a folder. **All the commands of the lab are run from this
`tp-IL-Lerobot` folder.**

### 0.4 Create the Python environment and install LeRobot

First create and activate a conda environment named `lerobot`, with Python 3.12 (same commands on every system):

```bash
conda create -y -n lerobot python=3.12
conda activate lerobot
conda install -y -c conda-forge ffmpeg
```

The beginning of the terminal line should now show `(lerobot)`. `ffmpeg` is a tool to read and write videos.

Then install **PyTorch**. The command depends on your system:

<details>
<summary><b>Windows</b> or <b>Linux</b> (without an NVIDIA GPU)</summary>

```bash
pip install "torch==2.11.0" "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cpu
```

This "CPU" version is much lighter (about 200 MB instead of several GB).

</details>

<details>
<summary><b>Windows</b> or <b>Linux</b> with an NVIDIA GPU (optional)</summary>

If you have a recent NVIDIA graphics card with up-to-date drivers, you can install the GPU version:

```bash
pip install "torch==2.11.0" "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cu128
```

This is not required for the lab: in case of trouble, use the CPU version above.

</details>

<details>
<summary><b>macOS with an Apple chip</b> (M1, M2, M3, M4…)</summary>

```bash
pip install "torch==2.11.0" "torchvision==0.26.0"
```

</details>

<details>
<summary><b>macOS with an Intel processor</b></summary>

PyTorch no longer publishes pip packages for Intel Macs, so we install it with conda:

```bash
conda install -y -c conda-forge "pytorch=2.11" "torchvision=0.26"
```

I could not test this configuration on every machine: if the installation or the check script fails, let me know
before the session (you will be able to work in pairs).

</details>

Finally, install **LeRobot** with the extras needed for the lab (same command on every system):

```bash
pip install "lerobot[pusht,training,diffusion]==0.6.1" matplotlib
```

- `pusht` adds the PushT simulation;
- `training` adds the training tools;
- `diffusion` adds what is needed to use the *Diffusion Policy* of Part 5.

### 0.5 Download the models of Part 5

Part 5 compares two pre-trained policies which weigh a little over 1 GB in total: download them **before the
session**, so as not to saturate the network on the day of the lab.

```bash
python scripts/recuperer_modeles.py
```

The script stores both models in the `modeles/` folder:

- `diffusion_pusht_keypoints/`: a *Diffusion Policy* published by the LeRobot team on the Hugging Face Hub. It was
  published with an older version of LeRobot: the script converts it to the format of version 0.6.1;
- `act_pusht_keypoints_200k/`: an ACT policy that I trained under the same conditions, downloaded from the
  *Releases* page of the lab's GitHub repository.

### 0.6 Check the installation

```bash
python scripts/verifier_installation.py
```

This script tests each component, downloads the demonstration dataset (a few MB) and opens a simulation window for
5 seconds (the blue agent follows your mouse). Every line must show `[OK]` and the script must end with
"Tout est prêt pour le TP ! Amusez-vous bien" ("Everything is ready for the lab! Have fun").

### 0.7 At each new work session

When you open a new terminal, you must reactivate the environment and go back to the lab folder:

```bash
conda activate lerobot
cd path/to/tp-IL-Lerobot
```

(replace `path/to` with the location where you cloned the repository).

---

## Part 1 – Discovering the task and the demonstrations (25 min)

### 1.1 Play it yourself

Run the recording script in **practice mode** (nothing is saved):

```bash
python scripts/enregistrer_demos.py --entrainement
```

Place the mouse on the blue disk: it starts following the mouse. Push the T onto the green area. The bottom banner
shows the percentage of the target that is covered. `R` restarts the episode, `Esc` quits. Do it 3 or 4 times to get
the hang of it.

### 1.2 The reference demonstration dataset

LeRobot provides a **dataset** of 206 PushT demonstrations performed by humans, published on the
**Hugging Face Hub**.

1. Watch a few demonstrations as videos with the online viewer:
   [lerobot/pusht in the viewer](https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Flerobot%2Fpusht%2Fepisode_0).
   (This `lerobot/pusht` dataset contains images; we will use its `lerobot/pusht_keypoints` version, which contains
   the same demonstrations but with the positions of the corners of the T instead of images.)
2. Run the exploration script on the reference dataset:
   ```bash
   python scripts/explorer_dataset.py --dataset lerobot/pusht_keypoints
   ```
   It prints key figures and saves five plots in `rapport/images/pusht_keypoints/`:
   - `trajectoires.png`: the path followed by the agent in a few demonstrations;
   - `positions_depart.png`: the starting positions of the T and of the agent in all demonstrations;
   - `durees.png`: the duration of the demonstrations;
   - `etat_et_action.png`: the position of the agent (the **state**) and the targeted position (the **action**)
     during one demonstration;
   - `carte_de_chaleur.png`: the areas of the scene where the T and the agent have been during all demonstrations
     (used in Part 4.4).

In LeRobot, a dataset is a sequence of **time steps** (*frames*), grouped into **episodes** (one episode = one
demonstration, from start to end). Each time step contains:

| Name in LeRobot | Size | Meaning |
|---|---|---|
| `observation.state` | 2 | (x, y) position of the agent |
| `observation.environment_state` | 16 | (x, y) positions of the 8 corners of the T |
| `action` | 2 | (x, y) position targeted by the agent at this time step |
| `episode_index`, `frame_index`, `timestamp`… | 1 | episode number, step number within the episode, time in seconds… |

**Q1.** How many episodes and time steps does the dataset contain? What is the average duration of a
demonstration? Insert the `etat_et_action.png` plot in your report (see the report template for the Markdown syntax
of an image).

**Q2.** On the `etat_et_action.png` plot, the action and the state are close but shifted. Which one is "ahead" of
the other? Why does this make sense?

**Q3.** Formulate imitation learning for this task as a supervised learning problem: what is the input of the
network, what is the expected output, and where do the examples come from?

**Q4.** Look at `positions_depart.png`. Do the starting positions cover the whole scene?

---

## Part 2 – Recording your own demonstrations (30 min)

You will now become the expert and record **20 successful demonstrations**:

```bash
python scripts/enregistrer_demos.py --nb-demos 20
```

- recording of an episode starts when the mouse touches the blue disk (a red frame appears);
- as soon as the T covers 95 % of the target, the demonstration is **saved automatically** and the next episode
  starts;
- after 30 s without success, the episode is dropped and **not saved**;
- `R` restarts the current episode (it is not saved), `Esc` quits (the successful demonstrations already recorded
  are kept).

Your demonstrations are saved in LeRobot format in the `donnees/mes_demos/` folder. If you quit before the end, run
the same command again: new demonstrations are appended to the previous ones.

> **Tips for good demonstrations.** The network will imitate **everything** you do, including your hesitations.
> Try to be consistent: calm movements, a coherent strategy from one demonstration to the next, no unnecessary pause.

When you are done, explore your own data:

```bash
python scripts/explorer_dataset.py --dossier donnees/mes_demos
```

The plots are saved in `rapport/images/mes_demos/`.

**Q5.** Compare your demonstrations with those of the reference dataset (average duration, shape of the
trajectories); I expect a table comparing the relevant statistics. Insert the table and your `trajectoires.png`
plot in the report.

**Q6.** How long did it take you to record 20 demonstrations, and how many attempts did you fail? What do you
conclude about the cost of collecting data for a real robot, which would need hundreds of them?

---

## Part 3 – Training and evaluating an ACT policy (55 min)

### 3.1 The ACT policy in short

We use **ACT** (*Action Chunking with Transformers*, [Zhao et al., 2023](https://arxiv.org/abs/2304.13705)), a policy
widely used in robotics and available in LeRobot. Its two main ideas:

- **Predicting sequences of actions** (*action chunking*). Instead of predicting one action at a time, the network
  predicts the next `chunk_size` actions at once (here 20 actions, i.e. 2 seconds). The policy then executes
  `n_action_steps` of them (here 10, i.e. 1 second) before making a new prediction from the new observation.
- **A Transformer network, trained as a CVAE.**
  A **CVAE** (*Conditional Variational AutoEncoder*) adds a small hidden variable, called the **latent variable**,
  which summarises the "style" of the demonstration (for example: going around the T on the left or on the right).
  During training, a second network (the encoder) computes this variable from the true actions of the expert; when
  the policy is used, it is set to zero (the "average" style).

The **loss** (the quantity that training tries to make as small as possible) is the sum of two terms:

- `l1_loss`: the mean absolute difference between the predicted actions and those of the expert;
- `kld_loss`: the **Kullback-Leibler divergence** (KL), which measures how far the latent variable is from a normal
  distribution; it prevents this variable from "cheating" by copying the answers.

So that training fits on a CPU, we use a version of ACT about **10 times smaller** than the original one (4 million
parameters instead of 40 million).

### 3.2 Start the training

Training takes **about 15 to 25 minutes** on a recent computer. **Start it right away**, then read on and answer the
questions while it runs.

On **macOS and Linux**:

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
<summary>On <b>Windows</b> (Miniforge Prompt): same command, with <code>^</code> instead of <code>\</code></summary>

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

Meaning of the options:

| Option | Meaning |
|---|---|
| `--dataset.repo_id`, `--dataset.root` | the name of the dataset and the folder where it is stored on your disk |
| `--policy.type=act` | the type of policy to train |
| `--policy.device=cpu` | train on the processor (CPU). With an NVIDIA GPU: `cuda`; on a Mac with an Apple chip, you can try `mps` |
| `--policy.push_to_hub=false` | do not upload the model to the Hugging Face Hub |
| `--policy.chunk_size=20` | number of actions predicted at once (20 actions = 2 s) |
| `--policy.n_action_steps=10` | number of actions executed before making a new prediction (10 actions = 1 s) |
| `--policy.dim_model`, `n_heads`, `dim_feedforward`, `n_encoder_layers`, `n_vae_encoder_layers` | the size of the network (reduced here for the CPU) |
| `--policy.normalization_mapping` | how each type of data is **normalised** before being fed to the network. `MEAN_STD`: subtract the mean and divide by the standard deviation (computed on the dataset), so that all inputs are of order 1. `STATE` is the position of the agent, `ENV` the coordinates of the corners of the T, `ACTION` the actions. By default, LeRobot does not normalise `ENV` inputs for ACT: we ask for it explicitly |
| `--policy.optimizer_lr=1e-4` | the **learning rate**: the size of the steps taken to correct the network weights at each iteration |
| `--steps=5000` | the number of **iterations** (*steps*): at each iteration, the network sees a batch of examples and corrects its weights |
| `--batch_size=64` | the size of a **batch**: the number of examples (time steps) used at each iteration |
| `--save_freq=1000` | save the model (a ***checkpoint***) every 1000 iterations |
| `--log_freq=200` | print the loss every 200 iterations |
| `--num_workers=2` | number of processes preparing the data in parallel |
| `--wandb.enable=false` | disable Weights & Biases (an online service for tracking training runs) |
| `--output_dir` | the folder where the results are stored |

Every 200 iterations, a line like this one is printed:

```
step:1K smpl:64K ep:493 epch:24.63 loss:0.297 grdn:6.303 lr:1.0e-04 updt_s:0.264 data_s:0.009 smp/s:235 l1_loss:0.287 kld_loss:0.001
```

`step` is the iteration number, `smpl` the number of examples seen, `epch` the number of **epochs** (one epoch = the
network has seen, on average, each example of the dataset once), `loss` the total loss, `grdn` the size of the
gradient (the direction in which the weights are corrected), `lr` the learning rate, `updt_s` the computation time of
one iteration in seconds. Write down a few values of `loss`, `l1_loss` and `kld_loss` during training for question Q9.

The models are stored in `outputs/train/act_mes_demos/checkpoints/`: one sub-folder per checkpoint (`001000`,
`002000`, …) and `last` for the latest one.

**Q7.** With 20 demonstrations, how many time steps does your dataset contain? With `--batch_size=64` and
`--steps=5000`, how many times has the network seen each example on average (that is, how many epochs)? Compare
with the number of epochs reported at the end of training.

**Q8.** With `chunk_size=20` and `n_action_steps=10`, how many times per second does the policy make a prediction?
What advantage and what drawback do you see in executing several actions "blindly" (that is, executing a sequence of
actions at once instead of one action at a time)?

**Q9.** How do `l1_loss` and `kld_loss` evolve during training? Does a very low final loss guarantee that the policy
will succeed at the task? (You will be able to answer fully after Part 3.3.) To go further: `kld_loss` becomes almost
zero; based on the description in Part 3.1, what can be inferred about how the network uses its latent variable?

### 3.3 Evaluate the policy

While waiting for training to finish, you can already watch a model that I trained beforehand, for longer, on the
206 reference demonstrations (see Part 4):

```bash
python scripts/tester_politique.py --politique modeles/act_pusht_206demos_long
```

Once **your** training is finished, evaluate your policy on **50 episodes**:

```bash
lerobot-eval --policy.path=outputs/train/act_mes_demos/checkpoints/last/pretrained_model --env.type=pusht --env.obs_type=environment_state_agent_pos --eval.n_episodes=50 --eval.batch_size=10 --eval.use_async_envs=false --policy.device=cpu --output_dir=outputs/eval/act_mes_demos
```

| Option | Meaning |
|---|---|
| `--policy.path` | the folder of the model to evaluate |
| `--env.type=pusht`, `--env.obs_type=environment_state_agent_pos` | the simulation to use, with observations given as positions |
| `--eval.n_episodes=50` | number of episodes played |
| `--eval.batch_size=10` | number of simulations played at the same time |
| `--eval.use_async_envs=false` | run the simulations in a single process (required with PushT in this version of LeRobot) |
| `--output_dir` | folder where the results and videos are written |

This command only works on a single line: it is the same on every system.

The evaluation takes 1 to 2 minutes. It prints and saves in `outputs/eval/act_mes_demos/eval_info.json`:

- `pc_success`: the **success rate**, as a percentage of episodes;
- `avg_max_reward`: the **average score**. For each episode, we keep the best coverage reached divided by 95 %
  (so 1 = success), then average over episodes. This score shows progress even when the policy does not fully
  succeed.

> **Do not be surprised by low success rates.** The PushT success criterion (95 % coverage) is very demanding: the T
> must be placed almost to the pixel. With a few minutes of training on CPU, a success rate close to 0 % is normal.
> Compare mainly the **scores** and the percentage of episodes that reached at least 80 % coverage (given by the
> `tableau_resultats.py` script, Part 4).

The starting positions of the 50 episodes are drawn with a fixed **random seed** (a number that determines the
sequence of "random" draws): all the models evaluated with this command face exactly the same 50 situations, which
makes the comparison fair.

The videos of the first 10 episodes are in `outputs/eval/act_mes_demos/videos/pusht_0/`. To put one in your report,
convert it to an animated GIF (GitHub displays GIFs directly in a Markdown file):

```bash
python scripts/video_vers_gif.py outputs/eval/act_mes_demos/videos/pusht_0/eval_episode_0.mp4 rapport/images/mon_modele.gif
```

Finally, watch your policy play live:

```bash
python scripts/tester_politique.py --politique outputs/train/act_mes_demos/checkpoints/last/pretrained_model
```

The red cross shows the point targeted by the current action. `Space` moves to the next episode, `P` pauses, `Esc`
quits.

**Q10.** Give the success rate and the average score of your policy. Insert a GIF of a successful episode (or, if
none, of a failed one) and describe the behaviour of the policy: does it imitate your way of playing?

**Q11.** (Optional, if you have time.) Also evaluate checkpoint `001000` (replace `last` with `001000` in
`--policy.path`, and change `--output_dir`). Does the model improve between 1000 and 5000 iterations?

---

## Part 4 – The limits of imitation learning (60 min)

The [modeles/](modeles/) folder contains ACT policies that I trained on the **first N demonstrations** of the
reference dataset `lerobot/pusht_keypoints`. The first four have **exactly the same settings as yours** (same
network size, 5000 iterations). The last one was trained 6 times longer, on a GPU:

| Model | Demonstrations used | Iterations |
|---|---|---|
| `modeles/act_pusht_10demos` | 10 | 5,000 |
| `modeles/act_pusht_20demos` | 20 | 5,000 |
| `modeles/act_pusht_50demos` | 50 | 5,000 |
| `modeles/act_pusht_206demos` | 206 (the whole dataset) | 5,000 |
| `modeles/act_pusht_206demos_long` | 206 (the whole dataset) | 30,000 |

### 4.1 The quantity of data

Evaluate the five models with the same command as in Part 3.3, changing `--policy.path` and `--output_dir`. For
example, for the first one:

```bash
lerobot-eval --policy.path=modeles/act_pusht_10demos --env.type=pusht --env.obs_type=environment_state_agent_pos --eval.n_episodes=50 --eval.batch_size=10 --eval.use_async_envs=false --policy.device=cpu --output_dir=outputs/eval/act_pusht_10demos
```

Then gather all your results in a Markdown table, ready to be copied into your report:

```bash
python scripts/tableau_resultats.py outputs/eval/act_pusht_10demos outputs/eval/act_pusht_20demos outputs/eval/act_pusht_50demos outputs/eval/act_pusht_206demos outputs/eval/act_pusht_206demos_long outputs/eval/act_mes_demos
```

**Q12.** Copy the table into your report. How do the success rate, the coverage ≥ 80 % and the average score evolve
with the number of demonstrations? And with the training duration (compare `act_pusht_206demos` and
`act_pusht_206demos_long`)? Even the best model is far from a reliable policy: in your opinion, what should be changed
(data, model, training) to get closer to one?

### 4.2 The quality and consistency of demonstrations

**Q13.** Compare **your** model (20 demonstrations) with the `act_pusht_20demos` model (20 reference
demonstrations). Which one is better? Suggest explanations (regularity of movements, duration of demonstrations,
strategy, diversity of starting positions…). Support your answer with the plots from `explorer_dataset.py`.

**Q14.** If you had to record 20 additional demonstrations to improve your model, how would you go about it (way of
playing, starting situations to favour…)? Justify using your plots.

### 4.3 Compounding errors

During training, the policy has only seen situations encountered by the expert. But at inference time, it makes
small errors, which lead it to slightly different situations, where it makes slightly bigger errors, and so on. The
errors accumulate (*compounding errors*). This is also called **covariate shift**: the situations encountered at test
time fall outside the distribution of the training data.

Test the robustness of the `act_pusht_206demos_long` model by **perturbing** it: while it plays, hold the left mouse
button to take control of the agent, move it elsewhere (or move the T), then release the button to hand control
back.

```bash
python scripts/tester_politique.py --politique modeles/act_pusht_206demos_long
```

**Q15.** Describe 2 or 3 perturbations you tested. Does the policy manage to recover? In which cases does it fail?
Interpretation?

### 4.4 Unseen situations

To find out which situations the policy has **never seen**, look at the **heat map** of the positions visited
during the 206 reference demonstrations (the `carte_de_chaleur.png` plot produced by `explorer_dataset.py` in
Part 1). The scene is split into 16 × 16-unit cells; the colour indicates how many time steps the centre of the T
(left) or the agent (right) spent in each cell. **White** cells have never been visited: a T placed there is
**out of distribution** (outside the area covered by the training data).

![Heat map of the positions visited in lerobot/pusht_keypoints](images/carte_de_chaleur_pusht.png)

The red crosses are the three centres of the T tested below: (300, 345), at the heart of the data, and (105, 93) and
(393, 93), in the white area.

The `--etat-initial` option of `tester_politique.py` sets the starting position: (x, y) position of the agent,
(x, y) position of the **centre of the T** (as on the map), then the angle of the T in radians. Reminder: the scene
goes from 0 to 512 in each direction, and (0, 0) is the top-left corner.

Compare a situation **similar** to the demonstrations:

```bash
python scripts/tester_politique.py --politique modeles/act_pusht_206demos_long --etat-initial 100 100 300 345 0
```

with situations **never seen** during training (agent in a corner of the scene, T near the top edge, in an area the
centre of the T has never visited):

```bash
python scripts/tester_politique.py --politique modeles/act_pusht_206demos_long --etat-initial 492 492 105 93 -2.66
python scripts/tester_politique.py --politique modeles/act_pusht_206demos_long --etat-initial 20 492 393 93 2.19
```

To choose your own situations, you can place crosses on the map with the `--marquer X Y` option (repeatable), then
check which area they fall in:

```bash
python scripts/explorer_dataset.py --dataset lerobot/pusht_keypoints --marquer 256 120 --marquer 450 450
```

Beware: the T is 120 units wide and high. If its centre is too close to a wall, it overlaps the wall and the
simulation pushes it away abruptly at the start of the episode.

**Q16.** How does the policy behave in these different situations? Interpretation?

---

## Part 5 – Another family of policies: Diffusion Policy (45 min)

Even the best ACT of Part 4 remains far from a reliable policy. Here we compare it with another class of policies,
**Diffusion Policy** ([Chi et al., 2023](https://arxiv.org/abs/2303.04137)), which relies on diffusion models
([Ho et al., 2020](https://arxiv.org/abs/2006.11239)).

### 5.1 Formal framework

**Behavior cloning.** Let $o_t$ be the observation at time $t$ and $A_t = (a_t, a_{t+1}, \dots, a_{t+H-1})$ the chunk
of the next $H$ actions of the expert ($H$ is the prediction **horizon**; here $H = 16$, and each
$a_t \in \mathbb{R}^2$). The demonstrations provide pairs $(o_t, A_t)$, considered as drawn from an unknown
distribution. For a given observation $o$, the expert does not always react in the same way: its actions follow a
**conditional distribution** $p_E(A \mid o)$ ("distribution of $A$ given $o$"). Learning a policy amounts to
approximating this conditional distribution.

**ACT.** As seen in Part 3.1, ACT is a **conditional variational autoencoder** (CVAE). During training, an encoder
$q_\phi(z \mid A, o)$ summarises the expert's action chunk into a latent variable $z \in \mathbb{R}^{32}$, and a
decoder $f_\theta(o, z)$ reconstructs the action chunk from $o$ and $z$ ($\theta$ and $\phi$ denote the weights of
the two networks). The loss minimised is:

$$
\mathcal{L}_{\text{ACT}}(\theta, \phi) = \mathbb{E}\Big[\, \big\| f_\theta(o, z) - A \big\|_1 \Big]
\;+\; \beta \; \mathbb{E}\Big[ D_{\text{KL}}\big(q_\phi(z \mid A, o) \,\big\|\, \mathcal{N}(0, I)\big) \Big],
\qquad z \sim q_\phi(z \mid A, o)
$$

where $\|\cdot\|_1$ is the sum of absolute values (the `l1_loss` term), $D_{\text{KL}}$ the Kullback-Leibler
divergence (the `kld_loss` term), $\mathcal{N}(0, I)$ the standard normal distribution, chosen as the **prior** of
$z$, and $\beta = 10$ a weighting coefficient. At inference time, the expert is no longer there to provide $A$: $z$ is
set to the mean of the prior, $z = 0$, and the policy returns $\hat{A} = f_\theta(o, 0)$.

**Diffusion Policy.** Diffusion Policy is a **generative model** of the conditional distribution $p_E(A \mid o)$,
that is, a model from which samples $A$ can be drawn for a given observation $o$. It relies on two processes:

- a **noising process** (used only during training): starting from an expert action chunk $A^0 = A$, increasingly
  noisy versions are built,
  $A^k = \sqrt{\bar{\alpha}_k}\, A^0 + \sqrt{1 - \bar{\alpha}_k}\; \varepsilon$, with
  $\varepsilon \sim \mathcal{N}(0, I)$ and $k = 1, \dots, K$ ($K = 100$). The coefficients
  $\bar{\alpha}_k \in (0, 1)$ decrease with $k$, so that $A^K$ is almost pure noise;
- a network $\varepsilon_\theta(A^k, k, o)$ trained to **recover the noise** $\varepsilon$ that was added, by
  minimising the squared error:

$$
\mathcal{L}_{\text{DP}}(\theta) = \mathbb{E}_{(o, A),\, k,\, \varepsilon}\Big[ \big\| \varepsilon - \varepsilon_\theta(A^k, k, o) \big\|_2^2 \Big]
$$

At inference time, $A^K \sim \mathcal{N}(0, I)$ is **drawn**, then a denoising step
$A^{k-1} = g\big(A^k, \varepsilon_\theta(A^k, k, o)\big)$ is applied iteratively until $A^0$ is obtained, the action
chunk that is executed. Here, the DDIM sampling rule ([Song et al., 2021](https://arxiv.org/abs/2010.02502)) allows
only **10 denoising steps** instead of 100. Each step requires a full evaluation of the network.

### 5.2 Experimental protocol

The two policies compared were trained on the same data and with the same budget: the 206 demonstrations of
`lerobot/pusht_keypoints`, the same observations (position of the agent and of the 8 corners of the T),
200,000 iterations on batches of 64 examples, an initial learning rate of $10^{-4}$, a horizon $H = 16$ of which
8 actions are executed before a new prediction, and normalised inputs and outputs (by minimum and maximum for
Diffusion Policy, by mean and standard deviation for ACT).

| Model | Type | Number of parameters | Origin |
|---|---|---|---|
| `modeles/act_pusht_keypoints_200k` | ACT (original size) | 40 million | trained by me with the protocol above (downloaded in Part 0) |
| `modeles/diffusion_pusht_keypoints` | Diffusion Policy | 249 million | model published by the LeRobot team (downloaded in Part 0) |

The two models nevertheless differ on three points, to keep in mind when interpreting the results: Diffusion Policy
is conditioned on the **two** latest observations $(o_{t-1}, o_t)$ whereas ACT only receives $o_t$; its learning rate
decreases during training (cosine schedule), whereas ACT's remains constant (LeRobot's default setting); finally, it
has six times more parameters.

### 5.3 Quantitative evaluation

First start the evaluation of Diffusion Policy, **slow on CPU** (5 to 15 minutes depending on your machine), and
read on while it runs:

```bash
lerobot-eval --policy.path=modeles/diffusion_pusht_keypoints --env.type=pusht --env.obs_type=environment_state_agent_pos --eval.n_episodes=50 --eval.batch_size=10 --eval.use_async_envs=false --policy.device=cpu --output_dir=outputs/eval/diffusion_pusht_keypoints
```

Then that of ACT (1 to 2 minutes) and the comparison table:

```bash
lerobot-eval --policy.path=modeles/act_pusht_keypoints_200k --env.type=pusht --env.obs_type=environment_state_agent_pos --eval.n_episodes=50 --eval.batch_size=10 --eval.use_async_envs=false --policy.device=cpu --output_dir=outputs/eval/act_pusht_keypoints_200k
```

```bash
python scripts/tableau_resultats.py outputs/eval/act_pusht_keypoints_200k outputs/eval/diffusion_pusht_keypoints
```

**Q17.** Copy the table into your report and compare the two policies. Note the duration of each evaluation. Based
on Section 5.1, express the number of network evaluations needed to produce an action chunk with each of the two
policies, and discuss the consequences for real-time control of a robot. Insert a GIF of an episode of each policy
(see `video_vers_gif.py`, Part 3.3).

### 5.4 Analysis in a given situation

The following script sets an initial situation $o$ (the agent below the T, which can be bypassed on the left or on
the right) and produces two plots in `rapport/images/comparaison/`:

- `predictions.png`: for **the same observation** $o$, each policy is queried **20 times**; each orange curve
  represents a predicted action chunk $\hat{A}$, that is, the sequence of positions targeted by the agent over about
  1.5 seconds;
- `episodes.png`: **8 complete episodes** are played by each policy from this same initial situation; the trajectory
  of the agent over the first 4 seconds is plotted.

```bash
python scripts/comparer_politiques.py --politiques modeles/act_pusht_keypoints_200k modeles/diffusion_pusht_keypoints
```

The script takes a few minutes, because of Diffusion Policy.

**Q18.** Insert both plots in your report and describe `predictions.png` for each policy. Based on Section 5.1,
explain why the 20 predictions of ACT coincide, and identify precisely the source of the variability of the
predictions of Diffusion Policy.

**Q19.** How many **modes** (distinct groups of trajectories) can you distinguish among the predictions of Diffusion
Policy? Does each correspond to an admissible strategy for bringing the T onto the target? Which of these modes does
ACT's prediction follow?

**Q20.** Theoretical analysis.

1. Suppose that, in the situation studied, the expert goes around the T on the left half of the time (chunk
   $A_{\text{left}}$) and on the right half of the time (chunk $A_{\text{right}}$):
   $p_E(A \mid o) = \tfrac{1}{2}\,\delta_{A_{\text{left}}} + \tfrac{1}{2}\,\delta_{A_{\text{right}}}$, where $\delta_x$
   denotes the distribution that gives the value $x$ with probability 1. A distribution with several modes is called
   **multimodal**. A deterministic predictor $f$ associates a single chunk $f(o)$ with each observation $o$: what part
   of $p_E(A \mid o)$ can it reproduce? What does this imply for the diversity of the policy's behaviours, from one
   episode to the next, starting from the same situation?
2. In principle, the latent variable $z$ of the CVAE could indicate which mode to follow. Based on the evolution of
   `kld_loss` observed in Q9 (the divergence $D_{\text{KL}}$ tends to 0) and on the fact that $z = 0$ is used at
   inference time, explain why ACT behaves here as a deterministic predictor $o \mapsto f_\theta(o)$. What can be
   inferred about the information that $z$ conveys to the decoder?
3. Explain why the sampling process of Diffusion Policy, on the contrary, can represent all the modes of
   $p_E(A \mid o)$.

**Q21.** Compare, on `episodes.png`, the variability of the trajectories from one episode to the next for the two
policies. Is the randomness of Diffusion Policy a flaw? Then run the script again from several initial situations of
your choice (`--etat-initial` option, same format as in Part 4.4). Do the predictions of Diffusion Policy always show
several modes? Finally, over 200 randomly drawn initial situations, I measured that the advantage of Diffusion Policy
over ACT is of the same order in the situations where it predicts several modes (62 % success for ACT versus 82 %)
and in the others (53 % versus 78 %). Is multimodality enough to explain the performance gap measured in 5.3? Which
other differences between the two policies (Section 5.2) could contribute to it, and which experiment would allow
these hypotheses to be told apart?

**Q22.** Conclusion: based on the whole lab, summarise in 5 to 10 lines what imitation learning does well, its main
limits, and what the choice of the class of policy can (or cannot) fix, using the notion of multimodality.

---

## Submitting your report with GitHub

The report is the `rapport/RAPPORT.md` file, together with its images in `rapport/images/`. To hand it in, you will
create **your own private GitHub repository** (only you and I will be able to see it), push your work to it, then
invite me.

### Step 1 – Create your private repository on GitHub

1. Log in to [github.com](https://github.com), click **+** (top right) then **New repository**.
2. *Repository name*: `tp-IL-Lerobot-LASTNAME-Firstname` (with your last name and first name).
3. Tick **Private**.
4. **Do not tick anything else** (no README, no .gitignore, no licence): the repository must be empty.
5. Click **Create repository**. Keep the page open: it shows the address of your repository, of the form
   `https://github.com/YOUR-USERNAME/tp-IL-Lerobot-LASTNAME-Firstname.git`.

### Step 2 – Link your local folder to this repository

In the terminal, from the `tp-IL-Lerobot` folder (replace the address with yours):

```bash
git remote rename origin prof
git remote add origin https://github.com/YOUR-USERNAME/tp-IL-Lerobot-LASTNAME-Firstname.git
```

Your folder now knows two remote repositories (*remotes*): `prof` (mine, where the lab instructions come from) and
`origin` (yours, where you hand in your work). If I fix the instructions during the lab, you will be able to get the
update with `git pull prof main`.

Then tell Git your name and email address (only once per computer):

```bash
git config --global user.name "Firstname Lastname"
git config --global user.email "your.address@etu.sorbonne-universite.fr"
```

### Step 3 – Push your report

Each time you want to save a version of your work:

```bash
git add rapport
git commit -m "Report: parts 1 to 3"
git push -u origin main
```

- `git add rapport`: selects the `rapport` folder (text and images) for the next version;
- `git commit -m "..."`: creates a version (a ***commit***) with a message describing it;
- `git push`: sends your versions to GitHub (`-u origin main` is only needed the first time; afterwards, `git push`
  is enough).

> **Authentication.** On the first `git push`, GitHub asks you to authenticate. On Windows and macOS, a login window
> usually opens by itself: log in with your GitHub account. If Git asks for a password in the terminal, your usual
> GitHub password will not work: you need a **personal access token**. Create one at
> [github.com/settings/tokens](https://github.com/settings/tokens) (*Generate new token (classic)*, tick the `repo`
> box), and paste it instead of the password.

The `donnees/` and `outputs/` folders (demonstrations, models, videos) are deliberately ignored by Git (see the
`.gitignore` file): they are too heavy and I do not need them. Only the `rapport/` folder matters to me.

### Step 4 – Invite me to your repository

On your GitHub repository page: **Settings** > **Collaborators** > **Add people**, and add the account
**`quentinRolld`**.

Check on the GitHub page of your repository that the `rapport/RAPPORT.md` file is displayed correctly, with its
images. **The version taken into account is the last one pushed before the deadline given in class.**
