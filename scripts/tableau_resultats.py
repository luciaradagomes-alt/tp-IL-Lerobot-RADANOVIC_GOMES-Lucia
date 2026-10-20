"""Rassembler les résultats de plusieurs évaluations (lerobot-eval) dans un tableau Markdown.

Chaque dossier donné en argument doit contenir le fichier eval_info.json produit par lerobot-eval.
Le tableau affiché peut être copié-collé directement dans votre rapport.

  - Taux de réussite : pourcentage d'épisodes où le T a recouvert au moins 95 % de la cible.
  - Recouvrement ≥ 80 % : pourcentage d'épisodes où le T a recouvert au moins 80 % de la cible à un moment
    (épisodes réussis ou presque réussis).
  - Score moyen : pour chaque épisode, on garde le meilleur recouvrement atteint, divisé par 95 %
    (et plafonné à 100 %) ; puis on fait la moyenne sur les épisodes. Ce score montre les progrès
    même quand la politique ne réussit pas complètement.

Exemple :
  python scripts/tableau_resultats.py outputs/eval/*
"""

import argparse
import json
from pathlib import Path

SEUIL_REUSSITE = 0.95  # recouvrement à atteindre pour réussir l'épisode
SEUIL_PRESQUE = 0.80


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dossiers", nargs="+", help="dossiers de sortie de lerobot-eval")
    args = parser.parse_args()

    lignes = []
    for dossier in args.dossiers:
        fichier = Path(dossier) / "eval_info.json"
        if not fichier.exists():
            print(f"(ignoré : pas de fichier eval_info.json dans '{dossier}')")
            continue
        infos = json.loads(fichier.read_text())
        resultats = infos["overall"]
        # Le "reward" de PushT vaut recouvrement / 95 % (plafonné à 1) : on retrouve le recouvrement en multipliant.
        recouvrements_max = [
            r * SEUIL_REUSSITE for tache in infos["per_task"] for r in tache["metrics"]["max_rewards"]
        ]
        presque = 100 * sum(r >= SEUIL_PRESQUE for r in recouvrements_max) / len(recouvrements_max)
        lignes.append(
            f"| {Path(dossier).name} | {resultats['n_episodes']} | {resultats['pc_success']:.0f} % "
            f"| {presque:.0f} % | {100 * resultats['avg_max_reward']:.0f} % |"
        )

    print()
    print("| Évaluation | Nombre d'épisodes | Taux de réussite | Recouvrement ≥ 80 % | Score moyen |")
    print("|---|---|---|---|---|")
    print("\n".join(lignes))
    print()


if __name__ == "__main__":
    main()
