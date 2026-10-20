"""Convertir une vidéo MP4 (par exemple une vidéo produite par lerobot-eval) en GIF animé.

Un GIF s'affiche directement dans un fichier Markdown sur GitHub, ce qui n'est pas le cas d'une vidéo MP4.

Exemple :
  python scripts/video_vers_gif.py outputs/eval/act_mes_demos/videos/pusht_0/eval_episode_0.mp4 rapport/images/mon_episode.gif
"""

import argparse
from pathlib import Path

import av
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="fichier vidéo d'entrée (.mp4)")
    parser.add_argument("gif", help="fichier GIF de sortie (.gif)")
    parser.add_argument("--largeur", type=int, default=256, help="largeur du GIF en pixels (plus petit = plus léger)")
    args = parser.parse_args()

    with av.open(args.video) as conteneur:
        flux = conteneur.streams.video[0]
        fps = float(flux.average_rate or 10)
        images = []
        for trame in conteneur.decode(flux):
            image = trame.to_image()
            hauteur = round(image.height * args.largeur / image.width)
            images.append(image.resize((args.largeur, hauteur), Image.LANCZOS))

    Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
    images[0].save(args.gif, save_all=True, append_images=images[1:], duration=round(1000 / fps), loop=0)
    print(f"GIF enregistré : {args.gif} ({len(images)} images, {Path(args.gif).stat().st_size / 1e6:.1f} Mo)")


if __name__ == "__main__":
    main()
