import os
import json

ANNOTATIONS_DIR = "data/annotations2"  # JSON actuels
IMAGES_DIR = "data/images2"            # PNG générés

def fix_json_paths():
    for fname in os.listdir(ANNOTATIONS_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(ANNOTATIONS_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Détermine le nom de base pour le PNG
        base_name = os.path.splitext(fname)[0]  # ex: "50328713_page1"
        page_num = 1  # commence à 1 par défaut
        if "_page" in base_name:
            try:
                page_num = int(base_name.split("_page")[-1])
            except:
                pass

        # Crée le chemin correct avec page_num
        png_path = os.path.join(IMAGES_DIR, f"{base_name.split('_page')[0]}_p{page_num}.png")
        data["image_path"] = png_path

        # Sauvegarde le JSON corrigé
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✅ {fname} corrigé → {png_path}")


def fix_json_paths():
    for fname in os.listdir(ANNOTATIONS_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(ANNOTATIONS_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Base name du JSON (ex: 15858254_page1)
        base_name = os.path.splitext(fname)[0]

        # Nouveau chemin vers l'image correspondant exactement au fichier existant
        png_path = os.path.join(IMAGES_DIR, f"{base_name}.png")
        data["image_path"] = png_path

        # Sauvegarde
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✅ {fname} corrigé → {png_path}")


if __name__ == "__main__":
    fix_json_paths()
