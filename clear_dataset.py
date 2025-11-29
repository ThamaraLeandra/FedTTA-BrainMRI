import os
import hashlib
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import shutil
from pathlib import Path


def calculate_file_hash(filepath):
    """Calcula hash SHA256 de um arquivo para detectar duplicatas exatas."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def clean_dataset(directory, output_dir="dataset_limpo"):
    """
    Remove apenas duplicatas exatas (baseadas no hash SHA256) de um dataset.

    Args:
        directory (str): Pasta com as imagens originais (treino + teste).
        output_dir (str): Pasta de saída para o dataset limpo.
    """
    os.makedirs(output_dir, exist_ok=True)

    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_paths = [
        os.path.join(root, file)
        for root, _, files in os.walk(directory)
        for file in files if file.lower().endswith(supported_formats)
    ]

    print(f"📂 Total de imagens encontradas: {len(image_paths)}")

    corrupted_files = []
    file_hashes = defaultdict(list)

    # --- Calcular hashes ---
    for path in tqdm(image_paths, desc="Calculando hashes"):
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception:
            corrupted_files.append(path)
            continue

        # Hash exato
        file_hash = calculate_file_hash(path)
        file_hashes[file_hash].append(path)

    # --- Encontrar duplicatas exatas ---
    exact_duplicates = {k: v for k, v in file_hashes.items() if len(v) > 1}

    # --- Remover duplicatas ---
    kept = set()
    removed = []

    for _, files in exact_duplicates.items():
        kept.add(files[0])  # mantém a primeira
        removed.extend(files[1:])  # remove o resto

    # --- Copiar imagens únicas para o dataset limpo ---
    for path in image_paths:
        if path not in removed:
            rel_path = os.path.relpath(path, directory)
            out_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            shutil.copy2(path, out_path)

    # --- Estatísticas ---
    print("\n✅ Limpeza concluída!")
    print(f"📂 Total original: {len(image_paths)}")
    print(f"❌ Duplicatas exatas removidas: {len(set(removed))}")
    print(f"✔ Restaram: {len(image_paths) - len(set(removed))}")
    print(f"📁 Dataset limpo salvo em: {output_dir}")


if __name__ == "__main__":
    # Exemplo de uso:
    clean_dataset(
        directory=Path("D:/TCC3/dataset_kaggle_preprocessed/all"),  # pasta onde estão treino+teste
        output_dir="dataset_processado_limpo"                     # pasta de saída
    )
