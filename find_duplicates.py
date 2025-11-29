import os
import hashlib
import imagehash
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import argparse

# --- Funções de Verificação de Duplicatas ---

def calculate_file_hash(filepath):
    """Calcula o hash SHA256 de um arquivo para encontrar duplicatas exatas."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        # Lê o arquivo em blocos para não sobrecarregar a memória com arquivos grandes
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def calculate_perceptual_hash(image_path):
    """Calcula o perceptual hash (pHash) de uma imagem."""
    try:
        with Image.open(image_path) as img:
            # Converter para 'L' (escala de cinza) para normalizar antes do hash
            return imagehash.phash(img.convert('L'))
    except Exception:
        # Retorna None se a imagem não puder ser aberta
        return None

# --- Função Principal ---

def find_duplicate_images(directory, p_hash_threshold=5, output_file="relatorio_duplicatas.txt"):
    """
    Analisa um diretório de imagens para encontrar duplicatas exatas e visuais
    e salva o relatório em um arquivo de texto.
    """
    image_paths = []
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(supported_formats):
                image_paths.append(os.path.join(root, file))

    if not image_paths:
        print("Nenhuma imagem encontrada no diretório especificado.")
        return
        
    print(f"Encontradas {len(image_paths)} imagens. Iniciando análise de duplicatas...\n")

    corrupted_files = []
    file_hashes = defaultdict(list)
    image_hashes = {}

    for path in tqdm(image_paths, desc="Calculando hashes"):
        try:
            with Image.open(path) as img:
                img.verify()
        except (IOError, SyntaxError):
            corrupted_files.append(path)
            continue

        file_hash = calculate_file_hash(path)
        file_hashes[file_hash].append(path)

        p_hash = calculate_perceptual_hash(path)
        if p_hash:
            image_hashes[path] = p_hash
        else:
            if path not in corrupted_files:
                corrupted_files.append(path)

    exact_duplicates = {k: v for k, v in file_hashes.items() if len(v) > 1}

    visual_duplicates = defaultdict(list)
    image_list = list(image_hashes.keys())
    processed_images = set()

    print("\nComparando hashes de imagem para encontrar duplicatas visuais...")
    for i in tqdm(range(len(image_list)), desc="Comparando imagens"):
        img1_path = image_list[i]
        if img1_path in processed_images:
            continue

        current_group = []
        for j in range(i + 1, len(image_list)):
            img2_path = image_list[j]
            if img2_path in processed_images:
                continue

            hash_dist = image_hashes[img1_path] - image_hashes[img2_path]
            
            if hash_dist <= p_hash_threshold:
                if not current_group:
                    current_group.append(img1_path)
                    processed_images.add(img1_path)
                
                current_group.append(img2_path)
                processed_images.add(img2_path)
        
        if current_group:
            visual_duplicates[current_group[0]] = current_group

    # --- Salvar Relatório em Arquivo ---
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("--- Relatório de Duplicatas no Dataset ---\n\n")

        f.write(f"[+] Imagens Corrompidas ou Ilegíveis: {len(corrupted_files)}\n")
        for file in corrupted_files:
            f.write(f"  - {file}\n")

        f.write(f"\n[+] Grupos de Duplicatas Exatas: {len(exact_duplicates)}\n")
        for i, (hash_val, files) in enumerate(exact_duplicates.items()):
            f.write(f"  Grupo {i+1}:\n")
            for file in files:
                f.write(f"    - {file}\n")

        f.write(f"\n[+] Grupos de Duplicatas Visuais (Distância <= {p_hash_threshold}): {len(visual_duplicates)}\n")
        for i, (original, duplicates) in enumerate(visual_duplicates.items()):
            f.write(f"  Grupo {i+1} (similar a {os.path.basename(original)}):\n")
            for file in duplicates:
                f.write(f"    - {file}\n")

        f.write("\n--- Fim do Relatório ---\n")

    print(f"\n✅ Relatório salvo em: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Encontra imagens duplicatas (exatas e visuais) em um dataset.")
    parser.add_argument("directory", type=str, help="Caminho para a pasta com as imagens.")
    parser.add_argument("--phash_threshold", type=int, default=5, 
                        help="Limiar de distância para duplicatas visuais. Valores menores são mais estritos. Padrão: 5.")
    parser.add_argument("--output", type=str, default="relatorio_duplicatas.txt",
                        help="Nome do arquivo de saída para o relatório (padrão: relatorio_duplicatas.txt).")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Erro: O diretório '{args.directory}' não foi encontrado.")
    else:
        find_duplicate_images(
            args.directory, 
            args.phash_threshold,
            args.output
        )
