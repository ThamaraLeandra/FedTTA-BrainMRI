import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Caminho da imagem original (substitua pelo caminho real)
image_path = "dataset_kaggle_preprocessed/Testing/glioma/Te-gl_0012.jpg"

# Carrega a imagem
image = Image.open(image_path).convert("RGB")

# Transformação base (redimensiona e normaliza)
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Transforma a imagem original (sem TTA)
original_tensor = base_transform(image)

# Define as transformações TTA (iguais às do seu treinamento)
tta_transforms = transforms.Compose([
    transforms.RandomApply([
        transforms.RandomRotation(degrees=10),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05))
    ], p=1.0),
    transforms.Resize((224, 224))
])

# Gera 3 variações com TTA
tta_images = []
for i in range(3):
    aug_image = tta_transforms(image)
    tta_images.append(aug_image)

# Função auxiliar para converter tensores para exibição
def tensor_to_pil(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.clone().detach().cpu()
        tensor = tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        tensor = torch.clamp(tensor, 0, 1)
        img = transforms.ToPILImage()(tensor)
        return img
    return tensor

# Cria uma figura com 4 imagens lado a lado
fig, axes = plt.subplots(1, 4, figsize=(12, 4))

# Mostra a imagem original
axes[0].imshow(image)
axes[0].set_title("Original", fontsize=10)

# Mostra as 3 variações
for i, aug_img in enumerate(tta_images):
    axes[i+1].imshow(aug_img)
    axes[i+1].set_title(f"TTA {i+1}", fontsize=10)

# Remove eixos
for ax in axes:
    ax.axis("off")

plt.tight_layout()

# Salva a figura em PDF
plt.savefig("figuras/tta_examples.pdf", format="pdf", bbox_inches="tight")

print("✅ Figura gerada com sucesso: figuras/tta_examples.pdf")
