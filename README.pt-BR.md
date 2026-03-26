# Aprendizado Federado para Classificação de Tumores Cerebrais com Test-Time Augmentation

Este repositório apresenta uma estrutura de aprendizado federado aplicada à classificação de tumores cerebrais utilizando dois clientes distintos: um com imagens **originais** e outro com imagens **pré-processadas**. O objetivo principal é comparar o desempenho dos modelos locais e do modelo global, avaliando o impacto do pré-processamento e do uso de TTA.

---

## Publicação Associada

Este repositório está associado ao seguinte artigo:

Exploiting Test-Time Augmentation in Federated Learning for Brain Tumor MRI Classification  
Thamara Leandra de Deus Melo, Rodrigo Moreira, Larissa Moreira, André Backes  
Proceedings of the 21st International Conference on Computer Vision Theory and Applications (VISAPP), 2026  
DOI: https://doi.org/10.5220/0014391000004084

---

## Pré-processamento

O pré-processamento das imagens é realizado por um script dedicado preprocess.py, responsável por:

* Redimensionamento com ou sem preservação de proporção
* Conversão opcional para escala de cinza
* Forçar RGB para compatibilidade com arquiteturas como ResNet18
* Aplicação de filtros de redução de ruído (Gaussiano/Bilateral)
* Aplicação opcional de CLAHE para realce de contraste
* Padronização de formato (JPG/PNG)
* Reorganização dos diretórios por classes

Essa etapa é executada antes do treinamento federado e aplicada exclusivamente ao cliente pré-processado.

---

## Objetivo do Projeto

Desenvolver e avaliar um modelo de classificação de tumores cerebrais utilizando Aprendizado Federado (FL) com dois clientes:

* Cliente 1: Imagens originais (dataset Kaggle)
* Cliente 2: Imagens pré-processadas com:

  * Redimensionamento
  * Conversão para escala de cinza
  * Normalização
  * Filtros de suavização
  * Equalização de histograma

O projeto busca responder:

* O pré-processamento melhora o desempenho local?
* O modelo federado é mais robusto?
* Como o FL se comporta com dados heterogêneos?

---

## Arquitetura do Sistema

O experimento utiliza uma arquitetura baseada em FedAvg:

* Servidor: Agrega os pesos dos clientes
* Clientes (2):

  * Cliente com dados originais
  * Cliente com dados pré-processados
* Modelo: ResNet18

Cada cliente treina localmente e envia seus pesos ao servidor, que realiza a agregação e redistribuição.

---

## Requisitos

* Python 3.8+
* PyTorch
* NumPy
* Matplotlib
* Flower (ou outro framework de FL)
* Scikit-image / OpenCV

---

## Como Executar

### 1. Preparar o dataset

data/original/  
data/preprocessed/

### 2. Iniciar o servidor

python federated/server.py

### 3. Iniciar os clientes

python -c "from federated.client import start_client; start_client('dataset_kaggle/Training')"  
python -c "from federated.client import start_client; start_client('dataset_kaggle_preprocessed/Train')"

---

## Test-Time Augmentation (TTA)

Após o treinamento federado, o modelo global é avaliado utilizando TTA (Test-Time Augmentation), aumentando a robustez das predições.

As transformações incluem:

* Rotação
* Flip horizontal/vertical
* Pequenas perturbações (ruído, brilho)

Processo:

1. Gerar múltiplas variações da imagem de teste
2. Realizar inferência em cada variação
3. Agregar as predições (média ou votação)

---

## Resultados Esperados

O estudo permite comparar:

* Acurácia dos modelos locais
* Acurácia do modelo federado
* Comportamento de convergência
* Impacto do pré-processamento e do TTA

---

## Observações Finais

Este repositório foi desenvolvido para fins acadêmicos, permitindo experimentos controlados com diferentes abordagens de dados em aprendizado federado.
