# Federated Learning para Classificação de Tumores Cerebrais com TTA para Detecção de Tumores Cerebrais

Este repositório apresenta uma estrutura de treinamento federado aplicada à detecção de tumores cerebrais utilizando dois clientes distintos: um com imagens **originais** e outro com imagens **pré-processadas**. O objetivo principal é comparar o desempenho dos modelos locais e do modelo global federado, avaliando o impacto do pré-processamento no aprendizado distribuído.

---

##  Pré-processamento

O pré-processamento das imagens é realizado por um script dedicado preprocess.py, responsável por:

* Redimensionamento com preservação ou não de proporção
* Conversão para tons de cinza (opcional)
* Forçar RGB quando necessário para backbones como ResNet18
* Aplicação de filtros de redução de ruído (Gaussian/Bilateral)
* Aplicação opcional de CLAHE para realce de contraste
* Padronização de formato (JPG/PNG) e qualidade
* Reorganização dos diretórios por classes

Esse script é executado antes da etapa de treinamento federado e alimenta exclusivamente o **cliente pré-processado**.
Este repositório utiliza um script dedicado de pré-processamento para preparar as imagens antes do treinamento. Esse script realiza tarefas como normalização, conversão, filtragem e organização das classes, sendo aplicado ao cliente que utiliza imagens pré-processadas.

---

## Objetivo do Projeto

Desenvolver e avaliar um modelo de **classificação de tumores cerebrais** utilizando **Aprendizado Federado (FL)** com dois clientes:

* **Cliente 1**: Imagens **originais** do dataset Kaggle.
* **Cliente 2**: Imagens **pré-processadas** com as seguintes etapas:

  * Redimensionamento
  * Conversão para escala de cinza
  * Normalização
  * Filtros de suavização
  * Equalização de histograma
  * Segmentação com U-Net

O projeto busca responder:

* O pré-processamento melhora a qualidade do modelo local?
* O modelo federado é mais robusto do que os modelos individuais?
* Como o FL se comporta com dados heterogêneos (original vs pré-processado)?

---

## Arquitetura do Sistema

O experimento utiliza uma arquitetura baseada em **FedAvg** com os seguintes componentes:

* **Servidor**: Agrega os pesos enviados pelos clientes.
* **Clientes (2)**:

  * Cliente Original
  * Cliente Pré-processado
* **Arquitetura de Rede**: ResNet18

Cada cliente treina localmente usando seu dataset e envia os pesos ao servidor, que realiza a agregação e redistribuição.

---

## Pré-requisitos

* Python 3.8+
* PyTorch
* NumPy
* Matplotlib
* Flower (ou outro framework FL)
* Scikit-image / OpenCV (para pré-processamento)

---

## Como Executar

### **1. Preparar o dataset**

Coloque as imagens nas pastas:

```
data/original/
data/preprocessed/
```

### **2. Iniciar o servidor**

```bash
python federated/server.py
```

### **3. Iniciar os clientes (cada um em um terminal)**

```bash
python -c "from federated.client import start_client; start_client('dataset_kaggle/Training')"
python -c "from federated.client import start_client; start_client('dataset_kaggle_preprocessed\\Train')"
```
---

## Aumento em Tempo de Teste (TTA)

Após o treinamento federado, o modelo global passa por uma etapa adicional de avaliação utilizando **TTA (Test-Time Augmentation)**. O TTA aumenta a robustez e estabilidade das predições ao gerar múltiplas versões transformadas da mesma imagem de teste e combinar suas saídas.

As transformações incluem:

* Rotação
* Flip horizontal/vertical
* Perturbações leves (ruído, alteração de brilho)

O processo segue:

1. Para cada imagem de teste, são geradas *n* variações.
2. O modelo prevê cada versão independentemente.
3. As predições são agregadas (média ou votação majoritária).

Isso reduz a variabilidade e melhora a confiabilidade do modelo global em cenários reais.

## Resultados Esperados

O estudo permite comparar:

* Acurácia dos modelos locais
* Acurácia do modelo global federado
* Gráficos de perda e convergência
* Impacto do pré-processamento no FL

## Observações Finais

Este repositório foi desenvolvido para fins **acadêmicos**, permitindo experimentos controlados com duas abordagens distintas de preparação de dados em um cenário de aprendizado federado.
