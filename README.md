# Sistema de Rastreamento de Posição de Jogadores de Futebol

Projeto desenvolvido no contexto da disciplina **Projeto Nível I em Computação e Processamento de Sinais I**, com o objetivo de desenvolver um sistema de visão computacional para detecção e rastreamento de jogadores de futebol a partir de vídeos de partidas.

## Sobre o projeto

O projeto investiga técnicas de processamento digital de sinais, visão computacional e aprendizado de máquina aplicadas à análise de partidas de futebol.

A proposta é desenvolver um sistema capaz de detectar jogadores em vídeos, acompanhar suas trajetórias ao longo do tempo e estimar suas posições no campo de futebol.

O desenvolvimento é organizado em diferentes etapas:

- **Detecção de objetos:** identificação dos jogadores nas imagens utilizando redes neurais convolucionais, com destaque para a arquitetura YOLO.
- **Rastreamento de jogadores:** associação das detecções entre frames para manter a identidade dos jogadores ao longo do vídeo.
- **Estimativa de posição:** transformação das coordenadas da imagem para coordenadas do campo utilizando homografia.
- **Análise espacial:** visualização e análise das posições e trajetórias dos jogadores no campo.

## Estrutura do repositório

```text
.
├── data/                   # Dados e arquivos auxiliares
├── SoccerTrackv2/          # Dataset SoccerTrack v2
├── weights/                # Pesos dos modelos treinados
├── runs/                   # Resultados de treinamento e experimentos
│
├── src/
│   └── tools/              # Ferramentas auxiliares
│       └── yolo_video.py   # Inferência YOLO em vídeos
│
├── tests/                  # Testes automatizados
│
├── .venv/                  # Ambiente virtual Python (não versionado)
├── requirements.txt        # Dependências do projeto
└── README.md
```

> A estrutura pode ser expandida conforme novas funcionalidades forem implementadas.

## Requisitos

- Python 3.10 ou superior.
- Git.
- Ambiente virtual Python (`venv`).
- PyTorch com suporte à CPU ou CUDA.
- GPU NVIDIA recomendada para treinamento e inferência mais rápidos.

## Configuração do ambiente

### 1. Clonar o repositório

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_NAME>
```

### 2. Criar o ambiente virtual

```bash
python -m venv .venv
```

### 3. Ativar o ambiente virtual

**Linux / macOS:**

```bash
source .venv/bin/activate
```

**Windows:**

```powershell
.venv\Scripts\activate
```

### 4. Instalar as dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Verificar a instalação do PyTorch

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA disponível:", torch.cuda.is_available())
```

> A instalação do PyTorch com CUDA depende da versão do CUDA e dos drivers disponíveis no sistema. Consulte a documentação oficial do PyTorch para selecionar a instalação adequada.

## Ferramentas

### YOLO Video Detection

Ferramenta de linha de comando para executar modelos YOLO sobre vídeos, desenhando bounding boxes nas detecções e salvando o resultado.

**Localização:** `src/tools/yolo_video.py`

#### Utilização

```bash
python src/tools/yolo_video.py \
    --weights weights/best.pt \
    --video data/video.mp4
```

#### Parâmetros

| Parâmetro | Obrigatório | Padrão | Descrição |
|---|---|---|---|
| `--weights` | Sim | — | Caminho para os pesos do modelo YOLO. |
| `--video` | Sim | — | Caminho para o vídeo de entrada. |
| `--show` | Não | `False` | Exibe o vídeo durante a inferência. |
| `--conf` | Não | `0.25` | Confiança mínima das detecções. |
| `--imgsz` | Não | `640` | Resolução de entrada da rede. |
| `--output` | Não | `yolo_testes` | Diretório para salvar os resultados. |

#### Exemplos

**Exibir o vídeo durante a inferência:**

```bash
python src/tools/yolo_video.py \
    --weights weights/best.pt \
    --video data/video.mp4 \
    --show
```

**Alterar o limiar de confiança e a resolução:**

```bash
python src/tools/yolo_video.py \
    --weights weights/best.pt \
    --video data/video.mp4 \
    --conf 0.4 \
    --imgsz 1280
```

**Definir diretório de saída:**

```bash
python src/tools/yolo_video.py \
    --weights weights/best.pt \
    --video data/video.mp4 \
    --output runs/inference
```

## Dataset

O projeto utiliza o dataset [SoccerTrack v2](https://atomscott.github.io/SoccerTrack-v2/), desenvolvido para tarefas de reconstrução do estado de jogo (Game State Reconstruction).

O dataset contém informações relacionadas às partidas de futebol, incluindo:

- Vídeos das partidas.
- Anotações de jogadores.
- Identificadores individuais (IDs).
- Informações geométricas e de calibração da câmera.
- Dados necessários para reconstrução espacial do campo.

A organização dos dados deve preservar a estrutura original do dataset sempre que possível.

## Desenvolvimento e experimentos

Os experimentos de aprendizado de máquina são organizados no diretório `runs/`.

Recomenda-se manter os seguintes registros para cada experimento:

- Arquitetura utilizada.
- Configuração de treinamento.
- Dataset e divisão utilizados.
- Métricas de avaliação.
- Pesos resultantes.
- Observações sobre os resultados.

Os pesos dos modelos utilizados na inferência devem ser armazenados em `weights/`.

## Testes

Os testes automatizados são desenvolvidos utilizando `pytest`.

Para executar todos os testes:

```bash
pytest
```

## Dependências

As dependências do projeto estão centralizadas em `requirements.txt`.

As principais bibliotecas utilizadas são:

| Biblioteca | Aplicação |
|---|---|
| NumPy | Computação numérica. |
| SciPy | Processamento científico. |
| OpenCV | Processamento de imagens e vídeos. |
| Matplotlib | Visualização de dados. |
| Mplsoccer | Visualização de dados em campos de futebol. |
| Pandas | Manipulação de dados tabulares. |
| Scikit-learn | Algoritmos de aprendizado de máquina. |
| PyTorch | Construção e treinamento de redes neurais. |
| Ultralytics | Implementação e treinamento de modelos YOLO. |
| MoviePy | Processamento de vídeos. |
| Loguru | Logging. |
| TQDM | Barras de progresso. |

## Referências

- [SoccerTrack v2 — Full-Pitch Soccer Dataset for Game State Reconstruction](https://atomscott.github.io/SoccerTrack-v2/)
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [OpenCV Documentation](https://docs.opencv.org/)

## Licença

Projeto desenvolvido para fins acadêmicos no contexto da disciplina Projeto Nível I em Computação e Processamento de Sinais I.