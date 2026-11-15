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
├── src/                    # Componentes reutilizáveis da aplicação
│   ├── video.py            # Leitura de vídeo
│   ├── calibration.py      # Calibração e marcações do campo
│   ├── detection.py        # Detecção de jogadores
│   ├── tracking.py         # Associação de detecções
│   ├── projection.py       # Projeção no campo
│   ├── trajectory.py       # Construção de trajetórias
│   ├── models.py           # Dados compartilhados entre etapas
│   └── field_geometry.py   # Geometria e parâmetros da câmera
├── tools/                  # Scripts executáveis
│   ├── create_dataset.py   # Geração do dataset YOLO
│   ├── process_video.py    # Rastreamento e projeção em vídeo
│   └── annotate_field.py   # Marcação, calibração e projeção
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

### Leitura de vídeo e calibração como componentes

O pacote `src` oferece `LeitorVideo`, `CalibradorCampo` e as classes de dados
`MetadadosVideo` e `CalibracaoCampo`. A partir da raiz do projeto:

```python
from src import CalibradorCampo, LeitorVideo

with LeitorVideo("partida.mp4") as video:
    dados = video.metadados()
    quadro = video.ler_quadro(60)  # Numeração começa em 1.
    # video.ler_quadros() fornece (número, tempo em segundos, quadro).

calibrador = CalibradorCampo()
marcacoes = calibrador.carregar_marcacoes("campo.yaml")
calibracao = calibrador.calibrar(marcacoes, (dados.largura, dados.altura),
                                 modelo_camera="radial")
calibracao.salvar("camera.yaml")
# Para carregar posteriormente, informe a resolução do vídeo:
calibracao = calibrador.carregar_calibracao("camera.yaml", (dados.largura, dados.altura))
```

A calibração reutiliza o ajuste geométrico descrito abaixo, incluindo suas
limitações e a necessidade de marcações apropriadas. `erro_rms` e
`atingiu_limite` estão disponíveis após o ajuste, mas não são persistidos no
YAML de câmera existente e retornam `None` ao carregá-lo.

### Projeção de jogadores no campo

`tools/annotate_field.py` permite marcar o campo, calibrar a câmera e,
opcionalmente, detectar e projetar jogadores em um quadro. Para usar marcações
já salvas, sem abrir o editor, e mostrar o resultado:

```bash
python tools/annotate_field.py \
    --video SoccerTrackv2/mot/clips/132877.mp4 \
    --output SoccerTrackv2/mot/field/132877.yaml --use-existing \
    --model weights/best.pt --frame 60 --camera-model fisheye
```

Para criar marcações, omita `--use-existing`; para editá-las, use `--edit`.
Sem `--model`, a detecção e a visualização não são executadas. A calibração
é executada quando `--model` ou `--camera-params` é informado.

O frame é numerado a partir de 1. Os pontos do YAML devem pertencer ao mesmo
frame (ou a uma câmera fixa), com as mesmas dimensões do vídeo. O modelo deve
ter uma classe chamada `player` ou `person`; use `--class-id` para especificar
outra classe. `--conf` define o limiar de confiança (padrão 0,25).

Acrescente `--camera-params caminho/camera.yaml` para salvar os parâmetros
da lente, os coeficientes de distorção e a homografia. O arquivo é sobrescrito se já
existir. O argumento é de saída: cada execução ainda estima os parâmetros a
partir das marcações. No modelo radial, `division_k1` é o coeficiente do modelo
de divisão usado pelo script, não um coeficiente radial OpenCV. No fisheye,
o modelo equidistante estima a distância focal, mas fixa os quatro coeficientes
adicionais em zero; eles não são estimados a partir das marcações. A matriz da
câmera é calculada com centro óptico fixo e distâncias focais iguais.

Na seleção de campo de `tools/annotate_field.py`, use `C` para marcar a linha central,
incluindo seus encontros com as linhas superior e inferior. Ao confirmar, os
cantos compartilhados são obtidos pela média dos extremos marcados nas linhas
adjacentes, e os extremos da linha central são inseridos nas linhas superior e
inferior. Quando o YAML contém `center`, o ajuste também usa essa linha e
prioriza sua correspondência com `x = 0`. Como os pontos podem não ser
perfeitamente compatíveis com o modelo de lente, isso pode deslocar os limites
projetados em relação a `x = ±1`. YAMLs anteriores sem `center` continuam aceitos.

Para adicionar o ponto central a um YAML já marcado, edite o mesmo frame usado
na marcação original:

```bash
python tools/annotate_field.py --video SoccerTrackv2/mot/clips/132877.mp4 \
    --output SoccerTrackv2/mot/field/132877.yaml --frame 60 --edit
```

Pressione `P`, clique no ponto central e pressione Enter para salvar. O campo
`center_spot: [x, y]` é opcional. Quando presente junto à linha `center`, a
projeção o associa a `(0, 0)` e mantém as linhas superior e inferior em
`y = -1` e `y = 1` no modelo ajustado; isto pode alterar as coordenadas laterais.

`--camera-model` aceita `radial` (padrão, modelo de divisão com um coeficiente)
ou `fisheye` (modelo equidistante com distância focal ajustada). Ambos fixam
o centro óptico no centro da imagem e aplicam uma homografia após a correção.
O fisheye é uma aproximação, não uma calibração completa de lente; suas
projeções exigem ângulos inferiores a 90 graus em relação ao eixo óptico.
A saída inclui o erro RMS de cada linha e avisa
se o ajuste atinge o limite do modelo. Isso não garante precisão: valide com
marcações internas conhecidas e considere calibrar a lente se o ajuste for ruim.
Uma linha com apenas dois pontos não informa sua curvatura. Pontos projetados
fora do campo são exibidos sem limitar as coordenadas.

### Vídeo com jogadores projetados

Para criar ou editar a marcação sem executar o modelo:

```bash
python tools/annotate_field.py --video caminho/video.mp4 \
    --output caminho/campo.yaml --frame 60
```

Acrescente `--edit` para carregar e editar um YAML existente. Para gerar o
YAML de câmera a partir de marcações existentes sem executar o detector:

```bash
python tools/annotate_field.py --video caminho/video.mp4 \
    --output caminho/campo.yaml --frame 60 --use-existing \
    --camera-model radial --camera-params caminho/camera.yaml
```

Depois execute:

```bash
python tools/process_video.py --video caminho/video.mp4 --weights weights/best.pt \
    --camera-params caminho/camera.yaml \
    --output runs/inference
```

Acrescente `--trajectories caminho/trajetorias.json` para salvar as trajetórias
construídas em JSON. Cada posição contém o instante em segundos e coordenadas
normalizadas; pontos fora dos limites permanecem nos dados. Os instantes são
calculados a partir do número do quadro e do FPS informado pelo vídeo.

O resultado `<nome>_field.mp4` mostra o vídeo e o campo lado a lado. Os números
são IDs de rastreamento, que podem mudar após oclusões. Por padrão é usado o
FastTracker (`fasttrack.yaml`); use `--tracker bytetrack.yaml` para comparar com o ByteTrack.
Também é possível informar um YAML próprio para um desses dois tipos de rastreador. Para preservar os
dois vídeos, use diretórios `--output` diferentes. O ponto projetado é o centro da borda inferior de cada caixa, como aproximação dos pés.
As posições exibidas vêm das caixas retornadas pelo rastreador, e podem desaparecer durante uma oclusão.
Detecções projetadas fora do campo são identificadas como `outside`. O limiar
padrão de detecção é `--conf 0.1`; `--show` exibe o processamento (pressione `q` para parar). Os parâmetros
da lente e a homografia são lidos do YAML de câmera. Arquivos de câmera antigos
sem homografia precisam ser gerados novamente com `annotate_field.py`.
As marcações pressupõem câmera fixa durante o vídeo, e os parâmetros devem ter
sido calculados para a resolução do vídeo. O modelo fisheye suportado é o
modelo equidistante simplificado com coeficientes adicionais nulos.

## Dataset

O projeto utiliza o dataset [SoccerTrack v2](https://atomscott.github.io/SoccerTrack-v2/), desenvolvido para tarefas de reconstrução do estado de jogo (Game State Reconstruction).

O dataset contém informações relacionadas às partidas de futebol, incluindo:

- Vídeos das partidas.
- Anotações de jogadores.
- Identificadores individuais (IDs).
- Informações geométricas e de calibração da câmera.
- Dados necessários para reconstrução espacial do campo.

A organização dos dados deve preservar a estrutura original do dataset sempre que possível.

### Geração do dataset YOLO

O gerador utiliza os clipes MOT e cria `images/{train,val,test}`,
`labels/{train,val,test}`, `dataset.yaml` e `report.json`. A divisão padrão é
por partida: os seis vídeos de treinamento oficiais para treino, os dois de
challenge para validação e os dois de teste para teste. Isso evita dividir
frames da mesma partida entre conjuntos. São necessárias as anotações e os dez
vídeos; o gerador interrompe a execução se algum arquivo estiver ausente.

```bash
python tools/create_dataset.py --mot-root SoccerTrackv2/mot --output data/dataset --interval 36
```

No notebook de geração, `FRAME_INTERVAL` está configurado como 10; ajuste-o
conforme a amostragem desejada. No comando acima, `--interval 36` seleciona um frame a cada 36, começando no índice zero. A
saída deve ser uma pasta inexistente ou vazia; escolha outra pasta após uma
execução interrompida ou remova manualmente os arquivos gerados antes de
recomeçar. O YAML contém o caminho absoluto da saída e deve ser atualizado se
o dataset for movido. O relatório registra as partidas, bases dos índices e
quantidades geradas. Para uma divisão diferente, passe `--splits arquivo.json`,
com listas não vazias de IDs nas chaves `train`, `val` e `test`.

A cópia local documentada em `SoccerTrackv2/mot/README.md` usa frames iniciados
em zero em todas as partidas. O arquivo original dos organizadores para 118578
começa em um: ao usá-lo sem normalização, acrescente `--one-based 118578`.
A base não é inferida automaticamente. No Colab, clone o repositório e ajuste
os caminhos e `FRAME_BASES` no notebook de geração. O notebook de treinamento
deve apontar `DATASET` para a mesma pasta de saída.

## Desenvolvimento e experimentos

### Executar notebooks no Google Colab

Abra um notebook diretamente da branch `main` do GitHub:

| Notebook | Abrir no Colab |
|---|---|
| Criação do dataset | [Abrir no Colab](https://colab.research.google.com/github/gvalcantara/ProjetoPDS-AI/blob/main/notebooks/00_dataset_creator.ipynb) |
| Treinamento | [Abrir no Colab](https://colab.research.google.com/github/gvalcantara/ProjetoPDS-AI/blob/main/notebooks/01_train.ipynb) |
| Teste em vídeo | [Abrir no Colab](https://colab.research.google.com/github/gvalcantara/ProjetoPDS-AI/blob/main/notebooks/02_test.ipynb) |

Os links abrem a versão publicada no GitHub, não alterações locais. Para usar
outra branch, substitua `main` pelo nome dela na URL. No Colab, selecione uma
GPU em **Ambiente de execução → Alterar tipo de ambiente de execução** para o
treinamento e execute as células em ordem. Autorize a montagem do Google Drive
quando solicitada e ajuste os caminhos dos dados e dos pesos para seu Drive.
No notebook de teste, os caminhos locais e a pasta de saída também precisam ser
ajustados para o ambiente Colab; monte o Drive antes de acessá-lo.

O ambiente do Colab é temporário: guarde dados, pesos e resultados no Drive.
Para preservar alterações feitas no notebook, use **Arquivo → Salvar uma cópia
no Drive** ou salve uma cópia no GitHub. A abertura do notebook não inicia o
treinamento automaticamente; ele é executado ao rodar as células.

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

Os testes automatizados são desenvolvidos utilizando `pytest`. O teste de geração
do dataset versionado na branch cobre a amostragem, os índices de frame, os
rótulos YOLO e a rejeição de divisões duplicadas e caixas inválidas.

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
