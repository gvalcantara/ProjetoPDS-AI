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

### Instalação como pacote

Para gerar um wheel com os componentes de `src/`, execute na raiz deste
repositório:

```bash
python -m pip install build
python -m build --wheel
```

O arquivo é criado em `dist/` com a versão definida em `pyproject.toml`.
Para testar a versão atual em um ambiente virtual:

```bash
python -m pip install dist/hattrick_ai-0.1.5-py3-none-any.whl
```

Para disponibilizar o wheel ao backend do ProjetoPDS-App, copie-o para a pasta
de wheels do app e atualize a entrada correspondente em
`ProjetoPDS-App/backend/requirements.txt` para o novo nome:

```bash
cp dist/hattrick_ai-0.1.5-py3-none-any.whl \
    ../ProjetoPDS-App/backend/wheels/
```

Depois, a partir de `ProjetoPDS-App/backend`, instale as dependências no
ambiente virtual do backend para aplicar a atualização:

```bash
python -m pip install -r requirements.txt
```

Ao alterar a versão em `pyproject.toml`, substitua `0.1.5` nos comandos e na
entrada de `requirements.txt` pela versão gerada. O caminho relativo de cópia
pressupõe que os dois repositórios estão lado a lado.

O pacote instalado pode ser importado como `hattrick_ai`:

```python
from hattrick_ai import CalibradorCampo, LeitorVideo
```

O wheel inclui os pesos `player_yolo26s_1280_baseline` usados pelo extrator e
`players_n_1280_baseline` usado experimentalmente pelo BoT-SORT, mas não os
scripts de `tools/` nem datasets.
Para uso com CUDA, instale a versão adequada do PyTorch para o ambiente.

Para obter as trajetórias de um vídeo com uma calibração existente:

```python
from hattrick_ai import ExtratorTrajetorias

extrator = ExtratorTrajetorias("camera.yaml")
trajetorias = extrator.processar("partida.mp4")
for trajetoria in trajetorias:
    print(trajetoria.id_rastreamento, trajetoria.posicoes)
```

Cada posição contém o instante em segundos e as coordenadas normalizadas do
campo; os limites do campo são -1 e 1. A calibração deve corresponder à
resolução do vídeo. Um novo rastreador é criado a cada chamada de `processar`.

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
BoT-SORT com ReID (`botsort.yaml`), usando experimentalmente o próprio modelo de detecção como extrator de características de aparência. Não é um modelo treinado especificamente para reidentificação; a qualidade deve ser avaliada em vídeos reais. Use `--tracker bytetrack.yaml` ou `--tracker fasttrack.yaml` para comparação.
Também é possível informar um YAML próprio para esses tipos de rastreador. Para preservar os
dois vídeos, use diretórios `--output` diferentes. O ponto projetado é o centro da borda inferior de cada caixa, como aproximação dos pés.
As posições exibidas vêm das caixas retornadas pelo rastreador, e podem desaparecer durante uma oclusão.
Detecções projetadas fora do campo são identificadas como `outside`. O limiar
padrão de detecção é `--conf 0.1`; `--show` exibe o processamento (pressione `q` para parar). Os parâmetros
da lente e a homografia são lidos do YAML de câmera. Arquivos de câmera antigos
sem homografia precisam ser gerados novamente com `annotate_field.py`.
As marcações pressupõem câmera fixa durante o vídeo, e os parâmetros devem ter
sido calculados para a resolução do vídeo. O modelo fisheye suportado é o
modelo equidistante simplificado com coeficientes adicionais nulos.

### Comparação com ground truth GSR

A comparação usa as caixas na imagem de jogadores e goleiros do JSON GSR:

```bash
python tools/process_video.py --video caminho/video.mp4 --weights caminho/pesos.pt \
    --camera-params caminho/camera.yaml --gt caminho/partida_gsr.json \
    --gt-frame-offset 0 --gt-iou 0.5 --output runs/inference
```

O deslocamento é **obrigatório** e deve ser confirmado a partir da origem do
vídeo: o quadro 1 do vídeo corresponde ao quadro `1 + offset` do GSR. Use `0`
somente se ambos começam no mesmo quadro; para um recorte que começa no quadro
GSR 1001, use `1000`. Nome de arquivo igual não garante alinhamento. O vídeo
deve ter a mesma resolução e FPS das imagens de referência, e o intervalo deve
existir integralmente no JSON. Vídeos cortados ou redimensionados requerem
alinhamento e transformação apropriados antes da comparação.

O JSON é lido incrementalmente com `ijson`, uma vez por execução, e somente as
anotações do intervalo selecionado são indexadas em um arquivo SQLite temporário,
removido ao final. Arquivos grandes ainda podem demorar para percorrer e exigem
espaço temporário em disco. A exportação de trajetórias só ocupa memória quando
`--trajectories` é solicitado.

O relatório `<video>_gt.json` contém quadros avaliados e não rotulados,
correspondências, falsos positivos e negativos, precisão, revocação, IoU médio,
trocas de ID e MOTA aproximado. `--gt-report` altera o caminho do relatório.
A associação em cada quadro usa o algoritmo húngaro e o limiar de IoU escolhido;
quadros sem `has_labeled_person` são ignorados. Uma troca é contada quando um
jogador de referência volta a ser associado a outro ID, inclusive após lacunas.
O MOTA apresentado usa essa regra e não é uma implementação oficial de benchmark:
não preserva associações anteriores antes do casamento nem aplica regiões de
ignoração, filtros de visibilidade ou a associação global de métricas como IDF1.
Outros objetos, como árbitros, não entram no GT de jogadores e podem ser contados
como falsos positivos. A comparação é feita em pixels da imagem e não mede o
erro de posição no campo. Resultados após interrupção com `q` cobrem somente os
quadros processados; métricas sem denominador são `null`.

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
| Baseline GSR do SoccerTrack | `notebooks/03_soccertrack_gsr_baseline.ipynb` |

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

### Baseline GSR oficial

O notebook `notebooks/03_soccertrack_gsr_baseline.ipynb` clona uma versão fixa do SoccerTrack v2 e executa seus módulos GSR de treinamento e avaliação. Ele exige dados GSR, vídeos panorâmicos calibrados, anotações MOT e homografias compatíveis para ambos os tempos das partidas selecionadas. O notebook verifica arquivos e registra saídas no caminho configurado. O baseline oficial treina apenas com o primeiro dataset por tempo gerado e usa um placeholder de ReID; não é uma rede de reidentificação treinada.

#### Download e preparação dos dados GSR

O dataset publicado requer acesso autorizado no [Hugging Face](https://huggingface.co/datasets/atomscott/soccertrack-v2). Instale `huggingface_hub` e autentique-se com `hf auth login`. Faça primeiro uma simulação para verificar tamanho e disponibilidade; as listas devem coincidir com as configuradas no notebook:

```bash
python tools/download_soccertrack_gsr.py --dest data/soccertrack-raw \
  --train 117092 --val 118578 --test 128057 --dry-run
```

Retire `--dry-run` para baixar. O download inclui os dois vídeos de cada partida, dados brutos de calibração e rastreamento e GSR das partidas de teste. São arquivos grandes; use uma revisão fixa com `--revision` para reproduzir o download. É possível acrescentar outras partidas às listas, mantendo-as disjuntas. Não são baixados nem reutilizados os MOT de clipes curtos como anotações de jogos completos.

A preparação usa a função de calibração original do SoccerTrack e requer um checkout do SoccerTrack v2 com suas dependências instaladas no mesmo Python. São necessários **arquivos MOT gerados para os vídeos completos calibrados**, em `MOT_ROOT/<partida>/<1st_half|2nd_half>/gt/gt.txt`, com frames alinhados ao vídeo e indexação compatível com o gerador do baseline. Eles não estão presentes como arquivos por tempo de jogo no dataset publicado. Os dados brutos de rastreamento não são, por si só, caixas de referência prontas: consulte o [processamento oficial](https://github.com/AtomScott/SoccerTrack-v2/blob/main/docs/ground_truth_creation.md) para geração e validação. Não renomeie os arquivos MOT dos clipes de quatro minutos como se fossem os tempos completos.

```bash
python tools/prepare_soccertrack_gsr.py \
  --source data/soccertrack-raw --dest data/soccertrack-v2-data \
  --soccertrack /caminho/SoccerTrack-v2 --mot-root /caminho/mot-calibrado \
  --train 117092 --val 118578 --test 128057
```

O script interrompe antes de processar se faltar alguma entrada e recusa sobrescrever resultados. Calibrar vídeos inteiros pode demorar muito e exigir armazenamento adicional. Confirme o alinhamento temporal, a base dos frames e a geometria das caixas manualmente: a presença de arquivos não garante sua correção. No notebook, ajuste `DATA_ROOT`, `TRAIN_MATCHES`, `VAL_MATCHES` e `TEST_MATCHES` para os mesmos valores usados nos scripts. O baseline treina o detector usando apenas o primeiro dataset por tempo criado e não treina ReID.

### Treinamento YOLO e ReID com os clipes MOT

Para treinar a YOLO e preparar recortes para ReID, basta baixar `mot/` do SoccerTrack v2. O notebook GSR acima é um experimento diferente e exige os dados adicionais indicados nele; os scripts GSR não são necessários para este fluxo MOT.

```bash
pip install -U huggingface_hub
hf auth login
python tools/download_soccertrack_mot.py --dest data/soccertrack --dry-run
python tools/download_soccertrack_mot.py --dest data/soccertrack
```

Solicite acesso ao dataset no Hugging Face se necessário. `--match ID` pode ser repetido para baixar partidas específicas. Os arquivos ficam em `data/soccertrack/mot/`, incluindo `clips/`. As anotações baixadas são originais: confirme a base de frames; a documentação da cópia local indica que `118578` original começa em 1. O parâmetro `--one-based 118578` abaixo aplica-se ao arquivo original, não à cópia local já normalizada.

```bash
python tools/create_dataset.py --mot-root data/soccertrack/mot --output data/yolo --interval 36 --one-based 118578
python tools/create_reid_dataset.py --mot-root data/soccertrack/mot --output data/reid --interval 36 --one-based 118578
```

Ambos aceitam `--splits arquivo.json` com listas não vazias de partidas `train`, `val` e `test`, sem partidas repetidas. O segundo grava recortes em `<split>/<partida>_<id>/` e um `metadata.csv`; a identidade é única por partida, não global entre partidas. A consistência dos IDs deve ser conferida antes do treinamento. Os recortes são dados para ReID: o script não treina uma rede. Os clipes duram cerca de quatro minutos e não permitem, por si só, avaliar persistência durante uma partida inteira. O tamanho de entrada da YOLO permanece 1280.

Para gerar os dois conjuntos com a mesma divisão e amostragem em um único comando:

```bash
python tools/prepare_yolo_reid.py --yolo --reid --mot-root data/soccertrack/mot \
  --output data/treinamento --interval 36 --one-based 118578
```

O notebook `notebooks/01_train.ipynb` também executa esse preparador antes de treinar; ajuste nele `MOT_ROOT`, `PREPARED`, `ONE_BASED` e, se necessário, `SPLITS`. No Colab, a branch remota precisa conter os scripts antes da clonagem.

Escolha `--yolo`, `--reid` ou ambos. A saída contém, conforme a seleção, `yolo/dataset.yaml` e/ou `reid/metadata.csv`. Use `--one-based 118578` apenas com a anotação original dessa partida; a cópia local documentada já foi normalizada para base zero. Se a execução for interrompida, escolha outro diretório vazio para recomeçar. Esse comando prepara dados: a YOLO é treinada com `imgsz=1280` e a rede ReID deve ser treinada separadamente com uma implementação compatível com os recortes.

No Colab, o notebook de treinamento prepara YOLO em `/content/dataset_yolo_reid` (temporário), usa os grupos MOT padrão e salva pesos e métricas no Drive. Após treinar, processa o clipe de teste `132877` com a calibração em `configs/cameras/132877.yaml`, salvando o vídeo na pasta da execução. Os dados temporários são perdidos quando a sessão termina. A calibração é específica desse vídeo.
