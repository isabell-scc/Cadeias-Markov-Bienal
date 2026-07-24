# -*- coding: utf-8 -*-
"""

Original file is located at
    https://colab.research.google.com/drive/135gEtoSTjwktmf7Sja4gPR6Nzyq-_ox6

VERSÃO FINAL: extração de corpus correta (sem o bug do regex guloso
"copyright.*" com DOTALL) + fontes e figuras ampliadas para pôster A0,
com legendas fora da área das barras (sem sobreposição).
"""

# ============================================================
# PROJETO
# Geração Automática de Textos usando Cadeias de Markov
# ============================================================

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import random
import re
import math
import textwrap

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURAÇÕES
# ============================================================

random.seed(42)
np.random.seed(42)

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300

plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = .3

# Fonte base ampliada (era 11) — pensada para leitura confortável
# num pôster A0 impresso, mesmo dividido em colunas estreitas.
plt.rcParams["font.size"] = 16

# Tamanhos específicos por elemento, usados explicitamente em cada
# gráfico (em vez de depender só do rcParams global).
FS_TITULO = 22
FS_EIXO = 18
FS_TICK = 16
FS_LEGENDA = 16
FS_ROTULO_BARRA = 14

# ============================================================
# DIRETÓRIOS
# ============================================================

BASE = Path(".")

CORPUS = BASE / "corpus"

OUT = BASE / "resultados"

OUT.mkdir(exist_ok=True)

# ============================================================
# ARQUIVOS
# ============================================================

ARQUIVOS = {
    "Corpus Musical": CORPUS / "djavan.txt",
    "Corpus Literário": CORPUS / "machado.txt"
}

# ============================================================
# CORES
# ============================================================

CORES = {
    "Corpus Musical": "#4E79A7",
    "Corpus Literário": "#E15759"
}

# ============================================================
# ESTRUTURAS PRINCIPAIS
# ============================================================

dados = {}

metricas = {}

# ============================================================
# PRÉ-PROCESSAMENTO
# ============================================================

STOPWORDS = {
    "o","a","os","as",
    "de","do","da","dos","das",
    "e","em","no","na","nos","nas",
    "que","um","uma","por","com",
    "para","pra","não"
}

# Marcadores de início/fim de cada obra no formato Project Gutenberg.
# Aplicados ANTES do lowercase (por isso são case-insensitive aqui,
# em vez de depender da caixa do texto original).
MARCADOR_INICIO_GUTENBERG = re.compile(
    r"\*\*\*\s*START OF.*?\*\*\*", re.IGNORECASE | re.DOTALL
)
MARCADOR_FIM_GUTENBERG = re.compile(
    r"\*\*\*\s*END OF.*?\*\*\*", re.IGNORECASE | re.DOTALL
)

# Padrões aplicados DEPOIS do lowercase. Todos restritos à própria
# linha ([^\n]*, sem DOTALL) para nunca apagar o resto do arquivo
# por engano (bug anterior: "copyright.*" com DOTALL apagava tudo
# a partir da primeira ocorrência da palavra, até o fim do texto).
PADROES_REMOVER = [
    r"all rights reserved[^\n]*",
    r"copyright[^\n]*",
    r"produced by[^\n]*",
    r"transcriber'?s? note[^\n]*",
    # metadados de sites de letras de música (várias línguas)
    r"\(thanks to [^)]*\)",
    r"\(grazie a [^)]*\)",
    r"\(gracias a [^)]*\)",
    r"\(merci [^)]*\)",
    r"due to copyright restrictions[^\n]*",
    r"lyrics/music:[^\n]*",
    r"\(\|\)",
]


# Neste corpus específico, as obras concatenadas usam um separador
# "===== Título =====" entre uma obra e a próxima (em vez do par
# clássico *** START OF *** / *** END OF *** do Project Gutenberg).
PADRAO_SEPARADOR_OBRA = re.compile(r"\n=+[^\n=]*=+\n")


def extrair_conteudo_gutenberg(texto):
    """
    Remove cabeçalhos/rodapés jurídicos do Project Gutenberg,
    suportando dois formatos:

    (a) Formato clássico: cada obra vem entre um marcador
        '*** START OF ... ***' e um '*** END OF ... ***'. Extrai-se
        apenas o conteúdo entre os dois.

    (b) Formato deste corpus: não há marcador de START, mas as obras
        são separadas por linhas '===== Título =====', e o rodapé
        jurídico (texto de licença, sempre truncado no meio) aparece
        logo após cada '*** END OF ... ***' e antes do próximo
        separador. Nesse caso, cada segmento entre separadores é
        cortado no ponto em que o marcador END começa, descartando
        o rodapé.

    Sem essa extração, remover o rodapé só com regex genérico (ex.:
    procurando a palavra "copyright") é frágil: um padrão guloso pode
    apagar texto real das obras seguintes — foi exatamente esse o bug
    que apagava 67% do corpus literário antes desta correção.
    """

    inicios = [m.end() for m in MARCADOR_INICIO_GUTENBERG.finditer(texto)]
    fins_globais = [m.start() for m in MARCADOR_FIM_GUTENBERG.finditer(texto)]

    if inicios and fins_globais:
        partes = []

        for inicio in inicios:
            candidatos_fim = [f for f in fins_globais if f > inicio]

            if not candidatos_fim:
                continue

            partes.append(texto[inicio:min(candidatos_fim)])

        if partes:
            return "\n".join(partes)

    if fins_globais:
        segmentos = PADRAO_SEPARADOR_OBRA.split(texto)

        limpos = []

        for segmento in segmentos:
            m = MARCADOR_FIM_GUTENBERG.search(segmento)
            limpos.append(segmento[:m.start()] if m else segmento)

        return "\n".join(limpos)

    return texto  # sem marcadores Gutenberg conhecidos: devolve como está


def preprocessar(arquivo):
    """
    Lê um corpus textual e devolve a lista de tokens.

    Etapas:
    --------
    1) leitura
    2) extração do conteúdo real (remove cabeçalhos/rodapés Gutenberg,
       suportando múltiplas obras concatenadas no mesmo arquivo)
    3) lowercase
    4) remoção de metadados residuais (créditos, "due to copyright...")
    5) remoção de pontuação
    6) tokenização
    7) remoção de stopwords
    """

    texto = Path(arquivo).read_text(
        encoding="utf8",
        errors="ignore"
    )

    texto = extrair_conteudo_gutenberg(texto)

    texto = texto.lower()

    for padrao in PADROES_REMOVER:
        texto = re.sub(
            padrao,
            " ",
            texto,
            flags=re.IGNORECASE
        )

    texto = re.sub(r"[^a-zà-ÿ\s]", " ", texto)

    texto = re.sub(r"\s+", " ", texto)

    palavras = texto.split()

    palavras = [
        p
        for p in palavras
        if p not in STOPWORDS
    ]

    return palavras


# ============================================================
# MODELOS DE MARKOV
# ============================================================

def construir_modelo_ordem1(tokens):

    modelo = defaultdict(Counter)

    for a, b in zip(tokens[:-1], tokens[1:]):
        modelo[a][b] += 1

    return modelo


def construir_modelo_ordem2(tokens):

    modelo = defaultdict(Counter)

    for a, b, c in zip(
        tokens[:-2],
        tokens[1:-1],
        tokens[2:]
    ):
        modelo[(a,b)][c] += 1

    return modelo

def construir_modelo_ordem3(tokens):

    modelo = defaultdict(Counter)

    for a, b, c, d in zip(
        tokens[:-3],
        tokens[1:-2],
        tokens[2:-1],
        tokens[3:]
    ):
        modelo[(a,b,c)][d] += 1

    return modelo

# ============================================================
# NORMALIZAÇÃO
# ============================================================

def normalizar_modelo(modelo):

    probabilidades = {}

    for estado, sucessores in modelo.items():

        total = sum(sucessores.values())

        probabilidades[estado] = {
            palavra: freq/total
            for palavra, freq in sucessores.items()
        }

    return probabilidades

# ============================================================
# CARREGAMENTO DOS CORPORA
# ============================================================

for nome, caminho in ARQUIVOS.items():

    tokens = preprocessar(caminho)

    modelo1 = construir_modelo_ordem1(tokens)

    modelo2 = construir_modelo_ordem2(tokens)

    modelo3 = construir_modelo_ordem3(tokens)

    dados[nome] = {

        "tokens": tokens,

        "modelo1": modelo1,

        "modelo2": modelo2,

        "modelo3": modelo3,

        "prob1": normalizar_modelo(modelo1),

        "prob2": normalizar_modelo(modelo2),



    }

print("Corpus carregados.")

# ============================================================
# ESTATÍSTICAS BÁSICAS
# ============================================================

for nome in dados:

    tokens = dados[nome]["tokens"]

    modelo1 = dados[nome]["modelo1"]

    modelo2 = dados[nome]["modelo2"]

    metricas[nome] = {

        "tokens": len(tokens),

        "vocab": len(set(tokens)),

        "n_estados_o1": len(modelo1),

        "n_estados_o2": len(modelo2)

    }

metricas

# ============================================================
# DISTRIBUIÇÃO ESTACIONÁRIA
# ============================================================

from scipy import sparse

def distribuicao_estacionaria(modelo, ordem=1, tol=1e-10, max_iter=500):
    """
    Calcula a distribuição estacionária de uma Cadeia de Markov via
    método das potências (pi @ P = pi), usando matriz ESPARSA.

    Por que esparsa: cada estado tem, em média, poucos sucessores
    possíveis (dezenas, no máximo). Guardar a matriz de transição
    como densa (n x n) inviabiliza corpora grandes — ex.: o corpus
    literário em Ordem 2 tem ~314.773 estados, o que exigiria uma
    matriz densa de ~790 GB de RAM. Com matriz esparsa, só as
    transições que realmente existem ocupam memória.

    IMPORTANTE (Ordem 2): o sucessor armazenado no modelo é apenas
    a próxima palavra isolada — o estado seguinte precisa ser
    reconstruído como (palavra_atual, próxima_palavra) antes de
    procurá-lo no índice de estados.
    """

    estados = list(modelo.keys())

    indice = {estado: i for i, estado in enumerate(estados)}

    n = len(estados)

    linhas_idx, colunas_idx, valores = [], [], []

    dangling = []

    for estado, sucessores in modelo.items():

        total = sum(sucessores.values())

        if total == 0:
            continue

        i = indice[estado]

        n_antes = len(valores)

        for prox, freq in sucessores.items():

            if ordem == 1:
                prox_estado = prox
            else:
                # estado = (w_{t-1}, w_t) -> próximo estado = (w_t, w_{t+1})
                prox_estado = (estado[1], prox)

            if prox_estado in indice:

                j = indice[prox_estado]

                linhas_idx.append(i)
                colunas_idx.append(j)
                valores.append(freq / total)

        if len(valores) == n_antes:
            dangling.append(i)

    P = sparse.csr_matrix(
        (valores, (linhas_idx, colunas_idx)),
        shape=(n, n)
    )

    # estados absorventes/sem sucessor válido: distribuem uniformemente
    if dangling:
        peso = 1.0 / n
        for i in dangling:
            linhas_idx.extend([i] * n)
            colunas_idx.extend(range(n))
            valores.extend([peso] * n)
        P = sparse.csr_matrix(
            (valores, (linhas_idx, colunas_idx)),
            shape=(n, n)
        )

    pi = np.ones(n) / n

    for _ in range(max_iter):

        novo = pi @ P

        novo = np.asarray(novo).ravel()

        if np.linalg.norm(novo - pi, 1) < tol:
            pi = novo
            break

        pi = novo

    return {

        estado: pi[indice[estado]]

        for estado in estados

    }

# ============================================================
# CÁLCULO EFETIVO DA DISTRIBUIÇÃO ESTACIONÁRIA
# (antes definida, mas nunca utilizada no pipeline)
# ============================================================

for nome in dados:

    dados[nome]["pi1"] = distribuicao_estacionaria(
        dados[nome]["modelo1"], ordem=1
    )

    dados[nome]["pi2"] = distribuicao_estacionaria(
        dados[nome]["modelo2"], ordem=2
    )

    # top-5 estados mais prováveis no regime estacionário (útil p/ discussão no banner)
    top1 = sorted(dados[nome]["pi1"].items(), key=lambda x: -x[1])[:5]
    top2 = sorted(dados[nome]["pi2"].items(), key=lambda x: -x[1])[:5]

    print(f"[{nome}] Top-5 estados estacionários (Ordem 1): {top1}")
    print(f"[{nome}] Top-5 estados estacionários (Ordem 2): {top2}")

# ============================================================
# GERAÇÃO DE TEXTO
# ============================================================

def escolher(probs):

    palavras = list(probs.keys())

    pesos = list(probs.values())

    return random.choices(
        palavras,
        weights=pesos,
        k=1
    )[0]


def gerar_texto_ordem1(modelo, tamanho=80):

    estado = random.choice(list(modelo.keys()))

    texto = [estado]

    for _ in range(tamanho - 1):

        if estado not in modelo:

            break

        estado = escolher(modelo[estado])

        texto.append(estado)

    return " ".join(texto)


def gerar_texto_ordem2(modelo, tamanho=80):

    estado = random.choice(list(modelo.keys()))

    texto = [estado[0], estado[1]]

    atual = estado

    for _ in range(tamanho - 2):

        if atual not in modelo:

            break

        prox = escolher(modelo[atual])

        texto.append(prox)

        atual = (atual[1], prox)

    return " ".join(texto)

# ============================================================
# EXEMPLOS
# ============================================================

for nome in dados:

    print("="*80)

    print(nome)

    print("\nORDEM 1\n")

    print(
        gerar_texto_ordem1(
            dados[nome]["prob1"]
        )
    )

    print("\nORDEM 2\n")

    print(
        gerar_texto_ordem2(
            dados[nome]["prob2"]
        )
    )

    print()

# ============================================================
# ENTROPIA
# ============================================================

def entropia(modelo):

    valores = []

    for sucessores in modelo.values():

        probs = np.array(
            list(sucessores.values())
        )

        probs = probs / probs.sum()

        h = -(probs * np.log2(probs)).sum()

        valores.append(h)

    return np.mean(valores)

for nome in dados:

    metricas[nome]["entropia_o1"] = entropia(
        dados[nome]["modelo1"]
    )

    metricas[nome]["entropia_o2"] = entropia(
        dados[nome]["modelo2"]
    )

# ============================================================
# DETERMINISMO
# ============================================================

def percentual_deterministico(modelo):

    det = 0

    total = len(modelo)

    for sucessores in modelo.values():

        if len(sucessores) == 1:

            det += 1

    return 100 * det / total

for nome in dados:

    metricas[nome]["pct_determ_o1"] = percentual_deterministico(
        dados[nome]["modelo1"]
    )

    metricas[nome]["pct_determ_o2"] = percentual_deterministico(
        dados[nome]["modelo2"]
    )

# ============================================================
# GRÁFICOS
# ============================================================

nomes = list(dados.keys())

labels = [n.replace("Corpus ", "") for n in nomes]

cores = [CORES[n] for n in nomes]

x = np.arange(len(labels))

w = .35

# ============================================================
# ESPAÇO DE ESTADOS (figura e fontes ampliadas)
# ============================================================

fig, ax = plt.subplots(figsize=(10, 7))

b1 = ax.bar(
    x-w/2,
    [metricas[n]["n_estados_o1"] for n in nomes],
    width=w,
    label="Ordem 1"
)

b2 = ax.bar(
    x+w/2,
    [metricas[n]["n_estados_o2"] for n in nomes],
    width=w,
    label="Ordem 2"
)

ax.set_xticks(x)

ax.set_xticklabels(labels, fontsize=FS_TICK)

ax.tick_params(axis="y", labelsize=FS_TICK)

ax.set_ylabel("Número de estados", fontsize=FS_EIXO)

ax.set_title("Complexidade do espaço de estados", fontsize=FS_TITULO, fontweight="bold", pad=18)

ax.set_yscale("log")

maior_n_estados = max(
    [metricas[n]["n_estados_o1"] for n in nomes]
    + [metricas[n]["n_estados_o2"] for n in nomes]
)
ax.set_ylim(top=maior_n_estados * 2.2)

ax.legend(fontsize=FS_LEGENDA, loc="upper center",
          bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)

for bars in (b1,b2):

    for b in bars:

        h=b.get_height()

        ax.text(
            b.get_x()+b.get_width()/2,
            h*1.08,
            f"{int(h):,}",
            ha="center",
            fontsize=FS_ROTULO_BARRA
        )

plt.tight_layout()

plt.savefig(
    OUT/"espaco_estados.png"
)

plt.close()


# ============================================================
# GRAU DE SAÍDA (figura e fontes ampliadas)
# ============================================================

for nome in nomes:

    fig, ax = plt.subplots(figsize=(10, 7))

    for ordem, marcador in [

        ("modelo1","o"),

        ("modelo2","s")

    ]:

        graus = [

            len(v)

            for v in dados[nome][ordem].values()

        ]

        cont = Counter(graus)

        xs = sorted(cont)

        ys = [cont[g] for g in xs]

        ax.plot(

            xs,

            ys,

            marker=marcador,

            linewidth=2.5,

            markersize=8,

            label=ordem.replace("modelo","Ordem ")

        )

    ax.set_yscale("log")

    ax.set_xlabel("Número de sucessores", fontsize=FS_EIXO)

    ax.set_ylabel("Quantidade de estados", fontsize=FS_EIXO)

    ax.tick_params(axis="both", labelsize=FS_TICK)

    ax.set_title(nome, fontsize=FS_TITULO, fontweight="bold", pad=18)

    ax.legend(fontsize=FS_LEGENDA, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)

    plt.tight_layout()

    plt.savefig(

        OUT/f"grau_saida_{nome.split()[-1].lower()}.png"

    )

    plt.close()

# ============================================================
# PAINEL DE RESULTADOS 1: Perplexidade + Estados determinísticos
# (fontes ampliadas, legenda fora das barras, sem sobreposição)
# ============================================================

# Calculate perplexity (Perplexity = 2^Entropy)
for nome in dados:
    metricas[nome]["perplexidade_o1"] = 2**metricas[nome]["entropia_o1"]
    metricas[nome]["perplexidade_o2"] = 2**metricas[nome]["entropia_o2"]

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

b1 = axs[0].bar(
    x-w/2,
    [metricas[n]["perplexidade_o1"] for n in nomes],
    width=w,
    label="Ordem 1"
)

b2 = axs[0].bar(
    x+w/2,
    [metricas[n]["perplexidade_o2"] for n in nomes],
    width=w,
    label="Ordem 2"
)

axs[0].set_title("Perplexidade", fontsize=FS_TITULO, fontweight="bold", pad=18)

axs[0].set_xticks(x)

axs[0].set_xticklabels(labels, fontsize=FS_TICK)

axs[0].tick_params(axis="y", labelsize=FS_TICK)

axs[0].set_yscale("log")

maior_perplexidade = max(
    [metricas[n]["perplexidade_o1"] for n in nomes]
    + [metricas[n]["perplexidade_o2"] for n in nomes]
)
axs[0].set_ylim(top=maior_perplexidade * 1.35)

axs[0].legend(fontsize=FS_LEGENDA, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)

# Determinismo: barras agrupadas Ordem 1 vs Ordem 2 (igual ao restante
# do pôster), em vez de uma única barra por corpus.
b1d = axs[1].bar(
    x-w/2,
    [metricas[n]["pct_determ_o1"] for n in nomes],
    width=w,
    label="Ordem 1"
)

b2d = axs[1].bar(
    x+w/2,
    [metricas[n]["pct_determ_o2"] for n in nomes],
    width=w,
    label="Ordem 2"
)

axs[1].set_ylim(0, 112)

axs[1].set_title("Estados determinísticos (%)", fontsize=FS_TITULO, fontweight="bold", pad=18)

axs[1].set_xticks(x)

axs[1].set_xticklabels(labels, fontsize=FS_TICK)

axs[1].tick_params(axis="y", labelsize=FS_TICK)

axs[1].legend(fontsize=FS_LEGENDA, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)

for bars in (b1, b2):

    for b in bars:

        h=b.get_height()

        axs[0].text(

            b.get_x()+b.get_width()/2,

            h*1.08,

            f"{h:.2f}",

            ha="center",

            fontsize=FS_ROTULO_BARRA

        )

for bars in (b1d, b2d):

    for b in bars:

        h=b.get_height()

        axs[1].text(

            b.get_x()+b.get_width()/2,

            h+2,

            f"{h:.1f}",

            ha="center",

            fontsize=FS_ROTULO_BARRA

        )

for ax in axs.ravel():

    ax.grid(alpha=.3)

plt.tight_layout()

plt.savefig(

    OUT/"painel_resultados_1.png",

    dpi=300

)

plt.close()

# ------------------------------------------------------------
# PAINEL DE RESULTADOS 2: Vocabulário + Número de estados
# (fontes ampliadas, legenda fora das barras, sem sobreposição)
# ------------------------------------------------------------

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

bars = axs[0].bar(

    labels,

    [metricas[n]["vocab"] for n in nomes],

    color=cores

)

axs[0].set_title("Vocabulário", fontsize=FS_TITULO, fontweight="bold", pad=18)

axs[0].set_ylabel("Palavras distintas", fontsize=FS_EIXO)

axs[0].tick_params(axis="both", labelsize=FS_TICK)

axs[0].set_ylim(top=max(metricas[n]["vocab"] for n in nomes) * 1.15)

for b in bars:

    h=b.get_height()

    axs[0].text(

        b.get_x()+b.get_width()/2,

        h*1.02,

        f"{int(h):,}",

        ha="center",

        fontsize=FS_ROTULO_BARRA

    )

b1 = axs[1].bar(

    x-w/2,

    [metricas[n]["n_estados_o1"] for n in nomes],

    width=w,

    label="Ordem 1"

)

b2 = axs[1].bar(

    x+w/2,

    [metricas[n]["n_estados_o2"] for n in nomes],

    width=w,

    label="Ordem 2"

)

axs[1].set_title("Número de estados", fontsize=FS_TITULO, fontweight="bold", pad=18)

axs[1].set_xticks(x)

axs[1].set_xticklabels(labels, fontsize=FS_TICK)

axs[1].tick_params(axis="y", labelsize=FS_TICK)

axs[1].set_yscale("log")

maior_n_estados_2 = max(
    [metricas[n]["n_estados_o1"] for n in nomes]
    + [metricas[n]["n_estados_o2"] for n in nomes]
)
axs[1].set_ylim(top=maior_n_estados_2 * 2.2)

axs[1].legend(fontsize=FS_LEGENDA, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)

for bars in (b1,b2):

    for b in bars:

        h=b.get_height()

        axs[1].text(

            b.get_x()+b.get_width()/2,

            h*1.08,

            f"{int(h):,}",

            ha="center",

            fontsize=FS_ROTULO_BARRA

        )

for ax in axs.ravel():

    ax.grid(alpha=.3)

plt.tight_layout()

plt.savefig(

    OUT/"painel_resultados_2.png",

    dpi=300

)

plt.close()

# ============================================================
# FIGURA - EXEMPLOS DE TEXTOS GERADOS (caixas e fonte ampliadas)
# ============================================================

# Reprodutibilidade (reseed local apenas para esta figura específica,
# garantindo que ela seja sempre igual independente de chamadas anteriores)
random.seed(42)

def quebra(texto, largura=24):
    """Quebra o texto em linhas para melhor visualização."""
    return textwrap.fill(texto, width=largura)

# ============================================================
# GERAÇÃO DOS TEXTOS (uma única vez)
# ============================================================

# Lista de stopwords mais enxuta, usada SÓ nesta figura ilustrativa
# (não nas métricas/painéis, para não mexer nos números já validados).
# Mantém preposições/contrações como "no", "na", "dos", "das", "nos",
# "nas": remover essas palavras deixava o texto gerado mais "picado";
# mantê-las dá mais fluidez às frases geradas.
STOPWORDS_GERACAO = {
    "o","a","os","as",
    "de","do","da",
    "e","que","em","um","uma",
    "não","pra","por","com",
}

def preprocessar_geracao(arquivo):
    """Mesmo pipeline de preprocessar(), mas com STOPWORDS_GERACAO
    (lista mais enxuta) — usado só para gerar os textos de exemplo
    do banner, não para as métricas quantitativas."""

    texto = Path(arquivo).read_text(encoding="utf8", errors="ignore")

    texto = extrair_conteudo_gutenberg(texto)

    texto = texto.lower()

    for padrao in PADROES_REMOVER:
        texto = re.sub(padrao, " ", texto, flags=re.IGNORECASE)

    texto = re.sub(r"[^a-zà-ÿ\s]", " ", texto)

    texto = re.sub(r"\s+", " ", texto)

    palavras = texto.split()

    return [p for p in palavras if p not in STOPWORDS_GERACAO]


for nome in dados:

    tokens_geracao = preprocessar_geracao(ARQUIVOS[nome])

    modelo1_geracao = normalizar_modelo(
        construir_modelo_ordem1(tokens_geracao)
    )

    modelo2_geracao = normalizar_modelo(
        construir_modelo_ordem2(tokens_geracao)
    )

    dados[nome]["texto_o1"] = gerar_texto_ordem1(
        modelo1_geracao,
        tamanho=20
    )

    dados[nome]["texto_o2"] = gerar_texto_ordem2(
        modelo2_geracao,
        tamanho=20
    )

# ============================================================
# TEXTOS
# ============================================================

musical_o1 = dados["Corpus Musical"]["texto_o1"]
musical_o2 = dados["Corpus Musical"]["texto_o2"]

literario_o1 = dados["Corpus Literário"]["texto_o1"]
literario_o2 = dados["Corpus Literário"]["texto_o2"]

# ============================================================
# ESTILO
# ============================================================

COR_O1 = "#DCEEFF"
COR_O2 = "#BFDDF5"

estilo_o1 = dict(
    boxstyle="round,pad=0.6,rounding_size=0.2",
    facecolor=COR_O1,
    edgecolor="#777777",
    linewidth=1.5
)

estilo_o2 = dict(
    boxstyle="round,pad=0.6,rounding_size=0.2",
    facecolor=COR_O2,
    edgecolor="#777777",
    linewidth=1.5
)

# ============================================================
# FIGURA (maior e com fonte maior nas caixas de texto)
# ============================================================

fig, ax = plt.subplots(figsize=(9, 11))

ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.axis("off")

# Títulos das colunas

ax.text(
    0.25,
    0.92,
    "Corpus Musical",
    ha="center",
    fontsize=22,
    fontweight="bold",
    color=CORES["Corpus Musical"]
)

ax.text(
    0.75,
    0.92,
    "Corpus Literário",
    ha="center",
    fontsize=22,
    fontweight="bold",
    color=CORES["Corpus Literário"]
)

# Ordem 1

ax.text(
    0.25,
    0.67,
    "Modelo O₁\n\n"+quebra(musical_o1),
    ha="center",
    va="center",
    fontsize=15,
    bbox=estilo_o1
)

ax.text(
    0.75,
    0.67,
    "Modelo O₁\n\n"+quebra(literario_o1),
    ha="center",
    va="center",
    fontsize=15,
    bbox=estilo_o1
)

# Ordem 2

ax.text(
    0.25,
    0.28,
    "Modelo O₂\n\n"+quebra(musical_o2),
    ha="center",
    va="center",
    fontsize=15,
    bbox=estilo_o2
)

ax.text(
    0.75,
    0.28,
    "Modelo O₂\n\n"+quebra(literario_o2),
    ha="center",
    va="center",
    fontsize=15,
    bbox=estilo_o2
)

plt.tight_layout()

plt.savefig(
    OUT / "textos_markov_coluna.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()

print("\nConcluído. Gráficos salvos em 'resultados/' com preprocessamento "
      "correto (sem o bug do regex guloso) e fonte ampliada para pôster A0.")