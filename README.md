# Exercícios de Python e Data Science — Turma B Noturno

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Google Colab](https://img.shields.io/badge/Google%20Colab-compatível-yellow)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-green)
![Versão](https://img.shields.io/badge/versão-v6.0.0-lightgrey)

Repositório com exercícios e projetos desenvolvidos em Python ao longo do curso, cobrindo desde os fundamentos de programação até Data Science e Machine Learning aplicado (classificação e regressão). Cada atividade traz código, explicação da lógica e saída de execução registrada em notebooks.

---

## Sumário

- [Conteúdo](#conteúdo)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Arquivos de dados](#arquivos-de-dados)
- [Como executar](#como-executar)
- [Bibliotecas utilizadas](#bibliotecas-utilizadas)
- [Requisitos](#requisitos)
- [Histórico de versões](#histórico-de-versões)
- [Autor](#autor)

---

## Conteúdo

### 1. Lógica de Programação — 50 exercícios

Prática dos principais fundamentos da linguagem Python:

- Variáveis e tipos de dados
- Entrada e saída de dados
- Estruturas condicionais
- Laços de repetição (`for` e `while`)
- Funções
- Listas e dicionários
- Manipulação de strings
- Recursão
- Algoritmos e estruturas de dados

Cada exercício apresenta o enunciado, o código da solução, a explicação da lógica utilizada e a saída de um teste executado.

### 2. Data Science — 80 exercícios

Exercícios práticos com as principais bibliotecas de Data Science, sendo 20 para cada uma:

| Biblioteca | Foco |
|---|---|
| **NumPy** | Criação e manipulação de arrays, matrizes e operações numéricas |
| **SciPy** | Computação científica, otimização, integração e álgebra linear |
| **Pandas** | Manipulação e análise de dados tabulares |
| **Matplotlib** | Criação e personalização de gráficos |

### 3. Machine Learning — Classificação de espécies de Iris

Projeto de classificação supervisionada no dataset Iris (150 amostras, 4 atributos, 3 classes balanceadas com 50 amostras cada). Três algoritmos foram implementados em notebooks separados, e cada um compara duas abordagens de definição de hiperparâmetros: **com** e **sem** `GridSearchCV`.

**Metodologia comum aos três notebooks**

- Divisão treino/teste de 80/20, estratificada, com `random_state=42` (120 amostras de treino e 30 de teste)
- Validação cruzada com 5 folds e métrica de acurácia
- Avaliação com acurácia, `classification_report` e matriz de confusão
- Análise de sensibilidade: curva de validação de um hiperparâmetro-chave e importância dos atributos

#### 3.1 SVM

**Arquivo:** `Iris_Species_com_SVM.ipynb`

`Pipeline` com `StandardScaler` + `SVC`, evitando vazamento de dados. Espaço de busca: `C ∈ {0,1; 1; 10; 100}`, `kernel ∈ {linear, rbf, poly}`, `gamma ∈ {scale, auto, 0,1; 1}`. Sensibilidade: curva de validação para `C` e importância por permutação.

| Configuração | Hiperparâmetros | Acurácia CV | Acurácia treino | Acurácia teste |
|---|---|:---:|:---:|:---:|
| Sem `GridSearchCV` | kernel `rbf`, `C=1`, `gamma='scale'` | 96,67% | 97,50% | **96,67%** |
| Com `GridSearchCV` | kernel `linear`, `C=0,1`, `gamma='scale'` | **97,50%** | 97,50% | 93,33% |

O kernel linear escolhido sugere que as classes do Iris são praticamente separáveis por fronteiras lineares.

#### 3.2 Boosting (Gradient Boosting)

**Arquivo:** `Iris_Species_com_Boosting.ipynb`

`GradientBoostingClassifier`. Espaço de busca: `n_estimators ∈ {50, 100, 200}`, `learning_rate ∈ {0,01; 0,1; 0,2; 0,5}`, `max_depth ∈ {1, 2, 3}`, `subsample ∈ {0,8; 1,0}`. Sensibilidade: curva de validação para `learning_rate`, importância das variáveis e importância por permutação.

| Configuração | Hiperparâmetros | Acurácia CV | Acurácia teste |
|---|---|:---:|:---:|
| Sem `GridSearchCV` | 100 árvores, `learning_rate=0,1`, `max_depth=3`, `subsample=1,0` | 96,67% | **96,67%** |
| Com `GridSearchCV` | idênticos aos do modelo sem busca | 96,67% | **96,67%** |

A busca em grade escolheu exatamente os hiperparâmetros usados no modelo sem busca, então os resultados são idênticos. O modelo atingiu 100% de acurácia no treino, sem prejuízo visível no teste. O único erro foi uma flor *Iris-versicolor* classificada como *Iris-virginica*.

#### 3.3 Árvore de Decisão

**Arquivo:** `Iris_Species_com_Arvore_de_Decisão.ipynb`

`DecisionTreeClassifier`. Espaço de busca: `criterion ∈ {gini, entropy}`, `max_depth ∈ {2, 3, 4, 5, 6, None}`, `min_samples_split ∈ {2, 5, 10}`, `min_samples_leaf ∈ {1, 2, 4}`. Inclui a visualização da árvore (`plot_tree`), curva de validação para `max_depth` e importância das variáveis.

| Configuração | Hiperparâmetros | Acurácia CV | Acurácia teste |
|---|---|:---:|:---:|
| Sem `GridSearchCV` | `gini`, `max_depth=3` (definido manualmente) | 93,33% | **96,67%** |
| Com `GridSearchCV` | `gini`, `max_depth=4` | **94,17%** | 93,33% |

A busca melhorou a validação cruzada em menos de um ponto percentual e piorou o teste em uma amostra, sem ganho real. Em ambas as configurações foi o modelo com menor acurácia em validação cruzada, comportamento esperado de uma árvore única, mais instável que métodos de conjunto.

#### 3.4 Comparativo dos modelos

| Modelo | Configuração | Acurácia CV | Acurácia teste |
|---|---|:---:|:---:|
| SVM | Sem `GridSearchCV` | 96,67% | 96,67% |
| SVM | Com `GridSearchCV` | 97,50% | 93,33% |
| Boosting | Sem `GridSearchCV` | 96,67% | 96,67% |
| Boosting | Com `GridSearchCV` | 96,67% | 96,67% |
| Árvore de Decisão | Sem `GridSearchCV` | 93,33% | 96,67% |
| Árvore de Decisão | Com `GridSearchCV` | 94,17% | 93,33% |

- A *Iris-setosa* foi classificada sem erros por todos os modelos; as confusões ocorrem entre *Iris-versicolor* e *Iris-virginica*.
- Boosting e SVM (sem `GridSearchCV`) tiveram os melhores resultados no teste, ambos com 96,67%.
- O `GridSearchCV` não trouxe ganho consistente neste dataset: melhorou a validação cruzada no SVM e na Árvore, mas o teste caiu de 96,67% para 93,33% nos dois casos.
- **Ressalva:** o conjunto de teste tem apenas 30 amostras, então cada diferença entre modelos equivale a 1 ou 2 observações. Os resultados não bastam para afirmar superioridade estatística de um modelo sobre outro.

### 4. Machine Learning — Regressão: previsão de preços de imóveis com SVR

Projeto de regressão supervisionada com **Support Vector Regression (SVR)** sobre o dataset *California Housing* (20.640 regiões, 10 atributos), com o objetivo de prever o valor mediano dos imóveis (`median_house_value`) atingindo **R² > 0,65** no conjunto de teste.

**Arquivos:** `Fernando_Panichek_Quinsan_SVM.ipynb` (notebook) e [`relatorios/fernando_panichek_quinsan_relatorio.pdf`](relatorios/fernando_panichek_quinsan_relatorio.pdf) (relatório completo, 8 páginas).

**Etapas**

1. Análise exploratória: distribuição do alvo, boxplots, matriz de correlação (`median_income` é a variável mais correlacionada com o preço, 0,69).
2. Dados ausentes: 207 valores em `total_bedrooms` (1,0%), preenchidos pela mediana dentro do `Pipeline`, aprendida só no treino. Sem registros duplicados.
3. Outliers: detecção pelo método IQR; foram **mantidos** no modelo final por não haver evidência de erro de cadastro.
4. Codificação: `ocean_proximity` com One-Hot Encoding, resultando em 13 atributos.
5. Divisão 80/20 com `random_state=42` (16.512 registros de treino e 4.128 de teste).
6. `Pipeline` com `SimpleImputer` + `StandardScaler` + `SVR`, com o preço dividido por 100.000 para treino.
7. Comparação de kernels e `GridSearchCV` (5 folds, `scoring='r2'`, 11 configurações).

**Impacto de cada tratamento** (mesmas 3.921 linhas completas do teste):

| Modelo | Tratamento | R² |
|---|---|:---:|
| Baseline | SVR padrão, só linhas completas | -0,0485 |
| 2 | Preenchimento pela mediana | -0,0485 |
| 3 | Mediana + limites de outliers | -0,0480 |
| 4 | Mediana + padronização + preço em 100 mil | 0,7483 |
| Final | Dados mantidos + escalas + `GridSearchCV` | **0,7759** |

**Comparação de kernels** (`C=1`, `epsilon=0,1`): linear 0,6108 · **rbf 0,7437** · poly 0,6527.

**Modelo final:** kernel `rbf`, `C=10`, `epsilon=0,1`, `gamma=0,1`.

| Métrica | Valor (4.128 registros de teste) |
|---|---|
| R² no teste | **0,7711** |
| R² médio na validação cruzada | 0,7806 |
| MAE | US$ 35.931,63 |
| RMSE | US$ 54.764,04 |

**Principais conclusões**

- A meta de R² > 0,65 foi atingida (0,7711).
- Quase todo o ganho vem da **padronização das variáveis e do preço**: o SVR sem escala teve R² negativo, pior que prever a média.
- Tratar ausentes e outliers isoladamente teve efeito desprezível; a otimização de hiperparâmetros acrescentou cerca de 0,03 de R².
- O modelo subestima os imóveis de valor máximo (o alvo é truncado em US$ 500.001): um valor real de US$ 500.001 foi previsto em ≈ US$ 323 mil.
- Melhorias sugeridas: criar variáveis como quartos por domicílio, testar novas faixas de `C` e `gamma` (o melhor `C=10` está na borda da grade testada) e reduzir a redundância entre `total_rooms`, `total_bedrooms`, `population` e `households`.

---

## Estrutura do repositório

```
.
├── README.md
├── RELEASES.md
├── iris.data                                  # Dataset Iris (exercícios de Pandas)
├── Iris.csv                                   # Dataset Iris em CSV (projeto de classificação)
├── housing.csv                                # Dataset California Housing (projeto de regressão)
├── <notebook-logica>.ipynb                    # 50 exercícios de Lógica de Programação
├── <notebook-data-science>.ipynb              # 80 exercícios de Data Science
├── Iris_Species_com_SVM.ipynb                 # Classificação: SVM
├── Iris_Species_com_Boosting.ipynb            # Classificação: Boosting
├── Iris_Species_com_Arvore_de_Decisão.ipynb   # Classificação: Árvore de Decisão
├── Fernando_Panichek_Quinsan_SVM.ipynb        # Regressão: SVR (Housing)
└── relatorios/
    └── fernando_panichek_quinsan_relatorio.pdf
```

> Ajuste os nomes dos notebooks de lógica e de Data Science conforme os arquivos reais do repositório.

---

## Arquivos de dados

- `iris.data`: utilizado nos exercícios de Pandas que leem arquivos externos.
- `Iris.csv`: utilizado nos notebooks de classificação. Fonte: [Iris Species no Kaggle](https://www.kaggle.com/datasets/menegidio/iris-species). Os notebooks removem a coluna `Id` após a leitura.
- `housing.csv`: utilizado no projeto de regressão. Fonte: [Housing no Kaggle](https://www.kaggle.com/datasets/kathuman/housing).

Ao executar no Google Colab, faça o upload do arquivo solicitado antes de rodar a célula correspondente. Os notebooks de Machine Learning foram executados no Kaggle e leem os dados de caminhos do Kaggle; ao rodar em outro ambiente, ajuste o `pd.read_csv`:

| Notebooks | Caminho original |
|---|---|
| Iris (SVM, Boosting, Árvore) | `/kaggle/input/datasets/menegidio/iris-species/Iris.csv` |
| Housing (SVR) | `/kaggle/input/datasets/kathuman/housing/housing.csv` |

---

## Como executar

Os notebooks foram desenvolvidos para o Google Colab e Kaggle, e também funcionam em qualquer ambiente com Jupyter Notebook.

1. Abra o arquivo `.ipynb` desejado.
2. Execute as células em ordem ou use **Ambiente de execução → Executar tudo**.
3. Em exercícios que usam `input()`, informe o valor solicitado quando a execução pausar.
4. Em exercícios e projetos com arquivos externos, faça o upload do arquivo correspondente (e ajuste o caminho, se necessário) antes da execução.

---

## Bibliotecas utilizadas

| Biblioteca | Uso |
|---|---|
| `math`, `random`, `string` | Cálculos matemáticos, geração aleatória e manipulação de caracteres |
| NumPy | Computação numérica e arrays |
| SciPy | Computação científica |
| Pandas | Manipulação e análise de dados |
| Matplotlib | Visualização de dados |
| Seaborn | Mapa de calor da matriz de correlação |
| scikit-learn | Pipelines, SVM/SVR, Gradient Boosting, Árvore de Decisão, `GridSearchCV`, validação cruzada e métricas |

---

## Requisitos

- Python 3.x
- Google Colab, Kaggle ou Jupyter Notebook

As bibliotecas já vêm instaladas por padrão no Google Colab e no Kaggle. Em ambiente local:

```bash
pip install numpy scipy pandas matplotlib seaborn scikit-learn
```

---

## Histórico de versões

| Versão | Entrega | Conteúdo |
|---|---|---|
| **v6.0.0** | 29/10/2026 | Regressão: previsão de preços de imóveis (Housing) com SVR + relatório em PDF |
| **v5.0.0** | 29/10/2026 | Classificação: Iris Species com Árvore de Decisão (com e sem `GridSearchCV`) |
| **v4.0.0** | 29/10/2026 | Classificação: Iris Species com Boosting (com e sem `GridSearchCV`) |
| **v3.0.0** | 29/10/2026 | Classificação: Iris Species com SVM (com e sem `GridSearchCV`) |
| **v2.0.0** | 21/08/2026 | 80 exercícios de Data Science (NumPy, SciPy, Pandas e Matplotlib) |
| **v1.0.0** | 16/08/2026 | 50 exercícios de Lógica de Programação em Python |

Detalhes completos em [RELEASES.md](RELEASES.md).

---

## Autor

**Fernando Panichek Quinsan**
Turma B — Noturno
