# Notas de Release

Todas as mudanças relevantes deste repositório estão documentadas aqui, da versão mais recente para a mais antiga.

---

## Release v7.0.0 — Classificação Multiclasse: Famílias de Peixes (FishMorph) com SVM

**Data de entrega:** 29/10/2026
**Turma B — Noturno**

### Novidades

- Novo notebook `Atividade_Entrega_FishMorphDB.ipynb`: classificação da **família** de peixes de água doce a partir de 10 medidas morfológicas, com `SVC`.
- Limpeza dos dados (1 registro duplicado removido, sem valores ausentes) e seleção das famílias com pelo menos 5 registros, resultando em **118 classes** e 8.341 registros.
- Divisão treino/teste de 80/20, estratificada e reprodutível (`random_state=42`).
- `Pipeline` com `StandardScaler` + `SVC` e `GridSearchCV` com validação cruzada de 3 folds (64 combinações de `kernel`, `C` e `gamma`, totalizando 192 treinos).
- Gráfico de distribuição das 30 famílias mais frequentes e matriz de confusão das 20 famílias mais frequentes no teste.
- Cada bloco de código acompanhado de uma explicação em texto.

### Resultados

| Item | Valor |
|---|---|
| Melhores hiperparâmetros | kernel `rbf`, `C=10`, `gamma=0,1` |
| Acurácia no teste | **66,97%** |

- O modelo tem mais acertos nas famílias *Cyprinidae*, *Cichlidae*, *Characidae* e *Loricariidae*, que também são as mais frequentes; o resultado por classe (recall) não foi calculado.
- O notebook não compara a acurácia com uma linha de base nem usa métricas robustas a desbalanceamento (como F1 macro), o que limita a interpretação do resultado.

### Arquivos de dados

- `FISHMORPH_Family_Dataset.csv`, utilizado pelo notebook (8.342 registros e 11 colunas).

### Compatibilidade

Nenhuma mudança incompatível. Os conteúdos das versões anteriores permanecem inalterados.

---

## Release v6.0.0 — Regressão: Previsão de Preços de Imóveis com SVR

**Data de entrega:** 29/10/2026
**Turma B — Noturno**

### Novidades

- Novo projeto de **regressão** com **Support Vector Regression (SVR)** sobre o dataset *California Housing* (20.640 regiões, 10 atributos), no notebook `Fernando_Panichek_Quinsan_SVM.ipynb`.
- Objetivo: prever o valor mediano dos imóveis (`median_house_value`) com **R² > 0,65** no teste.
- Análise exploratória (distribuição do alvo, boxplots e matriz de correlação), tratamento de valores ausentes (mediana), detecção de outliers pelo método IQR e codificação One-Hot de `ocean_proximity`.
- `Pipeline` com `SimpleImputer` + `StandardScaler` + `SVR`, com o preço em escala de 100 mil.
- Comparação de tratamentos (sem tratamento, mediana, limites de outliers, padronização), comparação de kernels (linear, rbf e poly) e otimização com `GridSearchCV` (5 folds, `scoring='r2'`, 11 configurações).
- **Relatório completo em PDF** (8 páginas) em `relatorios/fernando_panichek_quinsan_relatorio.pdf`, cobrindo introdução, descrição dos dados, tratamento, outliers, modelo, otimização, resultados e conclusão.

### Resultados

| Métrica | Valor |
|---|---|
| R² no teste (4.128 registros) | **0,7711** |
| R² médio na validação cruzada | 0,7806 |
| MAE | US$ 35.931,63 |
| RMSE | US$ 54.764,04 |

- Modelo final: kernel `rbf`, `C=10`, `epsilon=0,1`, `gamma=0,1`.
- Kernels (`C=1`): rbf 0,7437 · poly 0,6527 · linear 0,6108.
- Quase todo o ganho veio da padronização das variáveis e do preço (R² de -0,0485 para 0,7483). Preencher ausentes e limitar outliers teve efeito desprezível, e o `GridSearchCV` acrescentou cerca de 0,03.
- Os outliers foram mantidos no modelo final por não haver evidência de erro nos dados.
- O modelo subestima regiões de valor máximo (o alvo é truncado em US$ 500.001).

### Arquivos de dados

- `housing.csv`, obtido do dataset [Housing no Kaggle](https://www.kaggle.com/datasets/kathuman/housing).

### Compatibilidade

Nenhuma mudança incompatível. Os conteúdos das versões anteriores permanecem inalterados.

---

## Release v5.0.0 — Classificação: Iris Species com Árvore de Decisão

**Data de entrega:** 29/10/2026
**Turma B — Noturno**

### Novidades

- Novo notebook `Iris_Species_com_Arvore_de_Decisão.ipynb`: classificação das espécies de Iris com `DecisionTreeClassifier`.
- Comparação **sem** `GridSearchCV` (`gini`, `max_depth=3`, definido manualmente) e **com** `GridSearchCV` sobre `criterion`, `max_depth`, `min_samples_split` e `min_samples_leaf`.
- Visualização da árvore com `plot_tree`, curva de validação para `max_depth` e importância das variáveis.

### Resultados

| Configuração | Acurácia CV | Acurácia teste |
|---|:---:|:---:|
| Sem `GridSearchCV` (`gini`, profundidade 3) | 93,33% | 96,67% |
| Com `GridSearchCV` (`gini`, profundidade 4) | 94,17% | 93,33% |

- A busca em grade melhorou a validação cruzada em menos de um ponto percentual e piorou o teste em uma amostra, sem ganho real.
- Foi o modelo com menor acurácia em validação cruzada nas duas configurações, o que é esperado de uma árvore única, mais instável que métodos de conjunto.
- O teste possui 30 amostras; a diferença corresponde a 1 observação.

### Compatibilidade

Nenhuma mudança incompatível. Os conteúdos das versões anteriores permanecem inalterados.

---

## Release v4.0.0 — Classificação: Iris Species com Boosting

**Data de entrega:** 29/10/2026
**Turma B — Noturno**

### Novidades

- Novo notebook `Iris_Species_com_Boosting.ipynb`: classificação das espécies de Iris com `GradientBoostingClassifier`.
- Comparação **sem** `GridSearchCV` (100 árvores, `learning_rate=0,1`, `max_depth=3`, `subsample=1,0`) e **com** `GridSearchCV` sobre `n_estimators`, `learning_rate`, `max_depth` e `subsample`.
- Curva de validação para `learning_rate`, importância das variáveis e importância por permutação.

### Resultados

| Configuração | Acurácia CV | Acurácia teste |
|---|:---:|:---:|
| Sem `GridSearchCV` | 96,67% | 96,67% |
| Com `GridSearchCV` | 96,67% | 96,67% |

- A busca em grade escolheu exatamente os hiperparâmetros do modelo sem busca; por isso os resultados são idênticos e a busca não trouxe ganho.
- Recall no teste: 1,00 para *Iris-setosa*, 0,90 para *Iris-versicolor* e 1,00 para *Iris-virginica*. O único erro foi uma *Iris-versicolor* classificada como *Iris-virginica*.
- O modelo atingiu 100% de acurácia no treino, sem prejuízo visível no teste.

### Compatibilidade

Nenhuma mudança incompatível. Os conteúdos das versões anteriores permanecem inalterados.

---

## Release v3.0.0 — Classificação: Iris Species com SVM

**Data de entrega:** 29/10/2026
**Turma B — Noturno**

### Novidades

- Novo projeto de **Machine Learning**: classificação das espécies do dataset Iris com **Support Vector Machine (SVM)**, no notebook `Iris_Species_com_SVM.ipynb`.
- Comparação entre duas abordagens de definição de hiperparâmetros:
  - **Sem `GridSearchCV`**: kernel `rbf`, `C=1`, `gamma='scale'`.
  - **Com `GridSearchCV`**: busca em grade sobre `C`, `kernel` e `gamma`, com validação cruzada de 5 folds.
- `Pipeline` com `StandardScaler` + `SVC`, evitando vazamento de dados.
- Divisão treino/teste de 80/20, estratificada e reprodutível (`random_state=42`).
- Avaliação com acurácia, `classification_report` e matriz de confusão.
- Análise de sensibilidade com curva de validação para o hiperparâmetro `C` e importância de atributos por permutação.
- Conclusão comparativa entre os modelos do projeto (SVM, Boosting e Árvore de Decisão).

### Resultados

| Configuração | Acurácia CV | Acurácia treino | Acurácia teste |
|---|:---:|:---:|:---:|
| Sem `GridSearchCV` (rbf, C=1) | 96,67% | 97,50% | 96,67% |
| Com `GridSearchCV` (linear, C=0,1) | 97,50% | 97,50% | 93,33% |

- A *Iris-setosa* foi classificada perfeitamente; os erros ocorreram entre *Iris-versicolor* e *Iris-virginica*.
- O `GridSearchCV` elevou a acurácia em validação cruzada, mas não melhorou o resultado no teste.
- O teste possui 30 amostras, portanto a diferença entre as configurações corresponde a 1 observação e deve ser interpretada com cautela.

### Arquivos de dados

- `Iris.csv`, obtido do dataset [Iris Species no Kaggle](https://www.kaggle.com/datasets/menegidio/iris-species), utilizado pelos notebooks de classificação.

### Compatibilidade

Nenhuma mudança incompatível. Os conteúdos das versões anteriores permanecem inalterados.

---

## Release v2.0.0 — Data Science: NumPy, SciPy, Pandas e Matplotlib

**Data de entrega:** 21/08/2026
**Turma B — Noturno**

### Novidades

- Adição de **80 exercícios de Data Science**, divididos em quatro bibliotecas:
  - 20 exercícios de **NumPy**
  - 20 exercícios de **SciPy**
  - 20 exercícios de **Pandas**
  - 20 exercícios de **Matplotlib**
- Exercícios do nível fundamental ao avançado, incluindo manipulação de arrays e DataFrames, cálculos científicos, operações estatísticas e visualização de dados.
- Códigos executados, com as saídas registradas no notebook.
- Exercícios de Pandas com leitura e manipulação de arquivos de dados.

### Arquivos de dados

- `iris.data`, utilizado nos exercícios que dependem de dados externos.

### Compatibilidade

Nenhuma mudança incompatível foi introduzida nesta versão.

---

## Release v1.0.0 — Lançamento Inicial: Exercícios de Python

**Data de entrega:** 16/08/2026
**Turma B — Noturno**

### Novidades

- Lista completa com **50 exercícios de Python**.
- Conteúdo dos fundamentos da linguagem (variáveis, condicionais e laços de repetição) a funções, estruturas de dados, recursão e algoritmos.
- Códigos desenvolvidos e testados, com saídas registradas no notebook.
- Explicações da lógica utilizada na resolução de cada exercício.

### Compatibilidade

Por ser a primeira versão do repositório, não há mudanças incompatíveis.
