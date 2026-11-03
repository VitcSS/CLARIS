# CLARIS: Case-based Latent Analytics for Reliable Inference Synthesis

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: Neurosymbolic](https://img.shields.io/badge/Architecture-Neurosymbolic%20XAI-green.svg)](#arquitetura-do-sistema)


O **CLARIS** é um framework neurosimbólico agnóstico projetado para fornecer **explicabilidade local auditável, alinhamento causal estrito (zero-hallucination) e monitoramento contínuo de degradação temporal (*Data Drift* e *Concept Drift*)** para modelos de Machine Learning de caixa-preta em ambientes de missão crítica.

---

## 🏛️ Visão Geral e Fundamentação Metodológica

Em setores regulados (bancário, clínico, securitário e jurídico), as ferramentas convencionais de Explainable AI (como SHAP ou LIME) geram matrizes numéricas e gráficos de difícil interpretação para tomadores de decisão não-técnicos. Por outro lado, o uso ingênuo de Modelos de Linguagem de Grande Escala (LLMs) para redigir laudos técnicos introduz o risco inaceitável de **alucinações factuais e inferências infundadas**.

O **CLARIS** soluciona esse *trade-off* estabelecendo uma arquitetura híbrida de quatro pilares:
1. **Espaço Latente de Decisão ($S_{conf}$):** A predição de caixa-preta é convertida em um vetor de contribuições marginais (valores SHAP), projetando a instância no espaço topológico de decisão.
2. **Raciocínio Baseado em Casos (CBR via FAISS):** Um banco vetorial de alta performance indexado por grafos HNSW recupera instâncias históricas confiáveis de mesmo desfecho (**Âncoras**) e de desfecho contrastivo (**Contrafactuais**).
3. **Contratos Analíticos Determinísticos (Pydantic):** Extração estrita dos deltas matemáticos de convergência e divergência máxima, acompanhados de métricas geométricas de Distância Euclidiana e Similaridade de Cosseno.
4. **Síntese Factual com Grounding 100% (LLM):** Injeção do contrato serializado em JSON e de um Dossiê de Domínio em um LLM configurado em temperatura zero, forçando-o a atuar como um relator factual determinístico.
5. **Vigilância MLOps Contínua:** Reutilização das distâncias latentes em janelas móveis para disparar diagnósticos automáticos de *Covariate Shift* e *Concept Drift*.

---

## 📂 Estrutura Completa do Repositório

```text
claris/
├── .venv/                         # Ambiente virtual Python
├── claris/                        # Pacote principal da biblioteca
│   ├── __init__.py                # Ponto de entrada e versionamento do pacote
│   ├── xai/                       # Módulo de Explicabilidade e Mapeamento Latente
│   │   ├── __init__.py
│   │   └── explainer.py           # ModelExplainer (SHAP Tree/Kernel agnóstico)
│   ├── cbr/                       # Módulo de Raciocínio Baseado em Casos
│   │   ├── __init__.py
│   │   ├── vector_db.py           # VectorDatabase (FAISS HNSW Flat Index)
│   │   └── populate_db.py         # Rotina de indexação do histórico confiável S_conf
│   ├── analytics/                 # Módulo de Extração Analítica Determinística
│   │   ├── __init__.py
│   │   ├── contracts.py           # Esquemas tipados Pydantic (FeatureDelta, AnalyticsContract)
│   │   └── extractor.py           # DeltaExtractor (Agrupamentos, Cosseno e Divergência Máxima)
│   ├── nlg/                       # Módulo de Geração de Linguagem Natural
│   │   ├── __init__.py
│   │   ├── templates.py           # System Prompts universais de ancoragem estrita
│   │   └── llm_adapter.py         # LLMAdapter agnóstico (LiteLLM / Gemini / OpenAI / Ollama)
│   └── mlops/                     # Módulo de Monitoramento e Integridade Contínua
│       ├── __init__.py
│       └── drift_monitor.py       # DriftMonitor (Janela temporal, Euclidiana e Cosseno)
├── config/                        # Configurações do projeto e arquivos de ambiente
├── data/                          # Armazenamento e cache local de bases de benchmark
├── examples/                      # Suíte oficial de cadernos Jupyter (Benchmarks Reais)
│   ├── 01_credit_risk_heloc.ipynb          # Risco de Crédito FICO (PCA 2D e S_conf)
│   ├── 02_clinical_risk_mimic_benchmark.ipynb # Risco Clínico/Cardiológico (Alvos Terapêuticos)
│   ├── 03_subgroup_drift_adult.ipynb       # Equidade & Subgroup Drift (Censo Sociodemográfico)
│   └── 04_neighborhood_graph_german.ipynb  # Grafo Direcionado de Vizinhança (NetworkX)
├── tests/                         # Suíte de testes automatizados
│   ├── __init__.py
│   ├── test_hallucination.py      # Teste de fidelidade e ausência de alucinação (LLM-as-a-Judge)
│   └── test_drift_simulation.py   # Testes unitários do monitor temporal de MLOps
├── main.py                        # Pipeline de orquestração ponta a ponta
├── requirements.txt               # Dependências do framework
├── setup.py                       # Script de empacotamento do pacote Python
└── README.md                      # Documentação técnica integral da biblioteca
```

---

## 🔬 Módulos Internos do Core

### 1. `claris.xai.explainer` (`ModelExplainer`)
* Envelopa qualquer modelo preditivo de caixa-preta (`scikit-learn`, `xgboost`, `lightgbm`, redes neurais).
* Extrai as contribuições marginais via `shap.TreeExplainer` ou `shap.KernelExplainer`.
* Achata tensores multidimensionais garantindo vetores de dimensão $1 	imes d$ para conformidade geométrica.

### 2. `claris.cbr.vector_db` (`VectorDatabase` e `initialize_vector_database`)
* Implementa o índice `faiss.IndexHNSWFlat` com $M=32$ links de vizinhança.
* Converte e normaliza dados confiáveis de treino ($D_{conf}$) para o espaço latente de decisão ($S_{conf}$).
* Realiza buscas $k$-NN em escala de microssegundos resgatando metadados e distâncias geométricas.

### 3. `claris.analytics.extractor` (`DeltaExtractor` e `AnalyticsContract`)
* **Âncoras:** Identifica os pilares convergentes que sustentam a classificação atual através da agregação dos $N$ vizinhos de mesma classe com maior peso SHAP.
* **Contrafactuais Contrastivos:** Ordena as variáveis divergentes estritamente pela **maior divergência absoluta ($|x_{inst} - \mu_{cf}|$)** em relação ao centróide do agrupamento oposto.
* **Métricas Geométricas Reais:** Calcula matematicamente a **Distância Euclidiana** e a **Similaridade de Cosseno** exata entre os vetores de atributos latentes.

### 4. `claris.nlg.llm_adapter` (`LLMAdapter`)
* Conexão agnóstica a múltiplos provedores via `litellm` (Google Gemini, OpenAI, Claude ou modelos locais via Ollama).
* Trava estritamente a temperatura de decodificação em `temperature=0.0`.
* Injeta dinamicamente **Dossiês de Domínio Especializados**, permitindo que o LLM correlacione os deltas empíricos à teoria de negócio ou literatura médica sem alucinar.

### 5. `claris.mlops.drift_monitor` (`DriftMonitor`)
* Monitoramento de fluxo contínuo através de janelas temporais deslizantes de inferência ($W$).
* Rastreia **Data Drift (Covariate Shift)** através da média da Distância Euclidiana.
* Rastreia **Concept Drift (Prior Shift)** através da queda na Similaridade de Cosseno.
* Emite relatórios técnicos de diagnóstico quando os limiares de segurança estatística são ultrapassados.

---

## 📚 Suíte de Benchmarks e Demonstrações (`examples/`)

O repositório disponibiliza 4 cadernos interativos executáveis de ponta a ponta:

| Notebook | Domínio / Dataset | Atributos e Dimensão | Foco Central de Estudo |
| :--- | :--- | :--- | :--- |
| **`01_credit_risk_heloc.ipynb`** | Concessão de Crédito (`HELOC` / FICO) | 23 variáveis numéricas | Baseline de risco financeiro, saneamento de códigos FICO (`-7, -8, -9`), projeção PCA 2D e alavancas contratuais de aprovação. |
| **`02_clinical_risk_mimic_benchmark.ipynb`** | Risco Clínico (`UCI Heart` / análogo MIMIC) | 13 biomarcadores médicos | Decisão de alta gravidade (isquemia, angina, sobrecarga cardíaca), alvos terapêuticos de intervenção contrafactual e laudo cardiológico. |
| **`03_subgroup_drift_adult.ipynb`** | Equidade & Demografia (`Adult Census`) | 14 variáveis sociodemográficas | Auditoria de equidade (*Fairness*), avaliação de impacto sobre atributos protegidos (sexo, raça, idade) e drift de subpopulações. |
| **`04_neighborhood_graph_german.ipynb`** | Crédito Europeu (`German Credit`) | 20 variáveis contratuais | **Grafo direcionado interativo de vizinhança latente (NetworkX)** com pesos de aresta ponderados por proximidade vetorial e cosseno. |

---

## 🛠️ Instalação e Configuração

### 1. Clonar o Repositório e Criar o Ambiente Virtual

```bash
git clone https://github.com/seu-usuario/claris.git
cd claris

python3 -m venv .venv
source .venv/bin/activate
```

### 2. Instalar as Dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 3. Configurar Chaves de API (LiteLLM / Gemini)

Crie um arquivo `.env` na raiz do projeto ou exporte diretamente no seu terminal:

```bash
export GEMINI_API_KEY="sua-chave-api-aqui"
```

---

## 🚀 Guia de Uso Rápido (Quickstart)

```python
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from claris.xai.explainer import ModelExplainer
from claris.cbr.populate_db import initialize_vector_database
from claris.analytics.extractor import DeltaExtractor
from claris.nlg.llm_adapter import LLMAdapter

# 1. Dados e Modelo de Treinamento
X_train = np.array([[5000, 30], [2000, 22], [8000, 45], [1500, 19]], dtype=np.float32)
y_train = np.array([1, 0, 1, 0])
feature_names = ["renda_mensal", "idade"]
semantic_dict = {"renda_mensal": "Renda Mensal do Solicitante ($)", "idade": "Idade (anos)"}

model = GradientBoostingClassifier().fit(X_train, y_train)

# 2. Inicializar o Espaço Latente S_conf (FAISS HNSW)
vector_db = initialize_vector_database(
    X_train=X_train, y_train=y_train,
    feature_names=feature_names, model=model,
    db_dimension=len(feature_names)
)

# 3. Inferência e Extração Analítica
explainer = ModelExplainer(model, X_train, model_type="tree")
extractor = DeltaExtractor(semantic_dict)

x_new = np.array([[2100, 23]], dtype=np.float32)
pred, feats, shap_vec = explainer.predict_and_explain(x_new, feature_names)
neighbors, dists = vector_db.search_neighbors(shap_vec, k=3)

contract = extractor.extract_deltas(
    instance_id="req_001", predicted_class=str(pred),
    instance_features=feats, instance_shap=shap_vec,
    neighbors_metadata=neighbors, distances=dists
)

# 4. Síntese Explicativa Factual (Temperatura Zero)
llm = LLMAdapter(model_name="gemini/gemini-1.5-flash")
laudo = llm.generate_explanation(contract)
print(laudo)
```

---

## 🧪 Auditoria de Fidelidade e Anti-Alucinação

O framework é testado rigorosamente sob o paradigma *LLM-as-a-Judge*, validando se todas as asserções do laudo textual possuem correspondência unívoca no contrato JSON emitido pelo extrator.

Para executar os testes automatizados de fidelidade:

```bash
pytest -s tests/test_hallucination.py
```

O teste assegura que o índice de ancoragem (*Faithfulness Score*) atinja **1.0 (100% de aderência aos dados)**, rejeitando terminantemente qualquer inclusão de variáveis ou valores ausentes no contrato.

---

## 📄 Licença

Distribuído sob a licença MIT. Consulte `LICENSE` para mais informações.
