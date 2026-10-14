# Arquitetura do Sistema: CLARIS

O framework CLARIS foi desenhado como uma arquitetura modular que atua em 6 camadas principais. O objetivo deste design é garantir a separação estrita entre o cálculo matemático determinístico (XAI/CBR) e o motor de geração de texto (LLM), eliminando a ocorrência de alucinações.

## Diagrama de Fluxo (Mermaid)

```mermaid
flowchart TD
    classDef entrada fill:#e1f5fe,stroke:#03a9f4,stroke-width:2px;
    classDef motor fill:#fff3e0,stroke:#ff9800,stroke-width:2px;
    classDef db fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px;
    classDef llm fill:#e8f5e9,stroke:#4caf50,stroke-width:2px;
    classDef saida fill:#ffebee,stroke:#f44336,stroke-width:2px;

    subgraph Entradas ["1. Camada de Entradas"]
        X["Nova Instância (x_new)"]:::entrada
        DICT["Dicionário de Variáveis"]:::entrada
        MDL["Modelo Preditivo Treinado"]:::entrada
    end

    subgraph XAI ["2. Camada XAI"]
        PRED["Predição (y_new)"]:::motor
        SHAP["TreeExplainer (SHAP)"]:::motor
        VEC["Vetor Latente SHAP (s_new)"]:::motor
    end

    subgraph CBR ["3. Camada CBR"]
        VDB[("(Banco Vetorial)\nHistórico Confiável Indexado")]:::db
        KNN_A["Busca de Âncoras"]:::motor
        KNN_C["Busca Contrafactual"]:::motor
    end

    subgraph Deltas ["4. Camada de Deltas"]
        DELTA_A["Convergência (Âncoras)"]:::motor
        DELTA_C["Divergência (Contrafactuais)"]:::motor
        DRIFT["Monitor de Drift"]:::motor
    end

    subgraph LLM ["5. Síntese Generativa"]
        PROMPT["Engenharia de Prompt\n(JSON Inject)"]:::llm
        GEN["LLM (Temperatura=0.0)"]:::llm
    end

    subgraph Saidas ["6. Saídas"]
        EXP["Narrativa Explicativa"]:::saida
        REP["Relatório Diagnóstico"]:::saida
    end

    X --> MDL & SHAP
    MDL --> PRED
    SHAP --> VEC
    VEC --> KNN_A & KNN_C
    VDB -. "Cosseno" .-> KNN_A & KNN_C
    KNN_A --> DELTA_A
    KNN_C --> DELTA_C
    KNN_A & KNN_C -. "Distância" .-> DRIFT
    DELTA_A & DELTA_C & PRED & DICT --> PROMPT
    DRIFT -- "Gatilho" --> PROMPT
    PROMPT --> GEN --> EXP & REP