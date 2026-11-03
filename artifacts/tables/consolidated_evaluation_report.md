# CLARIS: Dossiê de Evidências Experimentais e Validação Teórica

Compilação automática de tabelas e métricas para suporte à escrita do artigo.

## Tabela 1: Latência Computacional por Estágio (ms)

| Dataset   |   Features | SHAP (ms)   | FAISS HNSW (ms)   | DeltaExtractor (ms)   |   Total Local (ms) |
|:----------|-----------:|:------------|:------------------|:----------------------|-------------------:|
| HELOC     |         23 | 0.22 ± 0.09 | 0.02 ± 0.02       | 0.13 ± 0.07           |               0.38 |
| HEART     |         13 | 0.26 ± 0.10 | 0.02 ± 0.01       | 0.10 ± 0.08           |               0.39 |
| ADULT     |         13 | 0.24 ± 0.10 | 0.02 ± 0.02       | 0.08 ± 0.04           |               0.34 |
| GERMAN    |         20 | 0.26 ± 0.10 | 0.02 ± 0.02       | 0.13 ± 0.05           |               0.41 |

---

## Tabela 2: Auditoria de Grounding e Testes Adversariais (H1)

| Condição Experimental                   | Grounding Score (RAGAS/Judge)   | Aderência ao Contrato                 |
|:----------------------------------------|:--------------------------------|:--------------------------------------|
| Inferência Factual Normal               | 1.00 (100%)                     | Estrita (Zero Alucinação)             |
| Ataque Mutacional (Contrato Adulterado) | 1.00 (100%)                     | Preservou Valor Injetado Literalmente |

---

## Tabela 3: Sensibilidade Dual de Detecção de Drift (H3)

|   Ruído (Sigma) |   Dist. Euclidiana |   Similaridade Cosseno |
|----------------:|-------------------:|-----------------------:|
|        0        |            0       |               1        |
|        0.222222 |            1.2017  |               0.999949 |
|        0.444444 |            1.90303 |               0.999867 |
|        0.666667 |            2.98664 |               0.999929 |
|        0.888889 |            4.08716 |               0.999993 |
|        1.11111  |            5.62405 |               0.999717 |
|        1.33333  |            5.81973 |               0.969769 |
|        1.55556  |            8.29766 |               0.982666 |
|        1.77778  |            9.71268 |               0.940355 |
|        2        |           10.8251  |               0.999628 |

---

## Tabela 4: Deltas Determinísticos do Caso de Estudo

| Tipo          | Variável                                           |   Instância |   Referência |
|:--------------|:---------------------------------------------------|------------:|-------------:|
| Âncora        | Estimativa Consolidada de Risco Externo            |          89 |         85.8 |
| Âncora        | Meses desde a Consulta de Crédito Anterior         |           5 |         10.2 |
| Âncora        | Média de Duração de Resolução de Linhas de Crédito |          96 |         84.6 |
| Âncora        | Taxa de Utilização de Limite Rotativo (%)          |          18 |         10.6 |
| Âncora        | Percentual de Linhas em Situação Legal             |         100 |        100   |
| Âncora        | Meses desde a Abertura da Linha Mais Recente       |           1 |          8   |
| Contrafactual | Meses desde a Abertura da Linha Mais Antiga        |         209 |        319   |
| Contrafactual | Percentual de Linhas com Saldo Devedor             |          71 |         46.5 |
| Contrafactual | Taxa de Utilização de Financiamento Parcelado (%)  |          59 |         74   |
| Contrafactual | Meses desde a Abertura da Linha Mais Recente       |           1 |         15.5 |
| Contrafactual | Percentual de Linhas com Pagamento Parcelado       |          38 |         25.5 |
| Contrafactual | Taxa de Utilização de Limite Rotativo (%)          |          18 |          6   |

---

## Tabela 5: Comparativo Estrutural contra Baselines da Literatura

| Abordagem               | Natureza da Explicabilidade                  | Fidelidade / Grounding                 | Aderência a Negócio                       | Raciocínio Contrafactual               | Vigilância de MLOps                         |
|:------------------------|:---------------------------------------------|:---------------------------------------|:------------------------------------------|:---------------------------------------|:--------------------------------------------|
| SHAP Isolado (Local)    | Pesos numéricos marginais (valores brutos)   | 1.00 (Determinístico)                  | Baixa (Gráficos técnicos)                 | Não (Apenas importância estática)      | Inexistente                                 |
| CBR Tradicional         | Casos tabulares análogos brutos              | 1.00 (Determinístico)                  | Moderada (Inspeção caso a caso)           | Parcial (Sem vetor de convergência)    | Inexistente                                 |
| Vanilla LLM (Zero-Shot) | Narrativa em linguagem natural livre         | Variável (Risco de Alucinação > 0)     | Alta (Fluidez semântica)                  | Alucinado (Não validado em dados)      | Inexistente                                 |
| CLARIS (Neurosimbólico) | Síntese textual vinculada a contrato estrito | 1.00 (Zero-Hallucination via Pydantic) | Alta (Dossiês setoriais contextualizados) | Sim (Centróides de divergência máxima) | Sim (Sensibilidade dual Euclidiana/Cosseno) |

---

