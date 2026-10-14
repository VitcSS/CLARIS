# 🔍 CLARIS Framework
**Case-based Latent Analytics for Reliable Inference Synthesis**

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Research%20Project-orange)

O **CLARIS** é uma arquitetura neurosimbólica de explicabilidade (XAI) e monitorização (MLOps) desenvolvida no contexto do Mestrado em Ciência da Computação (PPGCC/UFMG). O framework atua como uma camada mediadora entre modelos preditivos de caixa-preta e o utilizador final.

## ✨ Principais Funcionalidades

1. **Zero Alucinação (Grounded NLG):** Utiliza Raciocínio Baseado em Casos (CBR) no espaço latente de valores SHAP para extrair deltas matemáticos determinísticos, forçando o LLM a gerar explicações baseadas exclusivamente em factos ancorados no histórico do modelo.
2. **Justificativas Contrafactuais:** Responde "por que razão a decisão foi X e não Y" comparando instâncias no espaço latente.
3. **Deteção Orgânica de Drift:** Monitoriza alterações distributivas (Data Drift e Concept Drift) mapeando o aumento anómalo de distâncias (Euclidiana e Cosseno) entre a inferência atual e as suas âncoras históricas, gerando relatórios de degradação em linguagem natural.

## 🚀 Instalação (Desenvolvimento)

Como este é um projeto em desenvolvimento, a instalação é feita localmente utilizando o gestor de pacotes `poetry`:

```bash
git clone [https://github.com/seu-usuario/claris.git](https://github.com/seu-usuario/claris.git)
cd claris
poetry install