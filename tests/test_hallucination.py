import os
import json
import pytest
from litellm import completion

from claris.nlg.llm_adapter import LLMAdapter
from claris.analytics.contracts import AnalyticsContract, FeatureDelta

@pytest.fixture
def sample_contract():
    return AnalyticsContract(
        instance_id="req_audit_001",
        predicted_class="0",
        convergent_features=[
            FeatureDelta(
                feature_name="Renda Mensal do Solicitante (R$)",
                instance_value=2100.0,
                reference_value=1800.0,
                shap_contribution=-0.20
            )
        ],
        divergent_features=[
            FeatureDelta(
                feature_name="Renda Mensal do Solicitante (R$)",
                instance_value=2100.0,
                reference_value=7500.0,
                shap_contribution=-0.20
            )
        ],
        distance_metrics={
            "euclidean_distance": 0.0,
            "cosine_similarity": 0.95
        }
    )

def test_gemini_faithfulness_audit(sample_contract):
    """
    Avaliação de fidelidade (grounding): verifica se todas as afirmações
    da explicação estão 100% suportadas pelo contrato de dados gerados.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    assert api_key is not None, "GEMINI_API_KEY não configurada no ambiente."

    # 1. Gerar a explicação utilizando o adaptador com temperatura zero
    adapter = LLMAdapter(
        model_name="gemini/gemini-3.5-flash",
        api_key=api_key
    )
    explanation = adapter.generate_explanation(sample_contract)
    context_str = sample_contract.model_dump_json(indent=2)

    # 2. Avaliador independente (LLM-as-a-Judge)
    judge_prompt = f"""Você é um auditor rigoroso de fidelidade factual (Faithfulness Auditor).
Avalie se a EXPLICAÇÃO gerada contém apenas informações estritamente presentes no CONTEXTO JSON, sem inventar valores ou variáveis.

CONTEXTO JSON:
{context_str}

EXPLICAÇÃO AVALIADA:
{explanation}

Responda OBRIGATORIAMENTE em formato JSON com o seguinte esquema:
{{
  "score": <float entre 0.0 e 1.0>,
  "hallucinated_facts": [<lista de afirmações não fundamentadas, se houver>],
  "reasoning": "<breve justificativa do score>"
}}
"""

    response = completion(
        model="gemini/gemini-3.5-flash",
        messages=[
            {"role": "system", "content": "Você é um avaliador estrito de integridade factual."},
            {"role": "user", "content": judge_prompt}
        ],
        temperature=0.0,
        api_key=api_key
    )
    
    raw_content = response.choices[0].message.content.strip()
    
    # Limpeza de blocos de markdown caso o modelo os inclua
    if raw_content.startswith("```json"):
        raw_content = raw_content[7:]
    if raw_content.startswith("```"):
        raw_content = raw_content[3:]
    if raw_content.endswith("```"):
        raw_content = raw_content[:-3]
        
    eval_result = json.loads(raw_content.strip())
    
    print("\n" + "=" * 50)
    print(f"Score de Fidelidade: {eval_result['score']}")
    print(f"Alucinações detectadas: {eval_result.get('hallucinated_facts', [])}")
    print(f"Parecer: {eval_result.get('reasoning', '')}")
    print("=" * 50)

    assert eval_result["score"] >= 0.95, f"Falha de Grounding! Score: {eval_result['score']}"