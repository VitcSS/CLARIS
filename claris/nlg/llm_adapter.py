# claris/nlg/llm_adapter.py
from typing import Optional
from litellm import completion
from claris.analytics.contracts import AnalyticsContract
from claris.nlg.templates import EXPLANATION_SYSTEM_PROMPT

class LLMAdapter:
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key

    def generate_explanation(self, analytics_data: AnalyticsContract, domain_dossier: Optional[str] = None) -> str:
        analytics_json = analytics_data.model_dump_json(indent=2)
        
        # Formata a seção do dossiê caso fornecida pelo usuário
        if domain_dossier:
            domain_section = f"DOSSIÊ TEÓRICO E DIRETRIZES DO PROBLEMA:\n{domain_dossier.strip()}\n"
        else:
            domain_section = "DOSSIÊ TEÓRICO DO PROBLEMA: Interprete os resultados segundo a teoria geral do domínio dos dados fornecidos."

        system_prompt = EXPLANATION_SYSTEM_PROMPT.format(
            domain_dossier_section=domain_section,
            analytics_json=analytics_json
        )

        try:
            response = completion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Gere o parecer técnico completo, fundamentando os diferenciais numéricos do contrato com base na teoria de domínio do dossiê."}
                ],
                temperature=0.0,
                api_key=self.api_key
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Falha na comunicação com o LLM: {str(e)}")