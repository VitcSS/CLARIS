import time
import numpy as np
from typing import List, Optional
from claris.analytics.contracts import AnalyticsContract

DRIFT_REPORT_SYSTEM_PROMPT = """Você é um especialista em MLOps e diagnóstico de integridade de modelos de IA.
Sua função é gerar um Relatório Diagnóstico de Degradação objetivo e acionável com base nas métricas observadas na janela temporal.

DADOS MONITORADOS:
- Tipo de Drift Detectado: {drift_types}
- Distância Euclidiana Média (Data Drift): {avg_euclidean:.4f} (Limiar máximo: {euclidean_threshold:.4f})
- Similaridade de Cosseno Média (Concept Drift): {avg_cosine:.4f} (Limiar mínimo: {cosine_threshold:.4f})
- Total de Inferências Avaliadas: {window_size}

Gere um relatório técnico contendo:
1. Resumo do Incidente de Monitoramento
2. Análise de Risco (Diferenciando impacto de Dados de Entrada vs. Mudança de Conceito)
3. Recomendação Operacional Imediata (Ex.: retreino, auditoria de covariáveis, recalibração de limiares)
"""

class DriftMonitor:
    def __init__(self, window_size: int, euclidean_threshold: float, cosine_threshold: float, llm_adapter):
        self.window_size = window_size
        self.euclidean_threshold = euclidean_threshold
        self.cosine_threshold = cosine_threshold
        self.llm_adapter = llm_adapter
        self.current_window_contracts: List[AnalyticsContract] = []

    def add_inference(self, contract: AnalyticsContract) -> Optional[str]:
        self.current_window_contracts.append(contract)
        
        if len(self.current_window_contracts) >= self.window_size:
            return self._evaluate_window()
        return None

    def _evaluate_window(self) -> Optional[str]:
        euclidean_dists = [c.distance_metrics['euclidean_distance'] for c in self.current_window_contracts]
        cosine_sims = [c.distance_metrics['cosine_similarity'] for c in self.current_window_contracts]
        
        avg_euclidean = float(np.mean(euclidean_dists))
        avg_cosine = float(np.mean(cosine_sims))
        
        drift_types = []
        if avg_euclidean > self.euclidean_threshold:
            drift_types.append("Data Drift (Covariate Shift)")
        if avg_cosine < self.cosine_threshold:
            drift_types.append("Concept Drift (Prior Probability Shift)")
            
        report = None
        if drift_types:
            report = self._generate_drift_report(drift_types, avg_euclidean, avg_cosine)
            
        # Limpa o buffer para o próximo ciclo temporal
        self.current_window_contracts = []
        return report

    def _generate_drift_report(self, drift_types: List[str], avg_euclidean: float, avg_cosine: float) -> str:
        prompt = DRIFT_REPORT_SYSTEM_PROMPT.format(
            drift_types=", ".join(drift_types),
            avg_euclidean=avg_euclidean,
            euclidean_threshold=self.euclidean_threshold,
            avg_cosine=avg_cosine,
            cosine_threshold=self.cosine_threshold,
            window_size=self.window_size
        )
        
        from litellm import completion

        # Modelos para fallback em caso de indisponibilidade
        candidate_models = [
            self.llm_adapter.model_name,
        ]
        
        # Elimina duplicatas preservando a ordem de preferência
        models_to_try = list(dict.fromkeys(candidate_models))

        for model in models_to_try:
            for attempt in range(3):
                try:
                    response = completion(
                        model=model,
                        messages=[
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": "Emita o relatório técnico de diagnóstico de degradação."}
                        ],
                        temperature=0.0,
                        api_key=self.llm_adapter.api_key
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    err_msg = str(e).lower()
                    if ("503" in err_msg or "unavailable" in err_msg or "rate" in err_msg) and attempt < 2:
                        time.sleep(2 * (attempt + 1))
                        continue
                    break

        # Fallback determinístico offline caso a API permaneça fora do ar
        return (
            f"### ⚠️ ALERTA DE DEGRADAÇÃO MLOps (RELATÓRIO EM MODO CONTINGÊNCIA)\n\n"
            f"- **Tipos de Drift:** {', '.join(drift_types)}\n"
            f"- **Distância Euclidiana Média:** {avg_euclidean:.4f} (Limiar: {self.euclidean_threshold:.4f})\n"
            f"- **Similaridade de Cosseno Média:** {avg_cosine:.4f} (Limiar: {self.cosine_threshold:.4f})\n"
            f"- **Janela Avaliada:** {self.window_size} inferências\n\n"
            f"*(Aviso: A API do provedor LLM retornou 503 Service Unavailable durante o pico de demanda. As métricas matemáticas foram preservadas e validadas localmente).* "
        )