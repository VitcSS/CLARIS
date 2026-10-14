# Referência da API (Módulo `src/claris`)

O framework CLARIS foi modularizado para permitir a substituição de componentes (ex: trocar FAISS por Milvus, ou Llama3 por GPT-4) sem quebrar o sistema.

---

### `claris.core.ClarisFramework`
Classe principal que orquestra o pipeline *end-to-end*.

*   `__init__(model, historical_data, data_dictionary, llm_backend)`
*   `explain(X_new: pd.DataFrame) -> ExplanationObject`
*   `check_drift_status() -> str`

---

### `claris.explainer.LatentExplainer`
*Wrapper* focado na extração SHAP e conversão para o espaço latente.

*   `__init__(model, background_data)`
*   `transform_to_latent(X: pd.DataFrame) -> np.ndarray`

---

### `claris.cbr_engine.CBREngine`
Gere o Banco de Dados Vetorial e as buscas baseadas em casos.

*   `__init__(index_type="hnsw")`
*   `build_index(latent_matrix: np.ndarray, metadata: pd.DataFrame)`
*   `get_anchors(s_new: np.ndarray, target_label: int, k: int=5) -> pd.DataFrame`
*   `get_counterfactuals(s_new: np.ndarray, target_label: int, k: int=5) -> pd.DataFrame`

---

### `claris.delta_extractor.DeltaExtractor`
Responsável pela matemática determinística para evitar alucinações.

*   `extract_convergences(x_new, anchors_df) -> dict`
*   `extract_divergences(x_new, counterfactuals_df) -> dict`
*   `build_json_payload(convergences, divergences) -> str`

---

### `claris.llm_synthesizer.LLMSynthesizer`
Gere a injeção do contexto rígido no modelo de linguagem.

*   `__init__(backend="llama3", temperature=0.0)`
*   `generate_narrative(delta_json: str, target_label: int, data_dictionary: dict) -> str`
*   `generate_drift_report(drift_metrics_json: dict) -> str`

---

### `claris.drift_monitor.DriftMonitor`
Rastreia a saúde do modelo em produção baseado na variação das métricas do CBR.

*   `__init__(baseline_std_dev: float, window_size: int=100)`
*   `log_distances(euclidean_dist, cosine_sim)`
*   `is_drifting() -> bool`