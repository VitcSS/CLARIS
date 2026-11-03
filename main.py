import numpy as np
from claris.xai.explainer import ModelExplainer
from claris.cbr.vector_db import VectorDatabase
from claris.analytics.extractor import DeltaExtractor
from claris.nlg.llm_adapter import LLMAdapter

def run_claris_pipeline(instance: np.ndarray, feature_names: list, model, background_data, semantic_dict, db_dimension):
    # 1. Inicialização dos Módulos
    explainer = ModelExplainer(model, background_data, model_type="tree")
    vector_db = VectorDatabase(dimension=db_dimension)
    extractor = DeltaExtractor(semantic_dict)
    llm = LLMAdapter(model_name="gpt-4o") # Configurável
    
    # 2. Camada XAI: Inferência e Mapeamento SHAP
    pred_class, features_dict, shap_vector = explainer.predict_and_explain(instance, feature_names)
    
    # 3. Camada CBR: Busca no Banco Vetorial
    # Em produção, o banco já estaria populado com o histórico (D_conf)
    neighbors_meta, distances = vector_db.search_neighbors(shap_vector, k=5)
    
    # 4. Camada Analytics: Extração Determinística
    contract = extractor.extract_deltas(
        instance_id="req_001",
        predicted_class=str(pred_class),
        instance_features=features_dict,
        instance_shap=shap_vector,
        neighbors_metadata=neighbors_meta,
        distances=distances
    )
    
    # 5. Camada NLG: Geração da Justificativa
    explanation_text = llm.generate_explanation(contract)
    
    return explanation_text