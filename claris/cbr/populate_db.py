import numpy as np
from typing import List, Any
from claris.xai.explainer import ModelExplainer
from claris.cbr.vector_db import VectorDatabase

def initialize_vector_database(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    feature_names: List[str], 
    model: Any, 
    db_dimension: int
) -> VectorDatabase:
    """
    Executa a Fase 2 do CLARIS: Constrói o espaço latente S_conf e popula o Banco Vetorial.
    """
    explainer = ModelExplainer(model, X_train, model_type="tree")
    vector_db = VectorDatabase(dimension=db_dimension)
    
    vectors_to_add = []
    metadata_to_add = []
    
    # Itera sobre o histórico confiável para extrair os vetores SHAP
    for idx, instance in enumerate(X_train):
        instance_reshaped = instance.reshape(1, -1)
        
        # Mapeia a instância para o espaço latente
        _, features_dict, shap_vector = explainer.predict_and_explain(instance_reshaped, feature_names)
        
        vectors_to_add.append(shap_vector)
        
        # Armazena os metadados necessários para a extração analítica de deltas
        metadata_to_add.append({
            "instance_id": f"train_{idx}",
            "label": int(y_train[idx]),
            "features": features_dict,
            "shap_vector": shap_vector.tolist()  # <-- Permite calcular o cosseno real no extrator
        })
        
    # Indexa os vetores e metadados no banco vetorial utilizando HNSW
    vector_db.add_instances(np.array(vectors_to_add), metadata_to_add)
    
    print(f"Banco Vetorial populado com {len(vectors_to_add)} instâncias confiáveis.")
    return vector_db