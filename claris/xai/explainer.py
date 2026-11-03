import shap
import numpy as np
from typing import Any, Dict, Tuple

class ModelExplainer:
    def __init__(self, model: Any, background_data: np.ndarray, model_type: str = "tree"):
        """
        Encapsula o modelo preditivo e inicializa o extrator SHAP.
        """
        self.model = model
        self.model_type = model_type
        
        # Otimização: TreeExplainer é mais rápido para XGBoost/LightGBM
        if self.model_type == "tree":
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.explainer = shap.KernelExplainer(self.model.predict, background_data)

    def predict_and_explain(self, instance: np.ndarray, feature_names: list):
        # 1. Inferência da classe
        prediction = self.model.predict(instance)[0]
        
        # 2. Extração dos valores SHAP
        shap_values = self.explainer.shap_values(instance)
        
        # O TreeExplainer do scikit-learn pode retornar:
        # - Uma lista de arrays 2D: [shap_classe_0, shap_classe_1]
        # - Um array 3D: (1, n_features, n_classes)
        # - Um array 2D simples: (1, n_features)
        if isinstance(shap_values, list):
            # Para classificação binária, seleciona as contribuições da classe 1 (ou da classe predita)
            shap_vector = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        elif isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 3:
                # Formato (1, n_features, n_classes) -> seleciona a classe positiva (índice 1)
                shap_vector = shap_values[0, :, 1] if shap_values.shape[2] > 1 else shap_values[0, :, 0]
            elif shap_values.ndim == 2:
                shap_vector = shap_values[0]
            else:
                shap_vector = shap_values
        else:
            shap_vector = np.array(shap_values)

        # Converte para float32 e assegura que seja 1D com o tamanho exato das features
        shap_vector = np.array(shap_vector, dtype=np.float32).flatten()
        
        # 3. Dicionário de características
        instance_flat = np.array(instance).flatten()
        features_dict = {name: float(val) for name, val in zip(feature_names, instance_flat)}
        
        return prediction, features_dict, shap_vector