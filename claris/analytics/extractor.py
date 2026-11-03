import numpy as np
from typing import List, Dict, Any
from claris.analytics.contracts import AnalyticsContract, FeatureDelta

class DeltaExtractor:
    def __init__(self, semantic_dictionary: Dict[str, str]):
        self.semantic_dictionary = semantic_dictionary

    def _split_groups(self, predicted_class: str, neighbors_metadata: List[Dict[str, Any]]):
        anchors = [n for n in neighbors_metadata if str(n.get('label')) == str(predicted_class)]
        counterfactuals = [n for n in neighbors_metadata if str(n.get('label')) != str(predicted_class)]
        return anchors, counterfactuals

    def _compute_cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calcula a similaridade de cosseno matemática real entre dois vetores."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def extract_deltas(
        self, 
        instance_id: str, 
        predicted_class: str, 
        instance_features: Dict[str, float], 
        instance_shap: np.ndarray, 
        neighbors_metadata: List[Dict[str, Any]], 
        distances: List[float], 
        top_k_features: int = 6, 
        n_neighbors: int = 3
    ) -> AnalyticsContract:
        
        anchors, counterfactuals = self._split_groups(predicted_class, neighbors_metadata)
        feature_names = list(instance_features.keys())
        
        # 1. ÂNCORAS (Convergentes)
        convergent_features = []
        best_anchor_shap = None
        if anchors:
            active_anchors = anchors[:n_neighbors]
            top_conv_indices = np.argsort(np.abs(instance_shap))[::-1][:top_k_features]
            
            for idx in top_conv_indices:
                fn = feature_names[idx]
                vals = [float(a['features'].get(fn, instance_features[fn])) for a in active_anchors]
                mean_ref = float(np.mean(vals))
                
                convergent_features.append(FeatureDelta(
                    feature_name=self.semantic_dictionary.get(fn, fn),
                    instance_value=float(instance_features[fn]),
                    reference_value=round(mean_ref, 2),
                    shap_contribution=float(instance_shap[idx])
                ))
            
            # Recupera o vetor SHAP da âncora mais próxima se armazenado nos metadados, 
            # ou usa os atributos normalizados para calcular a direção
            if 'shap_vector' in anchors[0]:
                best_anchor_shap = np.array(anchors[0]['shap_vector'], dtype=np.float32)

        # 2. CONTRAFACTUAIS (Divergentes com maior contraste)
        divergent_features = []
        if counterfactuals:
            active_cfs = counterfactuals[:n_neighbors]
            cf_means = {
                fn: float(np.mean([float(c['features'].get(fn, instance_features[fn])) for c in active_cfs]))
                for fn in feature_names
            }
            
            divergence_list = []
            for idx, fn in enumerate(feature_names):
                diff = abs(float(instance_features[fn]) - cf_means[fn])
                divergence_list.append((diff, idx, fn))
            
            divergence_list.sort(key=lambda item: item[0], reverse=True)
            top_divergent = divergence_list[:top_k_features]
            
            for diff_val, idx, fn in top_divergent:
                divergent_features.append(FeatureDelta(
                    feature_name=self.semantic_dictionary.get(fn, fn),
                    instance_value=float(instance_features[fn]),
                    reference_value=round(cf_means[fn], 2),
                    shap_contribution=float(instance_shap[idx])
                ))

        # 3. CÁLCULO REAL DAS MÉTRICAS DE DISTÂNCIA
        euclidean_dist = float(distances[0]) if distances and len(distances) > 0 else 0.0
        
        if best_anchor_shap is not None:
            cosine_sim = self._compute_cosine_similarity(instance_shap, best_anchor_shap)
        elif anchors and 'features' in anchors[0]:
            # Fallback caso o banco guarde apenas features brutas nos metadados:
            anchor_vals = np.array([float(anchors[0]['features'].get(fn, 0.0)) for fn in feature_names])
            instance_vals = np.array([float(instance_features.get(fn, 0.0)) for fn in feature_names])
            cosine_sim = self._compute_cosine_similarity(instance_vals, anchor_vals)
        else:
            cosine_sim = 1.0

        return AnalyticsContract(
            instance_id=instance_id,
            predicted_class=str(predicted_class),
            convergent_features=convergent_features,
            divergent_features=divergent_features,
            distance_metrics={
                "euclidean_distance": euclidean_dist,
                "cosine_similarity": cosine_sim
            }
        )