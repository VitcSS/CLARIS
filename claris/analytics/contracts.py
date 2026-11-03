from pydantic import BaseModel, Field
from typing import List, Union, Dict

class FeatureDelta(BaseModel):
    feature_name: str = Field(..., description="Nome da variável mapeada no dicionário semântico")
    instance_value: Union[float, str] = Field(..., description="Valor bruto da variável na nova instância")
    reference_value: Union[float, str] = Field(..., description="Valor médio da variável no grupo de referência (âncora ou contrafactual)")
    shap_contribution: float = Field(..., description="Valor direcional de contribuição marginal (SHAP)")

class AnalyticsContract(BaseModel):
    instance_id: str = Field(..., description="Identificador único da inferência")
    predicted_class: str = Field(..., description="Classe final predita pelo modelo de caixa-preta")
    
    # Ponte de Convergência (Por que a decisão é esta?)
    convergent_features: List[FeatureDelta] = Field(
        ..., 
        description="Variáveis que matematicamente aproximam a inferência do Grupo Âncora"
    )
    
    # Ponte de Divergência (Por que a decisão não é a oposta?)
    divergent_features: List[FeatureDelta] = Field(
        ..., 
        description="Variáveis que matematicamente afastam a inferência do Grupo Contrafactual"
    )
    
    # Gatilhos de MLOps
    distance_metrics: Dict[str, float] = Field(
        ..., 
        description="Métricas espaciais da inferência: {'euclidean_distance': float, 'cosine_similarity': float}"
    )