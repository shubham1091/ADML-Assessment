from typing import Any, Dict, Optional

import pandas as pd
from utils.Helper import DataValidator, FeatureScaler, PlantSelector, Step, StepConfig
from utils.logging_config import get_logger

logger = get_logger(__name__)


class DataPreparation(Step):
    def __init__(self, config:Optional[StepConfig]=None):
        super().__init__(config or StepConfig("Step 1", "Data Preparation"))
        
        self.validator = DataValidator()
        self.scaler = FeatureScaler()
        self.plant_selector = PlantSelector()
        
        ## Results storage
        self.demand_df_clean = None
        self.plants_df_filtered = None
        self.costs_df_filtered = None
        self.demand_features = None
        self.plant_features = None
        self.metadata = {}
        
    
    
    def _print_summary(self):
        def _safe_len(obj):
            return len(obj) if obj is not None else 0

        logger.info("Data Preparation Summary:")
        logger.info(f"  - Demand scenarios: {_safe_len(self.demand_df_clean)}")
        logger.info(f"  - Plants: {_safe_len(self.plants_df_filtered)}")
        logger.info(f"  - Cost records: {_safe_len(self.costs_df_filtered)}")
        logger.info(f"  - Demand features: {_safe_len(self.demand_features)}")
        logger.info(f"  - Plant features: {_safe_len(self.plant_features)}")
    
    def get_results(self) -> Dict[str, Any]:
        return {
            "demand_df": self.demand_df_clean,
            "plants_df": self.plants_df_filtered,
            "costs_df": self.costs_df_filtered,
            "demand_features": self.demand_features,
            "plant_features": self.plant_features,
            "metadata": self.metadata
        }
            
    def execute(self, demand_path:str, plants_path:str, costs_path:str):
        
        ## Load data
        demand_df = pd.read_csv(demand_path, keep_default_na=False, na_values=[""])
        plants_df = pd.read_csv(plants_path, keep_default_na=False, na_values=[""])
        costs_df = pd.read_csv(costs_path, keep_default_na=False, na_values=[""])
        
        ## Handle missing values
        demand_feature_cols = self.validator.validate_features(demand_df, "DF")
        self.demand_features = demand_feature_cols
        
        missing_stats = self.validator.get_missing_stats(demand_df, demand_feature_cols)
        
        demand_df_clean = demand_df.copy()
        demand_df_clean[demand_feature_cols] = demand_df_clean[demand_feature_cols].fillna(demand_df_clean[demand_feature_cols].mean())
        
        ## Feature scaling
        plant_feature_cols = self.validator.validate_features(plants_df, "PF")
        self.plant_features = plant_feature_cols
        
        demand_df_scaled, plants_df_scaled = self.scaler.fit_transform(demand_df_clean, plants_df, self.demand_features, self.plant_features)
        
        
        ## Remove worst performing plants
        plant_analysis = self.plant_selector.analyze_plants(costs_df)
        
        demand_df_final, plant_df_filtered, costs_df_filtered = self.plant_selector.filter_data(demand_df_scaled, plants_df_scaled, costs_df)
        
        ## Store results
        self.demand_df_clean = demand_df_final
        self.plants_df_filtered = plant_df_filtered
        self.costs_df_filtered = costs_df_filtered
        
        ## remove NaN cost values
        nan_costs = self.costs_df_filtered["Cost_USD_per_MWh"].isna().sum()
        if nan_costs > 0:
            self.costs_df_filtered = self.costs_df_filtered.dropna(subset=["Cost_USD_per_MWh"])
        
        ## Store metadata
        self.metadata = {
            "demand_features": self.demand_features,
            "plant_features": self.plant_features,
            "missing_stats": missing_stats,
            "plant_analysis": plant_analysis
        }
        
        ## Print summary
        self._print_summary()
