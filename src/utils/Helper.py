# Import necessary libraries
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any


def Save_checkpoint(checkpoint_dir: str,name:str, logger, data:Any)->None:
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    filtered_data = {}
    for key, value in data.items():
        try:
            pickle.dumps(value)
            filtered_data[key] = value
        except (TypeError, pickle.PicklingError):
            logger.warning(f"Skipping non-serializable item in checkpoint: {key}")
    
    
    with open(f"{checkpoint_dir}{name}.pkl", "wb") as f:
        pickle.dump(filtered_data, f)
    logger.info(f"Checkpoint '{name}' saved with {len(filtered_data)} items.")


def Load_checkpoint(checkpoint_dir: str, name:str, logger)->Any:
    path = f"{checkpoint_dir}{name}.pkl"
    if not os.path.exists(path):
        logger.error(f"Checkpoint '{name}' does not exist.")
        return None
    
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    logger.info(f"Checkpoint '{name}' loaded with {len(data)} items.")
    return data

class Logger:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.logger = __import__("logging").getLogger(__name__)
    
    def header(self, text:str, width:int=80):
        if self.verbose:
            self.logger.info("="*width)
            self.logger.info(text)
            self.logger.info("="*width)
    
    def subheader(self,text:str):
        if self.verbose:
            self.logger.info(f"\n{text}")
            self.logger.info("-"*len(text))
            
    def info(self, text:str, indent:int=0):
        if self.verbose:
            self.logger.info(" "*indent + f"✔ {text}")
    
    def data(self, text:str, indent: int=0):
        if self.verbose:
            self.logger.info(" "*indent + f"➤ {text}")
    
    def metric(self, label:str, value: Any, unit:str="", indent:int=0):
        if self.verbose:
            self.logger.info(" "*indent +f" {label}: {value} {unit}")
        
    def success(self, text:str):
        if self.verbose:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"✔ {text}")
            self.logger.info(f"{'='*80}\n")
        

        
@dataclass
class StepConfig:
    name: str
    description: str
    verbose: bool = True

class Step:
    def __init__(self, config:Optional[StepConfig]=None):
        self.config = config or StepConfig("Step","Generic Step")
        self.logger = Logger(verbose=self.config.verbose) 

    def execute(self, *args, **kwargs):
        raise NotImplementedError("Execute method must be implemented by subclasses")
    
    def get_results(self) -> Dict[str, Any]:
        raise NotImplementedError("get_results method must be implemented by subclasses")


class DataValidator:
    """Validates data quality and properties"""
    
    @staticmethod
    def check_missing_values(df: pd.DataFrame, columns: List[str]) -> Dict[str, int]:
        """Check missing values in specific columns"""
        return {col: df[col].isnull().sum() for col in columns}
    
    @staticmethod
    def get_missing_stats(df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
        """Get comprehensive missing value statistics"""
        total = len(df)
        missing = df[columns].isnull().sum().sum()
        missing_pct = 100 * missing / (len(columns) * total) if total > 0 else 0
        
        # Calculate mean/std only on numeric columns
        numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
        sample_mean = df[numeric_cols].values.mean() if numeric_cols else 0
        sample_std = df[numeric_cols].values.std() if numeric_cols else 0
        
        return {
            'total_records': total,
            'missing_count': missing,
            'missing_percent': missing_pct,
            'feature_count': len(columns),
            'sample_mean': sample_mean,
            'sample_std': sample_std
        }
    
    @staticmethod
    def validate_features(df: pd.DataFrame, prefix: str) -> List[str]:
        """Extract and validate features with given prefix"""
        features = [col for col in df.columns if col.startswith(prefix) and pd.api.types.is_numeric_dtype(df[col])]
        return features

class FeatureScaler:
    """Wrapper for feature scaling operations"""
    
    def __init__(self):
        self.scaler_demand = None
        self.scaler_plant = None
    
    def fit_transform(self, demand_df: pd.DataFrame, plant_df: pd.DataFrame,
                      demand_features: List[str], plant_features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fit scalers and transform data"""
        self.scaler_demand = StandardScaler()
        self.scaler_plant = StandardScaler()
        
        demand_df_scaled = demand_df.copy()
        plant_df_scaled = plant_df.copy()
        
        demand_df_scaled[demand_features] = self.scaler_demand.fit_transform(demand_df[demand_features])
        plant_df_scaled[plant_features] = self.scaler_plant.fit_transform(plant_df[plant_features])
        
        return demand_df_scaled, plant_df_scaled
    
    def get_scalers(self) -> Tuple[Optional[StandardScaler], Optional[StandardScaler]]:
        """Return fitted scalers (may be None if not yet fitted)"""
        return self.scaler_demand, self.scaler_plant

class PlantSelector:
    """Identifies and removes worst-performing plants"""
    
    def __init__(self, threshold_percentile: float = 75):
        self.threshold_percentile = threshold_percentile
        self.threshold_cost = None
        self.good_plants = []
        self.worst_plants = []
    
    def analyze_plants(self, costs_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze plant performance"""
        plant_stats = costs_df.groupby('Plant ID')['Cost_USD_per_MWh'].agg(
            ['median', 'mean', 'count']
        ).sort_values('median')
        
        self.threshold_cost = plant_stats['median'].quantile(self.threshold_percentile / 100)
        self.worst_plants = plant_stats[plant_stats['median'] > self.threshold_cost].index.tolist()
        self.good_plants = plant_stats[plant_stats['median'] <= self.threshold_cost].index.tolist()
        
        return {
            'stats': plant_stats,
            'threshold': self.threshold_cost,
            'good_plants': self.good_plants,
            'worst_plants': self.worst_plants
        }
    
    def filter_data(self, demand_df: pd.DataFrame, plants_df: pd.DataFrame, 
                   costs_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Filter datasets based on plant selection"""
        plants_filtered = plants_df[plants_df['Plant ID'].isin(self.good_plants)].reset_index(drop=True)
        costs_filtered = costs_df[
            (costs_df['Demand ID'].isin(demand_df['Demand ID'])) &
            (costs_df['Plant ID'].isin(self.good_plants))
        ].reset_index(drop=True)
        
        return demand_df, plants_filtered, costs_filtered

class ErrorCalculator:
    """Calculates custom error metrics"""
    
    @staticmethod
    def calculate_plant_selection_error(model, X_test: np.ndarray, test_demands_set: set,
                                       combined_df_test: pd.DataFrame, demand_features: List[str],
                                       plant_features: List[str]) -> np.ndarray:
        """Calculate error using Equation 1: Error(d) = c(p_model, d) - min{c(p,d)}
        
        The model predicts costs for each plant given demand+plant features.
        We select the plant with the lowest predicted cost, then calculate the error
        as the difference between the actual cost of our selected plant and the oracle cost
        (the minimum actual cost across all plants for that demand).
        
        Positive error means we selected a plant worse than the oracle.
        """
        errors = []
        feature_cols = demand_features + plant_features
        
        for demand_id in test_demands_set:
            demand_test_costs = combined_df_test[combined_df_test['Demand ID'] == demand_id].copy()
            
            if len(demand_test_costs) == 0:
                continue
            
            # Oracle: minimum actual cost across all plants for this demand
            actual_costs = demand_test_costs['Cost_USD_per_MWh'].values
            min_actual_cost = actual_costs.min()
            
            # Model prediction: predict cost for each plant given its features
            X_demand = demand_test_costs[feature_cols].values
            predicted_costs = model.predict(X_demand)
            
            # Select plant with lowest predicted cost
            selected_idx = np.argmin(predicted_costs)
            
            # Get the actual cost of the selected plant (not predicted cost)
            selected_actual_cost = actual_costs[selected_idx]
            
            # Error: how much more we paid compared to oracle
            # Positive when we overpay vs oracle
            error = selected_actual_cost - min_actual_cost
            errors.append(error)
        
        return np.array(errors)
    
    @staticmethod
    def calculate_rmse(errors: np.ndarray) -> float:
        """Calculate RMSE from errors (Equation 2)"""
        return np.sqrt(np.mean(errors ** 2))
    
    @staticmethod
    def get_error_statistics(errors: np.ndarray) -> Dict[str, float]:
        """Get comprehensive error statistics (returns Python floats)"""
        # Handle empty input gracefully
        if errors.size == 0:
            return {
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'rmse': 0.0
            }
        return {
            'mean': float(errors.mean()),
            'median': float(np.median(errors)),
            'std': float(errors.std()),
            'min': float(errors.min()),
            'max': float(errors.max()),
            'rmse': float(np.sqrt(np.mean(errors ** 2)))
        }
        

        