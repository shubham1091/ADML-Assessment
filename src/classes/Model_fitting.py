from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from utils.Helper import ErrorCalculator, Step, StepConfig
from utils.logging_config import get_logger

logger = get_logger(__name__)


class MLModelFitting(Step):
    def __init__(self, config:Optional[StepConfig]=None, test_size_percent:float=5):
        super().__init__(config or StepConfig("Step 3", "Machine Learning Model Fitting"))
        self.error_calc = ErrorCalculator()
        self.test_size_percent = test_size_percent
        
        # Input data
        self.demand_df = None
        self.plants_df = None
        self.costs_df = None
        self.demand_features = None
        self.plant_features = None
        self.baseline_rmse = None
        
        # Results
        self.combined_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.test_demands_set = None
        self.model = None
        self.rsme_step3 = None
        self.errors_step3 = None
        self.model_performance = {}
        
        
    def execute(self, results_step_one:Dict[str,Any], results_step_two:Dict[str,Any]):
        
        # Unpack results
        self.demand_df = results_step_one["demand_df"]
        self.plants_df = results_step_one["plants_df"]
        self.costs_df = results_step_one["costs_df"]
        self.demand_features = results_step_one["demand_features"]
        self.plant_features = results_step_one["plant_features"]
        self.baseline_rmse = results_step_two["baseline_rmse_avg"]
        
        # Create combined dataset
        combined_data = []
        for _, cost_row in self.costs_df.iterrows():
            demand_id = cost_row["Demand ID"]
            plant_id = cost_row["Plant ID"]
            cost = cost_row["Cost_USD_per_MWh"]
            
            demand_row = self.demand_df[self.demand_df["Demand ID"] == demand_id].iloc[0]
            plant_row = self.plants_df[self.plants_df["Plant ID"] == plant_id].iloc[0]
            
            row = {"Demand ID": demand_id, "Plant ID": plant_id}
            for feat in self.demand_features:
                row[feat] = demand_row[feat]
                
            for feat in self.plant_features:
                row[feat] = plant_row[feat]
            
            row['Cost_USD_per_MWh'] = cost
            combined_data.append(row)
            
        self.combined_df = pd.DataFrame(combined_data)
        
        # Remove any rows with NaN values
        self.combined_df = self.combined_df.dropna().reset_index(drop=True)
        
        # Train/test split grouped by Demand ID
        np.random.seed(42)
        unique_demands = self.combined_df["Demand ID"].unique()
        n_test = max(1, int(len(unique_demands) * self.test_size_percent / 100))
        self.test_demands_set = set(np.random.choice(unique_demands, size=n_test, replace=False))

        train_mask = ~self.combined_df["Demand ID"].isin(self.test_demands_set)
        test_mask = self.combined_df["Demand ID"].isin(self.test_demands_set)
        
        all_features = self.demand_features + self.plant_features
        self.X_train = self.combined_df[train_mask][all_features].values.astype(np.float64)
        self.y_train = self.combined_df[train_mask]["Cost_USD_per_MWh"].values.astype(np.float64)
        self.X_test = self.combined_df[test_mask][all_features].values.astype(np.float64)
        self.y_test = self.combined_df[test_mask]["Cost_USD_per_MWh"].values.astype(np.float64)
        
        # Train regression model
        self.model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1, verbose=0)
        self.model.fit(self.X_train, self.y_train.tolist())
        
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        
        train_r2 = r2_score(self.y_train.tolist(), y_train_pred.tolist())
        test_r2 = r2_score(self.y_test.tolist(), y_test_pred.tolist())
        train_mae = mean_absolute_error(self.y_train.tolist(), y_train_pred.tolist())
        test_mae = mean_absolute_error(self.y_test.tolist(), y_test_pred.tolist())
        
        # Calculate custom error (plant selection error)
        combined_df_test = self.combined_df[test_mask].copy()
        self.errors_step3 = self.error_calc.calculate_plant_selection_error(self.model, self.X_test, self.test_demands_set, combined_df_test, self.demand_features, self.plant_features)
        self.rsme_step3 = self.error_calc.calculate_rmse(self.errors_step3)
        
        # Store performance metrics
        self.model_performance = {
            "train_r2": train_r2,
            "test_r2": test_r2,
            "train_mae": train_mae,
            "test_mae": test_mae,
            "model_rmse": self.rsme_step3,
            "baseline_rmse": self.baseline_rmse,
            "improvement_percent": (self.baseline_rmse - self.rsme_step3) / self.baseline_rmse * 100 if self.baseline_rmse > 0 else 0
        }
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("ML MODEL FITTING SUMMARY (Step 3)")
        logger.info("="*80)
        logger.info(f"Training set size: {len(self.X_train)} samples")
        logger.info(f"Test set size: {len(self.X_test)} samples")
        logger.info(f"Number of test demands: {len(self.test_demands_set)}")
        logger.info(f"\nModel Performance (Standard Regression Metrics):")
        logger.info(f"  - Train R²: {train_r2:.4f}")
        logger.info(f"  - Test R²: {test_r2:.4f}")
        logger.info(f"  - Train MAE: ${train_mae:.2f}/MWh")
        logger.info(f"  - Test MAE: ${test_mae:.2f}/MWh")
        logger.info(f"\nPlant Selection Error Metrics (Eq. 1 & 2):")
        logger.info(f"  - Model RMSE (Step 3): ${self.rsme_step3:.2f}/MWh")
        logger.info(f"  - Baseline RMSE (Step 2): ${self.baseline_rmse:.2f}/MWh")
        logger.info(f"  - Improvement: {self.model_performance['improvement_percent']:.2f}%")
        logger.info(f"  - Mean Error: ${np.mean(self.errors_step3):.2f}/MWh")
        logger.info(f"  - Std Error: ${np.std(self.errors_step3):.2f}/MWh")
        logger.info("="*80 + "\n")
    
    def get_results(self) -> Dict[str, Any]:
        return {
            "combined_df": self.combined_df,
            "X_train": self.X_train,
            "X_test": self.X_test,
            "y_train": self.y_train,
            "y_test": self.y_test,
            "test_demands_set": self.test_demands_set,
            "model": self.model,
            "model_rmse": self.rsme_step3,
            "model_errors": self.errors_step3,
            "model_performance": self.model_performance,
            "demand_features": self.demand_features,
            "plant_features": self.plant_features
        }
