from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from utils.Helper import ErrorCalculator, Step, StepConfig
from utils.logging_config import get_logger

logger = get_logger(__name__)


class CrossValidation(Step):
    def __init__(self, config: Optional[StepConfig] = None):
        super().__init__(config or StepConfig("Step 4", "Cross-Validation (Leave-One-Group-Out)"))
        self.error_calc = ErrorCalculator()
        
        # Input data
        self.X_combined = None
        self.y_combined = None
        self.groups_numeric = None
        self.combined_df = None
        self.demand_features = None
        self.plant_features = None
        self.baseline_rmse = None
        self.rmse_step3 = None
        
        # Results storage
        self.logo_errors_all = None
        self.logo_rmse_overall = None
        self.fold_results = []
        self.cv_scores = []
        
        
    def execute(self, results_step_one: Dict[str, Any], results_step_two: Dict[str, Any], 
                results_step_three: Dict[str, Any]):
        
        # Unpack results
        self.combined_df = results_step_three["combined_df"]
        self.demand_features = results_step_one["demand_features"]
        self.plant_features = results_step_one["plant_features"]
        self.baseline_rmse = results_step_two["baseline_rmse_avg"]
        self.rmse_step3 = results_step_three["model_rmse"]
        
        # Setup LOGO CV
        self.X_combined = self.combined_df[self.demand_features + self.plant_features].values
        self.y_combined = self.combined_df["Cost_USD_per_MWh"].values
        
        unique_demands_all = self.combined_df["Demand ID"].unique()
        demand_to_group = {d: i for i, d in enumerate(unique_demands_all)}
        self.groups_numeric = np.array([demand_to_group[d] for d in self.combined_df["Demand ID"].values])
        
        # Perform LOGO CV
        logo = LeaveOneGroupOut()
        logo_rmses = []
        self.logo_errors_all = []
        self.cv_scores = []
        self.fold_results = []

        total_folds = len(unique_demands_all)
        logger.info("\n" + "=" * 80)
        logger.info("LOGO CROSS-VALIDATION EXECUTION")
        logger.info("=" * 80)
        logger.info(f"Total folds (number of demands): {total_folds}")
        logger.info(f"Training model with {len(self.X_combined)} samples across {total_folds} demand groups\n")
        
        # Precompute progress milestones
        progress_interval = max(1, total_folds // 10)
        milestones = [(i + 1) * progress_interval for i in range(10)]
        
        for fold, (train_idx, test_idx) in enumerate(logo.split(self.X_combined, self.y_combined, self.groups_numeric)):
            X_train_logo = self.X_combined[train_idx]
            y_train_logo = self.y_combined[train_idx]
            X_test_logo = self.X_combined[test_idx]
            y_test_logo = self.y_combined[test_idx]
            test_demands_logo = self.combined_df.iloc[test_idx]["Demand ID"].values
            
            # Train model on this fold
            model_logo = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1, verbose=0)
            model_logo.fit(X_train_logo, y_train_logo)
            
            # Get predictions for test fold
            y_pred = model_logo.predict(X_test_logo)
            
            # Calculate error using Equation 1
            min_cost = np.min(y_test_logo)
            selected_idx = np.argmin(y_pred)
            selected_actual_cost = y_test_logo[selected_idx]
            error = selected_actual_cost - min_cost
            
            rmse = np.sqrt(error ** 2)
            logo_rmses.append(rmse)
            self.logo_errors_all.append(error)
            self.cv_scores.append(-error)  # Store negative for sklearn convention
            
            fold_results_item = {
                "demand_id": test_demands_logo[0],
                "fold": fold + 1,
                "error": float(error),
                "selected_cost": float(selected_actual_cost),
                "optimal_cost": float(min_cost),
                "test_set_size": len(y_test_logo)
            }
            
            self.fold_results.append(fold_results_item)
            
            # Print progress at milestones
            if fold + 1 in milestones:
                logger.info(f"  Completed fold {fold + 1}/{total_folds}")
        
        # Convert to numpy array
        self.logo_errors_all = np.array(self.logo_errors_all)
        self.cv_scores = np.array(self.cv_scores)
        
        # Calculate overall RMSE using Equation 2
        self.logo_rmse_overall = np.sqrt(np.mean(self.logo_errors_all ** 2))
        
        # Get error statistics
        error_stats = self.error_calc.get_error_statistics(self.logo_errors_all)
        
        # Print results
        logger.info("\n" + "=" * 80)
        logger.info("LOGO CROSS-VALIDATION RESULTS (Step 4)")
        logger.info("=" * 80)
        logger.info(f"\nError Statistics (Equation 1):")
        logger.info(f"  - Mean Error: ${error_stats['mean']:.2f}/MWh")
        logger.info(f"  - Median Error: ${error_stats['median']:.2f}/MWh")
        logger.info(f"  - Std Error: ${error_stats['std']:.2f}/MWh")
        logger.info(f"  - Min Error: ${error_stats['min']:.2f}/MWh")
        logger.info(f"  - Max Error: ${error_stats['max']:.2f}/MWh")
        
        logger.info(f"\nOverall RMSE (Equation 2): ${self.logo_rmse_overall:.2f}/MWh")
        
        logger.info(f"\nComparison with Previous Results:")
        logger.info(f"  - Step 2 Baseline RMSE: ${self.baseline_rmse:.2f}/MWh")
        logger.info(f"  - Step 3 Train-Test RMSE: ${self.rmse_step3:.2f}/MWh")
        logger.info(f"  - Step 4 LOGO RMSE: ${self.logo_rmse_overall:.2f}/MWh")
        
        improvement_vs_baseline = (self.baseline_rmse - self.logo_rmse_overall) / self.baseline_rmse * 100
        improvement_vs_step3 = (self.rmse_step3 - self.logo_rmse_overall) / self.rmse_step3 * 100
        
        logger.info(f"\nImprovement Analysis:")
        logger.info(f"  - vs Baseline (Step 2): {improvement_vs_baseline:+.2f}%")
        logger.info(f"  - vs Train-Test (Step 3): {improvement_vs_step3:+.2f}%")
        logger.info("=" * 80 + "\n")
    
    def get_results(self) -> Dict[str, Any]:
        return {
            "logo_errors_all": self.logo_errors_all,
            "logo_rmse_overall": self.logo_rmse_overall,
            "cv_scores": self.cv_scores,
            "fold_results": self.fold_results,
        }
