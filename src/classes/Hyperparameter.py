from typing import Any, Dict, Optional

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from utils.Helper import ErrorCalculator, Step, StepConfig


class HyperparameterOptimization (Step):
    """Step 5: Hyperparameter Optimization with GridSearch"""
    
    def __init__(self, config: Optional[StepConfig] = None):
        super().__init__(config or StepConfig("Step 5", "Hyperparameter Optimization"))
        self.error_calc = ErrorCalculator()
        
        # Input data
        self.X_combined = None
        self.y_combined = None
        self.groups_numeric = None
        self.combined_df = None
        self.demand_features = None
        self.plant_features = None
        self.baseline_rmse = None
        self.logo_rmse_overall = None
        
        # Results storage
        self.param_grid = None
        self.gs_model = None
        self.best_model_rmse = None
        self.best_model_errors = None
        self.best_params = None
    
    def execute(self, results_step1: Dict[str, Any], results_step2: Dict[str, Any], 
               results_step3: Dict[str, Any], results_step4: Dict[str, Any]):
        """Execute hyperparameter optimization"""
        
        # Unpack results
        self.combined_df = results_step3['combined_df']
        self.demand_features = results_step1['demand_features']
        self.plant_features = results_step1['plant_features']
        self.baseline_rmse = results_step2['baseline_rmse_avg']
        self.logo_rmse_overall = results_step4['logo_rmse_overall']
        
        self.X_combined = self.combined_df[self.demand_features + self.plant_features].values
        self.y_combined = self.combined_df['Cost_USD_per_MWh'].values
        
        unique_demands = self.combined_df['Demand ID'].unique()
        demand_to_group = {d: i for i, d in enumerate(unique_demands)}
        self.groups_numeric = np.array([demand_to_group[d] for d in self.combined_df['Demand ID'].values])
        
        # 5.1 Define parameter grid (simplified for faster execution)
        param_candidates: list[dict[str, int]] = [
            {'n_estimators': 100, 'max_depth': 15, 'min_samples_split': 2, 'min_samples_leaf': 1},
            {'n_estimators': 150, 'max_depth': 15, 'min_samples_split': 2, 'min_samples_leaf': 1},
            {'n_estimators': 100, 'max_depth': 20, 'min_samples_split': 2, 'min_samples_leaf': 1},
            {'n_estimators': 100, 'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 2},
        ]
        
        # 5.2 Evaluate parameter combinations (using standard 5-fold CV for speed)
        
        best_score = float('inf')
        self.best_params = None
        cv_results = []
        
        
        for i, params in enumerate(param_candidates):
            model = RandomForestRegressor(random_state=42, n_jobs=-1, **params)  # type: ignore
            
            # Use negative MSE as scoring metric (sklearn convention)
            scores = cross_val_score(model, self.X_combined, self.y_combined, 
                                    cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
            rmse_cv = np.sqrt(-scores.mean())
            
            cv_results.append({
                'params': params,
                'rmse': rmse_cv,
                'std': np.sqrt(scores.std())
            })
            
            if rmse_cv < best_score:
                best_score = rmse_cv
                self.best_params = params
        
        # 5.3 Report best parameters
        # 5.4 Train final model with best params on full data
        self.gs_model = RandomForestRegressor(random_state=42, n_jobs=-1, **self.best_params)  # type: ignore
        self.gs_model.fit(self.X_combined, self.y_combined)
        
        # 5.5 Evaluate with LOGO CV (sample-based for speed)
        
        logo = LeaveOneGroupOut()
        best_model_errors_list = []
        folds_evaluated = 0
        max_folds = min(50, len(self.groups_numeric))  # Sample 50 folds max for speed
        
        for fold, (train_idx, test_idx) in enumerate(logo.split(self.X_combined, self.y_combined, self.groups_numeric)):
            if folds_evaluated >= max_folds:
                break
            
            X_train_best = self.X_combined[train_idx]
            y_train_best = self.y_combined[train_idx]
            X_test_best = self.X_combined[test_idx]
            y_test_best = self.y_combined[test_idx]
            
            model_best = RandomForestRegressor(**self.best_params, random_state=42, n_jobs=-1, verbose=0)  # type: ignore
            model_best.fit(X_train_best, y_train_best)
            
            y_pred_best = model_best.predict(X_test_best)
            min_cost_best = np.min(y_test_best)
            selected_idx_best = np.argmin(y_pred_best)
            selected_cost_best = y_test_best[selected_idx_best]
            error_best = selected_cost_best - min_cost_best
            
            best_model_errors_list.append(error_best)
            folds_evaluated += 1
        
        self.best_model_errors = np.array(best_model_errors_list)
        self.best_model_rmse = np.sqrt(np.mean(self.best_model_errors ** 2))
    
    def get_results(self) -> Dict[str, Any]:
        """Return step results"""
        return {
            'gs_model': self.gs_model,
            'best_params': self.best_params,
            'best_model_rmse': self.best_model_rmse,
            'best_model_errors': self.best_model_errors
        }