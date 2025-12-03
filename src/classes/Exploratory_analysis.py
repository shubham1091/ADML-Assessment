import os
from typing import Any, Optional,Dict
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from utils.Helper import ErrorCalculator, Step, StepConfig


class ExploratoryAnalysis(Step):
    def __init__(self, config:Optional[StepConfig]=None):
        super().__init__(config or StepConfig("Step 2", "Exploratory Data Analysis"))
        self.error_clc = ErrorCalculator()
        
        self.demand_df = None
        self.plants_df = None
        self.costs_df = None
        self.demand_features = None
        self.plant_features = None
        
        self.correlation_insights = {}
    
    def execute(self, results_step_one:Dict[str,Any]):
        
        os.makedirs("Data/plots", exist_ok=True)
        
        self.demand_df = results_step_one["demand_df"]
        self.plants_df = results_step_one["plants_df"]
        self.costs_df = results_step_one["costs_df"]
        self.demand_features = results_step_one["demand_features"]
        self.plant_features = results_step_one["plant_features"]
        
        merged_df = self.costs_df.copy()
        merged_df = merged_df.merge(self.demand_df[["Demand ID"] + self.demand_features], on="Demand ID", how="left")
        merged_df = merged_df.merge(self.plants_df[["Plant ID"] + self.plant_features], on="Plant ID", how="left")
        
        # ============================================================================
        # 1. DEMAND FEATURES DISTRIBUTION
        # ============================================================================
        fig, axes = plt.subplots(3, 4, figsize=(16, 10))
        fig.suptitle("Distribution of Demand features (DF1-DF12)", fontsize=12, fontweight='bold')
        
        for idx, col in enumerate(self.demand_features):
            ax = axes[idx//4, idx%4]
            self.demand_df[col].hist(bins=30, ax=ax, edgecolor="black", alpha=0.7)
            ax.set_title(col, fontweight='bold')
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
        
        plt.tight_layout()
        plt.savefig("Data/plots/demand_features_distribution.png", dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()
        
        # ============================================================================
        # 1B. PLANT FEATURES DISTRIBUTION
        # ============================================================================
        fig, axes = plt.subplots(3, 6, figsize=(18, 10))
        fig.suptitle("Distribution of Plant features (PF1-PF18)", fontsize=12, fontweight='bold')
        
        for idx, col in enumerate(self.plant_features):
            ax = axes[idx//6, idx%6]
            self.plants_df[col].hist(bins=30, ax=ax, edgecolor="black", alpha=0.7, color='steelblue')
            ax.set_title(col, fontweight='bold')
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
        
        plt.tight_layout()
        plt.savefig("Data/plots/plant_features_distribution.png", dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()
        
        # ============================================================================
        # 2. DEMAND-PLANT FEATURE CORRELATION
        # ============================================================================
        combined_df = pd.concat([self.demand_df[self.demand_features], self.plants_df[self.plant_features]], axis=1)
        corr_cross = combined_df[self.demand_features + self.plant_features].corr()
        cross_corr_matrix = corr_cross.loc[self.demand_features, self.plant_features]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(cross_corr_matrix, annot=True, fmt=".2f", cmap="coolwarm_r", center=0, ax=ax, cbar_kws={"label": "Correlation"})
        ax.set_title("Demand Features vs Plant Features Correlation", fontweight='bold', fontsize=12)
        plt.tight_layout()
        plt.savefig("Data/plots/demand_plant_feature_correlation.png", dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()
        
        # ============================================================================
        # 3. COST DATA ANALYSIS
        # ============================================================================
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        costs_summary = self.costs_df.groupby("Plant ID")["Cost_USD_per_MWh"].agg(["median", "mean", "count"]).sort_values("median")
        axes[0].bar(range(len(costs_summary)), costs_summary["median"], alpha=0.7, edgecolor="black")
        axes[0].set_title("Median Cost by Plant (Sorted)", fontweight="bold")
        axes[0].set_xlabel("Plant Index")
        axes[0].set_ylabel("Cost (USD/MWh)")
        axes[0].grid(axis="y", alpha=0.3)
        
        axes[1].hist(self.costs_df["Cost_USD_per_MWh"], bins=50, edgecolor="black", alpha=0.7, color="orange")
        axes[1].set_title("Distribution of Generation Costs", fontweight="bold")
        axes[1].set_xlabel("Cost (USD/MWh)")
        axes[1].set_ylabel("Frequency")
        mean_cost = self.costs_df["Cost_USD_per_MWh"].mean()
        axes[1].axvline(mean_cost, color="red", linestyle="--", label=f"Mean: ${mean_cost:.2f}")
        axes[1].legend()
        axes[1].grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("Data/plots/cost_data_analysis.png", dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()
        
        # ============================================================================
        # 4. PLANT TYPE AND REGION ANALYSIS
        # ============================================================================
        costs_with_plants = self.costs_df.merge(self.plants_df[['Plant ID', 'Plant Type', 'Region']], on='Plant ID', how='left')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Cost distribution by Plant Type
        sns.boxplot(data=costs_with_plants, x='Plant Type', y='Cost_USD_per_MWh', hue='Plant Type', ax=axes[0, 0], palette='Set2', legend=False)
        axes[0, 0].set_title('Cost Distribution by Plant Type', fontweight='bold')
        axes[0, 0].set_ylabel('Cost (USD/MWh)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Cost distribution by Region
        sns.boxplot(data=costs_with_plants, x='Region', y='Cost_USD_per_MWh', hue='Region', ax=axes[0, 1], palette='Set1', legend=False)
        axes[0, 1].set_title('Cost Distribution by Region', fontweight='bold')
        axes[0, 1].set_ylabel('Cost (USD/MWh)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Average cost by Plant Type
        plant_type_avg = costs_with_plants.groupby('Plant Type')['Cost_USD_per_MWh'].mean().sort_values(ascending=False)
        axes[1, 0].bar(range(len(plant_type_avg)), plant_type_avg.values, color='skyblue', edgecolor='black')
        axes[1, 0].set_xticks(range(len(plant_type_avg)))
        axes[1, 0].set_xticklabels(plant_type_avg.index, rotation=45)
        axes[1, 0].set_title('Average Cost by Plant Type', fontweight='bold')
        axes[1, 0].set_ylabel('Average Cost (USD/MWh)')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Average cost by Region
        region_avg = costs_with_plants.groupby('Region')['Cost_USD_per_MWh'].mean().sort_values(ascending=False)
        axes[1, 1].bar(range(len(region_avg)), region_avg.values, color='lightcoral', edgecolor='black')
        axes[1, 1].set_xticks(range(len(region_avg)))
        axes[1, 1].set_xticklabels(region_avg.index, rotation=45)
        axes[1, 1].set_title('Average Cost by Region', fontweight='bold')
        axes[1, 1].set_ylabel('Average Cost (USD/MWh)')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("Data/plots/plant_type_region_analysis.png", dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()
        
        # ============================================================================
        # 5. COST DISTRIBUTION BY PLANT (VIOLIN PLOT)
        # ============================================================================
        fig, ax = plt.subplots(figsize=(16, 6))
        sns.violinplot(data=self.costs_df, x='Plant ID', y='Cost_USD_per_MWh', hue='Plant ID', ax=ax, palette='Set2', legend=False)
        ax.set_title('Cost Distribution by Plant', fontweight='bold', fontsize=12)
        ax.set_ylabel('Cost (USD/MWh)')
        ax.tick_params(axis='x', rotation=90)
        plt.tight_layout()
        plt.savefig("Data/plots/cost_distribution_by_plant.png", dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()
        
        # ============================================================================
        # 6. PLANT TYPE AND REGION INTERACTION HEATMAP
        # ============================================================================
        fig, ax = plt.subplots(figsize=(10, 6))
        pivot_type_region = costs_with_plants.pivot_table(values='Cost_USD_per_MWh', index='Plant Type', columns='Region', aggfunc='mean')
        sns.heatmap(pivot_type_region, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, cbar_kws={'label': 'Avg Cost (USD/MWh)'})
        ax.set_title('Average Cost by Plant Type and Region', fontweight='bold', fontsize=12)
        plt.tight_layout()
        plt.savefig("Data/plots/plant_type_region_heatmap.png", dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()
        
        # ============================================================================
        # 7. ERROR ANALYSIS: PER-SCENARIO ERRORS AND PLANT RMSE
        # ============================================================================
        
        # Calculate optimal (minimum) cost for each demand scenario
        optimal_costs = self.costs_df.groupby('Demand ID')['Cost_USD_per_MWh'].min()
        
        # For each plant, calculate error for each scenario
        plant_errors = {}
        plant_rmse = {}
        plant_mae = {}
        
        for plant_id in self.costs_df['Plant ID'].unique():
            plant_costs = self.costs_df[self.costs_df['Plant ID'] == plant_id].set_index('Demand ID')['Cost_USD_per_MWh'] # type: ignore
            
            # Calculate error: difference between optimal cost and plant's cost
            errors = optimal_costs - plant_costs
            errors = errors.dropna()
            
            if len(errors) > 0:
                plant_errors[plant_id] = errors.values
                plant_rmse[plant_id] = np.sqrt(np.mean(errors.values ** 2))
                plant_mae[plant_id] = np.mean(np.abs(errors.values))
        
        # Create summary dataframe
        plant_performance = pd.DataFrame({
            'Plant ID': list(plant_rmse.keys()),
            'RMSE': list(plant_rmse.values()),
            'MAE': list(plant_mae.values())
        }).sort_values('RMSE')
        
        # Merge with plant info
        plant_performance = plant_performance.merge(
            self.plants_df[['Plant ID', 'Plant Type', 'Region']], 
            on='Plant ID', 
            how='left'
        )
        
        # Store insights
        self.correlation_insights['error_analysis'] = {
            'plant_rmse': plant_rmse,
            'plant_mae': plant_mae,
            'overall_rmse_mean': np.mean(list(plant_rmse.values())),
            'overall_rmse_std': np.std(list(plant_rmse.values())),
            'best_plant_id': plant_performance.iloc[0]['Plant ID'],
            'best_plant_rmse': plant_performance.iloc[0]['RMSE'],
            'worst_plant_id': plant_performance.iloc[-1]['Plant ID'],
            'worst_plant_rmse': plant_performance.iloc[-1]['RMSE']
        }
        
        # Visualization 1: RMSE by Plant (sorted)
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Top plot: RMSE bar chart (all plants)
        axes[0].bar(range(len(plant_performance)), plant_performance['RMSE'].values, alpha=0.7, edgecolor='black', color='coral')
        axes[0].set_title('Root Mean Square Error (RMSE) by Plant', fontweight='bold', fontsize=12)
        axes[0].set_xlabel('Plant Index (sorted by RMSE)')
        axes[0].set_ylabel('RMSE (USD/MWh)')
        axes[0].axhline(np.mean(list(plant_rmse.values())), color='red', linestyle='--', linewidth=2, label=f'Mean RMSE: ${np.mean(list(plant_rmse.values())):.2f}')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Bottom plot: Error distribution (top 15 best performing plants)
        top_15_plants = plant_performance.head(15)['Plant ID'].values
        errors_data = [plant_errors[p] for p in top_15_plants]
        bp = axes[1].boxplot(errors_data, labels=[f"P{p}" for p in top_15_plants], patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightgreen')
        axes[1].set_title('Error Distribution for Top 15 Best-Performing Plants', fontweight='bold', fontsize=12)
        axes[1].set_ylabel('Error (USD/MWh)')
        axes[1].grid(axis='y', alpha=0.3)
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig("Data/plots/error_distribution_top_15_plants.png", dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()
        
        # Visualization 2: Plant Performance Summary Table
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        display_df = plant_performance.head(20)[['Plant ID', 'Plant Type', 'Region', 'RMSE', 'MAE']].copy()
        display_df['RMSE'] = display_df['RMSE'].apply(lambda x: f'${x:.2f}')
        display_df['MAE'] = display_df['MAE'].apply(lambda x: f'${x:.2f}')
        
        # convert DataFrame to list of lists (strings) to satisfy matplotlib.table type requirements
        display_table_data = display_df.astype(str).values.tolist()
        table = ax.table(cellText=display_table_data, colLabels=list(display_df.columns), cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Color header
        for i in range(len(display_df.columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Top 20 Best-Performing Plants by RMSE', fontweight='bold', fontsize=12, pad=20)
        plt.tight_layout()
        plt.savefig("Data/plots/top_20_best_performing_plants.png", dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()
        
        # Visualization 3: RMSE Distribution by Plant Type
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Boxplot: RMSE by Plant Type
        sns.boxplot(data=plant_performance, x='Plant Type', y='RMSE', hue='Plant Type', ax=axes[0], palette='Set2', legend=False)
        axes[0].set_title('RMSE Distribution by Plant Type', fontweight='bold')
        axes[0].set_ylabel('RMSE (USD/MWh)')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Boxplot: RMSE by Region
        sns.boxplot(data=plant_performance, x='Region', y='RMSE', hue='Region', ax=axes[1], palette='Set1', legend=False)
        axes[1].set_title('RMSE Distribution by Region', fontweight='bold')
        axes[1].set_ylabel('RMSE (USD/MWh)')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("Data/plots/rmse_distribution_by_type_region.png", dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()
        
        # Print summary statistics
        print("\n" + "="*80)
        print("ERROR ANALYSIS SUMMARY")
        print("="*80)
        print(f"Total plants analyzed: {len(plant_rmse)}")
        print(f"Mean RMSE across all plants: ${np.mean(list(plant_rmse.values())):.2f}")
        print(f"Std Dev of RMSE: ${np.std(list(plant_rmse.values())):.2f}")
        print(f"Best performing plant: {plant_performance.iloc[0]['Plant ID']} (RMSE: ${plant_performance.iloc[0]['RMSE']:.2f})")
        print(f"Worst performing plant: {plant_performance.iloc[-1]['Plant ID']} (RMSE: ${plant_performance.iloc[-1]['RMSE']:.2f})")
        print(f"RMSE range: ${plant_performance.iloc[-1]['RMSE'] - plant_performance.iloc[0]['RMSE']:.2f}")
        print("="*80 + "\n")
        
        # Store comprehensive results for downstream use
        self.plant_performance_df = plant_performance
        self.plant_errors_dict = plant_errors
        self.optimal_costs = optimal_costs
        self.baseline_rmse_avg = np.mean(list(plant_rmse.values()))
    
    def get_results(self) -> Dict[str, Any]:
        return {
            "correlation_insights": self.correlation_insights,
            "plant_performance": self.plant_performance_df if hasattr(self, 'plant_performance_df') else None,
            "plant_errors": self.plant_errors_dict if hasattr(self, 'plant_errors_dict') else None,
            "optimal_costs": self.optimal_costs if hasattr(self, 'optimal_costs') else None,
            "baseline_rmse_avg": self.baseline_rmse_avg if hasattr(self, 'baseline_rmse_avg') else None,
            "demand_df": self.demand_df,
            "plants_df": self.plants_df,
            "costs_df": self.costs_df,
            "demand_features": self.demand_features,
            "plant_features": self.plant_features
        }
