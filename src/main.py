from classes.Data_preparation import DataPreparation
from classes.Exploratory_analysis import ExploratoryAnalysis
from utils.Helper import Load_checkpoint, Save_checkpoint, StepConfig
from utils.logging_config import get_logger
from pyparsing import Any


CHECKPOINT_DIR = "Data/checkpoints/"
logger = get_logger("main")


def main():
    logger.info("Starting main process...")
    step_one_data = Load_checkpoint(CHECKPOINT_DIR, "step_one", logger)
    if step_one_data is None:
        logger.info("Running Step One...")
        step_one = DataPreparation(StepConfig("Step 1", "Data Preparation"))
        step_one.execute(
            demand_path="Data/raw/demand.csv",
            plants_path="Data/raw/plants.csv",
            costs_path="Data/raw/generation_costs.csv"
        )
        step_one_data = step_one.get_results()
        Save_checkpoint(CHECKPOINT_DIR, "step_one", logger, step_one_data)
    
    step_two_data = Load_checkpoint(CHECKPOINT_DIR, "step_two", logger)
    if step_two_data is None:
        logger.info("Running Step Two...")
        step_two = ExploratoryAnalysis(StepConfig("Step 2", "Exploratory Analysis"))
        step_two.execute(step_one_data)
        step_two_data = step_two.get_results()
        Save_checkpoint(CHECKPOINT_DIR, "step_two", logger, step_two_data)
    
    logger.info("Main process completed.")



if __name__ == "__main__":
    main()