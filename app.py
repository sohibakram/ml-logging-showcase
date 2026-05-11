
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging

logging.info("Application started")

ingestion = DataIngestion()
df = ingestion.initiate_data_ingestion()

transformation = DataTransformation()
X_train, X_test, y_train, y_test = transformation.initiate_data_transformation(df)

trainer = ModelTrainer()
score = trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)

logging.info(f"Final model score: {score}")

logging.info("Application completed")
