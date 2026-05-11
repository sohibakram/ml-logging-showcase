
import pandas as pd
from src.logger import logging
from src.exception import CustomException
import sys

class DataIngestion:

    def initiate_data_ingestion(self):

        logging.info("Data ingestion started")

        try:

            df = pd.read_csv("data/stud.csv")

            logging.info(f"Dataset loaded successfully with shape {df.shape}")

            logging.info("Data ingestion completed")

            return df

        except Exception as e:

            logging.error("Error occurred during data ingestion")

            raise CustomException(e, sys)
