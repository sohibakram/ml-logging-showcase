
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from src.logger import logging
from src.exception import CustomException
import sys

class ModelTrainer:

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):

        try:

            logging.info("Model training started")

            model = LinearRegression()

            model.fit(X_train, y_train)

            logging.info("Model trained successfully")

            predictions = model.predict(X_test)

            score = r2_score(y_test, predictions)

            logging.info(f"Model R2 Score: {score}")

            logging.info("Model training completed")

            return score

        except Exception as e:

            logging.error("Error occurred during model training")

            raise CustomException(e, sys)
