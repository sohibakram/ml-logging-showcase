
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.logger import logging
from src.exception import CustomException
import sys

class DataTransformation:

    def initiate_data_transformation(self, df):

        try:

            logging.info("Data transformation started")

            encoder = LabelEncoder()

            for col in df.columns:

                if df[col].dtype == 'object':

                    df[col] = encoder.fit_transform(df[col])

            logging.info("Categorical encoding completed")

            X = df.drop(columns=['math score'], axis=1)

            y = df['math score']

            logging.info("Train test split started")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            logging.info("Data transformation completed")

            return X_train, X_test, y_train, y_test

        except Exception as e:

            logging.error("Error occurred during data transformation")

            raise CustomException(e, sys)
