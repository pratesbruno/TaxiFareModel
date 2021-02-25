# imports
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        dist_pipeline = Pipeline(steps=[('dist_transformer', DistanceTransformer()),
                            ('standarizer', StandardScaler())])

        time_pipeline = Pipeline(steps=[('time_transformer', TimeFeaturesEncoder('pickup_datetime')),
                           ('ohe', OneHotEncoder())])
        dist_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        time_cols = ['pickup_datetime']
        prep_pipeline = ColumnTransformer([('dist_pipeline', dist_pipeline, dist_cols),
                                        ('time_pipeline', time_pipeline, time_cols)])
        
        self.pipeline = Pipeline(steps=[('prep_pipeline', prep_pipeline),
                                ('regressor', LinearRegression())])
                                        
                                
        #return self.pipeline

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        return compute_rmse(y_test, y_pred)

if __name__ == "__main__":
    # get data
    df = get_data()
    # clean data
    clean_df = clean_data(df)
    # set X and y
    X = clean_df.drop(columns=['fare_amount'])
    y = clean_df['fare_amount']
    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    # train
    trainer = Trainer(X_train, y_train)
    trainer.set_pipeline()
    trainer.run()
    trainer.evaluate(X_val,y_val)
    # evaluate
    print(trainer.evaluate(X_val,y_val))
