import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner

import pickle
import datetime 
import os

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, logger,  train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical, logger):

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr, logger):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return



def get_paths(date=None):
    print("----- " , date)
    if date is None:
        dd = datetime.date.today()
    else:
        dd = datetime.date.fromisoformat(date)

    format = '%Y-%m'
    month_ago = dd - datetime.timedelta(days=30)
    val = month_ago.strftime(format)
    train = (month_ago - datetime.timedelta(days=30)).strftime(format)

    train_path: str = train
    val_path: str = val
    return {'train': train_path, 'val': val_path}




@flow(task_runner=SequentialTaskRunner())
def main(date="2021-08-15"):

    categorical = ['PUlocationID', 'DOlocationID']

    logger = get_run_logger()

    train = get_paths(date)['train']
    print(f"-- TRAIN DATA HERE -- {train}")
    val = get_paths(date)['val']
    train_path: str = f'./data/fhv_tripdata_{train}.parquet'
    val_path: str = f'./data/fhv_tripdata_{val}.parquet'


    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical, logger)
    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, logger, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical, logger).result()

    run_model(df_val_processed, categorical, dv, lr, logger)

    # save the model
    path = f"dv-{train}.b"
    print(f"+++++++ Saving to: {path}")
    with open(path, "wb") as f_out:
        pickle.dump(dv, f_out)
        logger.info(f"Dv file size is: {os.path.getsize(path)}")


main()
