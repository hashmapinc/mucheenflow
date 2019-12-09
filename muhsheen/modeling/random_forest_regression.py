# Modifications © 2019 Hashmap, Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from prefect import task

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split


@task
def train(**kwargs):

    # set data paths
    local_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.realpath(os.path.join(local_path, '../data/split/superconduct/train_test'))

    # Load the data
    data = pd.read_parquet(data_path)
    target = data.loc[:, ['critical_temp']]
    data = data.loc[:, data.columns[:-1]]

    with mlflow.start_run(experiment_id=0):
        split_size = 0.1
        random_state_train_test = 42
        data_train, data_test, target_train, target_test = train_test_split(data,
                                                                            target,
                                                                            test_size=split_size,
                                                                            random_state=random_state_train_test)

        n_estimators = 10
        random_state_rf = 42
        reg = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state_rf)\
            .fit(data_train, target_train.values.ravel())

        score = reg.score(data_train, target_train)

        mse = mean_squared_error(reg.predict(data_test), target_test.values.ravel())

        mlflow.log_metrics(
            {
                'score': score,
                'MSE': mse
            }
        )

        mlflow.log_params(
            {
                'train_test_split_size': split_size,
                'train_test_split_random_state': random_state_train_test,
                'rf_n_estimators': n_estimators,
                'random_state': random_state_rf
            }
        )

        mlflow.sklearn.log_model(reg, 'rf_model')

        mlflow.set_tags(
            {
                'model': 'random_forest_regression'
            }
        )