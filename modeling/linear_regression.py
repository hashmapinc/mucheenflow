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

import pandas as pd
from prefect import task
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
import dill as pickle


@task
def train(**kwargs):

    # set data paths
    local_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.realpath(os.path.join(local_path, '../data/split/superconduct/train_test'))

    # Load the data
    data = pd.read_parquet(data_path)
    target = data.loc[:, ['critical_temp']]
    data = data.loc[:, data.columns[:-1]]

    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.1, random_state=42)

    reg = LinearRegression().fit(data_train, target_train)

    reg.score(data_train, target_train)

    print('regression coefficients: ', reg.coef_)

    print('regression intercept: ', reg.intercept_)

    print('MSE: ', mean_squared_error(reg.predict(data_test), target_test.values))

    # Persist the model
    local_path = os.path.dirname(os.path.abspath(__file__))
    pickle_path = os.path.realpath(os.path.join(local_path, '../artifacts/modeling'))
    if not os.path.exists(pickle_path):
        os.makedirs(pickle_path, exist_ok=True)
    pickle_path = os.path.realpath(os.path.join(local_path, '../artifacts/modeling/linear_regression.pkl'))

    with open(pickle_path, 'wb') as pickle_file:
        pickle.dump(reg, pickle_file)
