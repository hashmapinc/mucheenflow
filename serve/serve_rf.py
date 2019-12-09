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

import dill as pickle
import pandas as pd
from prefect import task
from sklearn.ensemble import RandomForestRegressor


@task
def serve(**kwargs):

    # Retrieve model
    local_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.realpath(os.path.join(local_path, '../artifacts/modeling/random_forest_regression.pkl'))

    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
        if not isinstance(model, RandomForestRegressor):
            raise TypeError

    testing_data_path = os.path.realpath(os.path.join(local_path, '../data/split/superconduct/holdout'))
    test_data = pd.read_parquet(testing_data_path)
    target = test_data.loc[:, ['critical_temp']]
    test_data = test_data.loc[:, test_data.columns[:-1]]

    predictions = model.predict(test_data)

    test_data['expected_critical_temp'] = pd.DataFrame(target)
    test_data['predicted_critical_temp'] = predictions

    output_path = os.path.realpath(os.path.join(local_path, '../data/serve/superconduct'))
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    output_path = os.path.realpath(os.path.join(local_path, '../data/serve/superconduct/rf_predictions.csv'))

    test_data.to_csv(output_path)
