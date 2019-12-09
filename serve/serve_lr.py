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


@task
def serve(**kwargs):

    mlflow.set_experiment(experiment_name='test')
    run_id = mlflow.search_runs(filter_string="tags.model = 'lr_model'")['run_id'][0]
    model = mlflow.sklearn.load_model('runs:/{run_id}/lr_model'.format(run_id=run_id))

    local_path = os.path.dirname(os.path.abspath(__file__))
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
    output_path = os.path.realpath(os.path.join(local_path, '../data/serve/superconduct/lr_predictions.csv'))
    test_data.to_csv(output_path)
