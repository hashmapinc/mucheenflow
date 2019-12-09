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

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso


@task
def train(**kwargs):

    # set the data paths
    local_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.realpath(os.path.join(local_path, '../data/engineered/superconduct/interactions'))

    # Load the raw data
    data_engineered = pd.read_parquet(data_path)
    target = data_engineered.loc[:, ['critical_temp']]
    data_engineered = data_engineered.loc[:, data_engineered.columns[:-1]]

    # Retain the columns, these are needed to be added back in after the model has been trained,
    # and the features have been selected
    features = data_engineered.columns

    # Train the model selector
    clf = Lasso(alpha=.35, precompute=True, max_iter=50)
    model = SelectFromModel(clf, threshold=0.25)
    model.fit(data_engineered, target['critical_temp'].values)

    # Apply the model selector
    data_engineered = pd.DataFrame(
        model.transform(data_engineered)
    )
    data_engineered.columns = [feature for support, feature in zip(model.get_support(), features) if support]

    # Append target column back onto dataset
    data_engineered['critical_temp'] = target

    # Write the results to parquet
    output_path = os.path.realpath(os.path.join(local_path, '../data/selected/superconduct'))
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    output_path = os.path.realpath(os.path.join(local_path, '../data/selected/superconduct/lasso'))
    data_engineered.to_parquet(output_path)
