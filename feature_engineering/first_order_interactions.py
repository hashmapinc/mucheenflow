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

from sklearn import preprocessing


@task
def train(**kwargs):

    # data paths
    local_path = os.path.dirname(os.path.abspath(__file__))
    data_raw_path = os.path.realpath(os.path.join(local_path, '../data/raw/superconduct/train.csv'))
    data_pca_path = os.path.realpath(os.path.join(local_path, '../data/engineered/superconduct/pca'))

    # Load the data
    data_raw = pd.read_csv(data_raw_path)
    target = data_raw.loc[:, ['critical_temp']]
    data_raw = data_raw.loc[:, data_raw.columns[:-1]]

    data_pca = pd.read_parquet(data_pca_path)

    # Join the datasets (using default index)
    data = data_pca.join(data_raw)
    features = data.columns

    # Scale the data
    data = pd.DataFrame(preprocessing.StandardScaler().fit_transform(data))
    data.columns = features

    # Clean up the data
    del data_raw
    del data_pca

    columns = ['{left}_times_{right}'.format(left=left, right=right) for left in data.columns for right in data.columns]
    interaction_df = pd.DataFrame(
        [
            data[left] * data[right] for left in data.columns for right in data.columns
        ],
        columns
    ).T

    # Clean up data
    del data

    # Add the target column back in
    interaction_df['critical_temp'] = target

    # Save output to parquet
    output_path = os.path.realpath(os.path.join(local_path, '../data/engineered/superconduct'))
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    output_path = os.path.realpath(os.path.join(local_path, '../data/engineered/superconduct/interactions'))
    interaction_df.to_parquet(output_path)
