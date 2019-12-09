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

from sklearn.model_selection import train_test_split


@task
def train(**kwargs):

    # set the data paths
    local_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.realpath(os.path.join(local_path, '../data/selected/superconduct/lasso'))

    # Load the data
    data = pd.read_parquet(data_path)

    # Split the data into a train & test as well as a holdout (to emulate usage)
    data_train_test, data_holdout = train_test_split(data, test_size=0.01, random_state=42)

    # Persist data to parquet
    output_path = os.path.realpath(os.path.join(local_path, '../data/split/superconduct'))
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    output_path_train = os.path.realpath(os.path.join(local_path, '../data/split/superconduct/train_test'))
    output_path_holdout = os.path.realpath(os.path.join(local_path, '../data/split/superconduct/holdout'))
    pd.DataFrame(data_train_test).to_parquet(output_path_train)
    pd.DataFrame(data_holdout).to_parquet(output_path_holdout)
