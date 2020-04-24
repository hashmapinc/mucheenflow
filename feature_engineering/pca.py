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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from dask.distributed import Client
import joblib


@task
def train(**kwargs):

    # data path
    local_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.realpath(os.path.join(local_path, '../data/raw/superconduct/train.csv'))

    # Load the data
    data = pd.read_csv(data_path)

    # extract the columns
    features = data.columns[:-1]
    target = ['critical_temp']

    # Split the datasets
    train_df = data.loc[:, features]
    _ = data.loc[:, target]

    # remove old data - garbage collection
    del data

    with mlflow.start_run(experiment_id=1):
        client = Client()
        with joblib.parallel_backend('dask'):
            # Standardize the data
            scaler = StandardScaler()
            train_df = pd.DataFrame(scaler.fit_transform(train_df))

            # Apply PCA
            num_pca_components = 20
            pca = PCA(n_components=num_pca_components)
            pca_df = pd.DataFrame(pca.fit_transform(train_df))

            # Make sure that there are columns
            pca_df.columns = ['pca_{index}'.format(index=index) for index in pca_df.columns]

            # Save output to parquet
            output_path = os.path.realpath(os.path.join(local_path, '../data/engineered/superconduct'))
            if not os.path.exists(output_path):
                os.makedirs(output_path, exist_ok=True)
            output_path = os.path.realpath(os.path.join(local_path, '../data/engineered/superconduct/pca'))
            pca_df.to_parquet(output_path)

            # log scaler, pca, columns and num_pca_components
            mlflow.log_params(
                {
                    'num_pca_components': num_pca_components
                }
            )
            mlflow.sklearn.log_model(scaler, 'scaler_pre_pca')
            mlflow.sklearn.log_model(pca, 'pca')
            mlflow.log_artifact(output_path, 'pca')
            mlflow.set_tags(
                {
                    'model': 'pca'
                }
            )