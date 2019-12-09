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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def train():

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

    # Standardize the data
    train_df = pd.DataFrame(StandardScaler().fit_transform(train_df))

    # Apply PCA
    pca = PCA(n_components=20)
    pca_df = pd.DataFrame(pca.fit_transform(train_df))

    # Make sure that there are columns
    pca_df.columns = ['pca_{index}'.format(index=index) for index in pca_df.columns]

    # Save output to parquet
    output_path = os.path.realpath(os.path.join(local_path, '../data/engineered/superconduct'))
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    output_path = os.path.realpath(os.path.join(local_path, '../data/engineered/superconduct/pca'))
    pca_df.to_parquet(output_path)


if __name__ == '__main__':
    train()
