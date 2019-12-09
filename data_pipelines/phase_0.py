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

# Please reference the following: http://archive.ics.uci.edu/ml/datasets/Superconductivty+Data#

import os
import zipfile

import pycurl


def run():

    # Set the configuration information
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00464/superconduct.zip'

    local_path = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.realpath(os.path.join(local_path, '../data/raw/superconduct'))

    filename = os.path.join(output_path, 'superconduct.zip')

    # Create the output path
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # Download the zip file
    with open(filename, "wb") as file_handle:
        curl = pycurl.Curl()
        curl.setopt(pycurl.URL, url)
        curl.setopt(pycurl.WRITEDATA, file_handle)
        curl.perform()
        curl.close()

    # Unzip the data
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(output_path)

    # Remove the downloaded data (zip)
    os.remove(filename)


if __name__ == '__main__':
    run()
