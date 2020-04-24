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
from importlib import import_module

import mlflow
import yaml
import traceback
from prefect import Flow


def dependencies_pacified(dependencies, pacified):
    return len([1 for _dependency in dependencies if _dependency in pacified]) == len(dependencies)


def add_task(pipe, task_registry):

    arguments = {}
    if pipe['dependent_on']:
        for dependency in pipe['dependent_on']:
            arguments.update({dependency: task_registry[dependency]})

    task_registry[pipe['name']] = pipe['task'](**arguments)


def main():

    # Load configuration
    with open("flow.yml", 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError:
            traceback.format_exc()

    # Create workflows objects
    if config.get('version') == 1:
        workflows = config['workflows']

        pipes = []
        # The workflows are collection of logically collocated pipes - must iterate over the collection.
        for item in workflows:
            pipeline = item['pipeline']
            directory = pipeline['name']

            # Each logical separation will be processed and put into a pipe
            for stage in pipeline['stages']:

                parent_stage = stage['stage'].get('depends_on')
                if parent_stage and not isinstance(parent_stage, list):
                    parent_stage = [parent_stage]
                stage = stage['stage']
                file = stage['file']
                method = stage['method']
                name = stage['name']

                # Create the pipe from the configuration. Will add the task as it is dynamically created
                new_pipe = {
                        'name': name,
                        'dependent_on': parent_stage,
                        'task': getattr(import_module(directory + '.' + file.split('.')[0]), method),
                    }
                pipes.append(
                    new_pipe
                )

        # Build the flow dynamically from the pipes collection.
        pacified_tasks = []
        task_registry = {}
        with Flow('ETL') as flow:
            while len(pipes) > 0:

                for pipe in pipes:
                    if not pipe['dependent_on'] or dependencies_pacified(pipe['dependent_on'], pacified_tasks):
                        add_task(pipe, task_registry)
                        pipes.remove(pipe)
                        pacified_tasks.append(pipe['name'])

        try:
            mlflow.create_experiment('test')
        except Exception:
            pass

        # Execute the pipeline
        flow.run()


if __name__ == '__main__':

    import datetime
    start = datetime.datetime.now()
    main()
    stop = datetime.datetime.now()

    print(stop-start)
