# Copyright 2026 The android_world Authors.
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

"""Runs the eval suite through the Docker HTTP server.

This script mirrors the high-level flow in run.py, but uses the HTTP Android
environment server (server/android_server.py) instead of direct ADB control.
"""

from collections import defaultdict
from collections.abc import Sequence
import datetime
import json
import time
from typing import Any

from absl import app
from absl import flags
from absl import logging
from android_world import checkpointer as checkpointer_lib
from android_world import constants
from android_world import episode_runner
from android_world import registry
from android_world.agents import base_agent
from android_world.agents import human_agent
from android_world.agents import infer
from android_world.agents import m3a
from android_world.agents import random_agent
from android_world.env import interface
from android_world.env import json_action
from android_world.env import representation_utils
import numpy as np
import pydantic
import requests

logging.set_verbosity(logging.WARNING)


_SERVER_URL = flags.DEFINE_string(
    'server_url',
    'http://localhost:5432',
    'Base URL for the Docker Android environment server.',
)

_SUITE_FAMILY = flags.DEFINE_enum(
    'suite_family',
    registry.TaskRegistry.ANDROID_WORLD_FAMILY,
    [
        registry.TaskRegistry.ANDROID_WORLD_FAMILY,
        registry.TaskRegistry.MINIWOB_FAMILY_SUBSET,
        registry.TaskRegistry.MINIWOB_FAMILY,
        registry.TaskRegistry.ANDROID_FAMILY,
        registry.TaskRegistry.INFORMATION_RETRIEVAL_FAMILY,
    ],
    'Suite family to run. See registry.py for more information.',
)

_TASK_RANDOM_SEED = flags.DEFINE_integer(
    'task_random_seed', 30, 'Random seed for task randomness.'
)

_TASKS = flags.DEFINE_list(
    'tasks',
    None,
    'List of specific tasks to run in the given suite family. If None, run all'
    ' tasks in the suite family.',
)

_N_TASK_COMBINATIONS = flags.DEFINE_integer(
    'n_task_combinations',
    1,
    'Number of task instances to run for each task template.',
)

_CHECKPOINT_DIR = flags.DEFINE_string(
    'checkpoint_dir',
    '',
    'The directory to save checkpoints and resume evaluation from.',
)

_OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    'runs',
    'The path to save results to if a checkpoint directory is not provided.',
)

_AGENT_NAME = flags.DEFINE_string(
    'agent_name',
    'random_agent',
    'Agent name. Supported: random_agent, human_agent, m3a_gemini_gcp, '
    'm3a_gpt4v.',
)

_FIXED_TASK_SEED = flags.DEFINE_boolean(
    'fixed_task_seed',
    False,
    'Whether to use the same task seed for all combinations. Not supported by '
    'the current Docker server API and will be ignored.',
)

_MAX_STEPS = flags.DEFINE_integer(
    'max_steps',
    10,
    'Maximum number of agent steps per task instance.',
)

_START_ON_HOME = flags.DEFINE_boolean(
    'start_on_home_screen',
    False,
    'Whether each episode should start from home screen reset.',
)


# MiniWoB is very lightweight and new screens/View Hierarchy load quickly.
_MINIWOB_TRANSITION_PAUSE = 0.2

# Additional guidelines for the MiniWoB tasks.
_MINIWOB_ADDITIONAL_GUIDELINES = [
    (
        'This task is running in a mock app, you must stay in this app and'
        ' DO NOT use the `navigate_home` action.'
    ),
]


class Response(pydantic.BaseModel):
  status: str
  message: str


class DockerEnvClient:
  """HTTP client for the Docker Android environment server."""

  def __init__(self, base_url: str):
    self.base_url = base_url.rstrip('/')

  def reset(self, go_home: bool) -> Response:
    response = requests.post(
        f'{self.base_url}/reset', params={'go_home': go_home}, timeout=120
    )
    response.raise_for_status()
    return Response(**response.json())

  def _deserialize_ui_element(
      self, ui_element: dict[str, Any]
  ) -> representation_utils.UIElement:
    element_data = dict(ui_element)
    for bbox_key in ('bbox', 'bbox_pixels'):
      bbox = element_data.get(bbox_key)
      if isinstance(bbox, dict):
        element_data[bbox_key] = representation_utils.BoundingBox(**bbox)
    return representation_utils.UIElement(**element_data)

  def get_state(
      self, wait_to_stabilize: bool = False
  ) -> tuple[np.ndarray, list[representation_utils.UIElement]]:
    response = requests.get(
        f'{self.base_url}/screenshot',
        params={'wait_to_stabilize': wait_to_stabilize},
        timeout=120,
    )
    response.raise_for_status()
    state = response.json()
    pixels = np.array(state['pixels'])
    ui_elements = [
        self._deserialize_ui_element(element)
        for element in state.get('ui_elements', [])
        if isinstance(element, dict)
    ]
    return pixels, ui_elements

  def execute_action(self, action: json_action.JSONAction) -> Response:
    response = requests.post(
        f'{self.base_url}/execute_action',
        json=json.loads(action.json_str()),
        timeout=120,
    )
    response.raise_for_status()
    return Response(**response.json())

  def get_suite_task_list(self, max_index: int) -> list[str]:
    response = requests.get(
        f'{self.base_url}/suite/task_list',
        params={'max_index': max_index},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()['task_list']

  def get_suite_task_length(self, task_type: str) -> int:
    response = requests.get(
        f'{self.base_url}/suite/task_length',
        params={'task_type': task_type},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()['length']

  def reinitialize_suite(
      self,
      n_task_combinations: int,
      seed: int,
      task_family: str,
  ) -> Response:
    response = requests.get(
        f'{self.base_url}/suite/reinitialize',
        params={
            'n_task_combinations': n_task_combinations,
            'seed': seed,
            'task_family': task_family,
        },
        timeout=120,
    )
    response.raise_for_status()
    return Response(**response.json())

  def initialize_task(self, task_type: str, task_idx: int) -> Response:
    response = requests.post(
        f'{self.base_url}/task/initialize',
        params={'task_type': task_type, 'task_idx': task_idx},
        timeout=120,
    )
    response.raise_for_status()
    return Response(**response.json())

  def tear_down_task(self, task_type: str, task_idx: int) -> Response:
    response = requests.post(
        f'{self.base_url}/task/tear_down',
        params={'task_type': task_type, 'task_idx': task_idx},
        timeout=120,
    )
    response.raise_for_status()
    return Response(**response.json())

  def get_task_score(self, task_type: str, task_idx: int) -> float:
    response = requests.get(
        f'{self.base_url}/task/score',
        params={'task_type': task_type, 'task_idx': task_idx},
        timeout=120,
    )
    response.raise_for_status()
    return float(response.json()['score'])

  def get_task_goal(self, task_type: str, task_idx: int) -> str:
    response = requests.get(
        f'{self.base_url}/task/goal',
        params={'task_type': task_type, 'task_idx': task_idx},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()['goal']

  def health(self) -> bool:
    try:
      response = requests.get(f'{self.base_url}/health', timeout=10)
      response.raise_for_status()
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f'Environment is not healthy: {e}')
      return False
    return True


class DockerAsyncEnv(interface.AsyncEnv):
  """Lightweight AsyncEnv adapter backed by Docker HTTP endpoints."""

  def __init__(self, client: DockerEnvClient):
    self._client = client
    self._logical_screen_size = (1080, 2400)
    self._interaction_cache = ''

  @property
  def controller(self) -> Any:
    # Docker client does not expose direct controller access.
    return None

  def reset(self, go_home: bool = False) -> interface.State:
    self._interaction_cache = ''
    self._client.reset(go_home=go_home)
    return self.get_state(wait_to_stabilize=False)

  def get_state(self, wait_to_stabilize: bool = False) -> interface.State:
    pixels, ui_elements = self._client.get_state(
        wait_to_stabilize=wait_to_stabilize
    )
    if pixels.ndim >= 2:
      height, width = pixels.shape[0], pixels.shape[1]
      self._logical_screen_size = (int(width), int(height))
    return interface.State(
        pixels=pixels,
        forest=None,
        ui_elements=ui_elements,
        auxiliaries={},
    )

  def ask_question(
      self, question: str, timeout_seconds: float = -1.0
  ) -> str | None:
    del timeout_seconds
    return input(f'{question}\n> ')

  def execute_action(self, action: json_action.JSONAction) -> None:
    if action.action_type == json_action.ANSWER:
      self._interaction_cache = action.text or ''
      return
    if action.action_type == json_action.STATUS:
      return
    self._client.execute_action(action)

  @property
  def foreground_activity_name(self) -> str:
    return ''

  @property
  def device_screen_size(self) -> tuple[int, int]:
    return self._logical_screen_size

  @property
  def logical_screen_size(self) -> tuple[int, int]:
    return self._logical_screen_size

  def close(self) -> None:
    # Keep server alive for subsequent runs.
    return

  @property
  def interaction_cache(self) -> str:
    return self._interaction_cache

  def hide_automation_ui(self) -> None:
    # No direct equivalent over HTTP.
    return

  @property
  def orientation(self) -> int:
    width, height = self._logical_screen_size
    return 1 if width > height else 0

  @property
  def physical_frame_boundary(self) -> tuple[int, int, int, int]:
    width, height = self._logical_screen_size
    return (0, 0, width, height)


def _get_agent(
    env: interface.AsyncEnv,
    family: str | None = None,
) -> base_agent.EnvironmentInteractingAgent:
  """Gets agent for Docker mode."""
  print('Initializing agent...')
  agent = None
  import os
  if _AGENT_NAME.value == 'human_agent':
    agent = human_agent.HumanAgent(env)
  elif _AGENT_NAME.value == 'random_agent':
    agent = random_agent.RandomAgent(env)
  elif _AGENT_NAME.value == 'm3a_gemini_gcp':
    agent = m3a.M3A(
        env, infer.GeminiGcpWrapper(model_name='gemini-1.5-pro-latest')
    )
  elif _AGENT_NAME.value == 'm3a_gpt4v':
    os.environ['OPENAI_BASE_URL'] = os.environ.get('OPENAI_BASE_URL', "https://api.openai.com/v1")
    agent = m3a.M3A(env, infer.Gpt4Wrapper('gpt-4-turbo-2024-04-09'))
  elif _AGENT_NAME.value == 'm3a_qwen3vl':
    os.environ['OPENAI_BASE_URL'] = os.environ.get('OPENAI_BASE_URL', "http://localhost:8000/v1")
    agent = m3a.M3A(env, infer.Gpt4Wrapper('Qwen/Qwen3-VL-4B-Instruct'))
  elif _AGENT_NAME.value in ('t3a_gemini_gcp', 't3a_gpt4', 'seeact'):
    raise ValueError(
        f'Agent {_AGENT_NAME.value} is not supported in Docker mode because '
        'the HTTP server does not expose full controller/UI-tree interfaces '
        'required by that agent.'
    )

  if not agent:
    raise ValueError(f'Unknown agent: {_AGENT_NAME.value}')

  if (
      agent.name in ['M3A']
      and family
      and family.startswith('miniwob')
      and hasattr(agent, 'set_task_guidelines')
  ):
    agent.set_task_guidelines(_MINIWOB_ADDITIONAL_GUIDELINES)
  agent.name = _AGENT_NAME.value

  return agent


def _load_checkpoint_lookup(
    checkpointer: checkpointer_lib.Checkpointer,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
  fields = [
      constants.EpisodeConstants.TASK_TEMPLATE,
      constants.EpisodeConstants.INSTANCE_ID,
      constants.EpisodeConstants.EXCEPTION_INFO,
  ]
  episodes = checkpointer.load(fields=fields)
  completed = defaultdict(list)
  failed = defaultdict(list)
  for episode in episodes:
    instance_name = (
        episode[constants.EpisodeConstants.TASK_TEMPLATE]
        + checkpointer_lib.INSTANCE_SEPARATOR
        + str(episode[constants.EpisodeConstants.INSTANCE_ID])
    )
    if episode.get(constants.EpisodeConstants.EXCEPTION_INFO) is not None:
      failed[instance_name].append(episode)
    else:
      completed[instance_name].append(episode)
  return dict(completed), dict(failed)


def _create_failed_result(
    task_name: str,
    goal: str,
    instance_id: int,
    exception: str,
    run_time: float,
) -> dict[str, Any]:
  return {
      constants.EpisodeConstants.GOAL: goal,
      constants.EpisodeConstants.TASK_TEMPLATE: task_name,
      constants.EpisodeConstants.INSTANCE_ID: instance_id,
      constants.EpisodeConstants.IS_SUCCESSFUL: np.nan,
      constants.EpisodeConstants.EPISODE_DATA: np.nan,
      constants.EpisodeConstants.EPISODE_LENGTH: np.nan,
      constants.EpisodeConstants.RUN_TIME: run_time,
      constants.EpisodeConstants.EXCEPTION_INFO: exception,
      constants.EpisodeConstants.AUX_DATA: None,
      constants.EpisodeConstants.AGENT_NAME: _AGENT_NAME.value,
        constants.EpisodeConstants.FINISH_DTIME: datetime.datetime.now(),
      constants.EpisodeConstants.SCREEN_CONFIG: {
          'width': 1080,
          'height': 2400,
          'orientation': 'portrait',
          'config_name': 'docker',
      },
      constants.EpisodeConstants.SEED: _TASK_RANDOM_SEED.value,
  }


def _main() -> None:
  if _FIXED_TASK_SEED.value:
    print(
        '--fixed_task_seed is currently ignored in Docker mode because the '
        'server API does not expose this setting.'
    )

  client = DockerEnvClient(_SERVER_URL.value)
  while not client.health():
    print('Environment is not healthy, waiting for 1 second...')
    time.sleep(1)

  env = DockerAsyncEnv(client)
  agent = _get_agent(env, _SUITE_FAMILY.value)

  if _SUITE_FAMILY.value.startswith('miniwob'):
    agent.transition_pause = _MINIWOB_TRANSITION_PAUSE
  else:
    agent.transition_pause = None

  if _CHECKPOINT_DIR.value:
    checkpoint_dir = _CHECKPOINT_DIR.value
  else:
    checkpoint_dir = checkpointer_lib.create_run_directory(_OUTPUT_PATH.value)
  checkpointer = checkpointer_lib.IncrementalCheckpointer(checkpoint_dir)

  client.reinitialize_suite(
      n_task_combinations=_N_TASK_COMBINATIONS.value,
      seed=_TASK_RANDOM_SEED.value,
      task_family=_SUITE_FAMILY.value,
  )

  task_list = client.get_suite_task_list(max_index=-1)
  if _TASKS.value:
    unknown = [task for task in _TASKS.value if task not in task_list]
    if unknown:
      raise ValueError(
          f'Unknown task(s) for suite {_SUITE_FAMILY.value}: {unknown}'
      )
    task_list = [task for task in task_list if task in _TASKS.value]

  completed_tasks, failed_tasks = _load_checkpoint_lookup(checkpointer)
  total, correct = 0, 0.0

  print(
      f'Starting Docker eval with agent {_AGENT_NAME.value} and writing to '
      f'{checkpoint_dir}'
  )

  for task_name in task_list:
    print(f'Running task: {task_name}')
    num_tasks = client.get_suite_task_length(task_type=task_name)

    for task_idx in range(num_tasks):
      instance_name = task_name + checkpointer_lib.INSTANCE_SEPARATOR + str(
          task_idx
      )
      if instance_name in completed_tasks and instance_name not in failed_tasks:
        print(f'Skipping already processed task {instance_name}')
        continue

      start = time.time()
      goal = client.get_task_goal(task_type=task_name, task_idx=task_idx)
      try:
        client.initialize_task(task_type=task_name, task_idx=task_idx)
        interaction_results = episode_runner.run_episode(
            goal=goal,
            agent=agent,
            max_n_steps=_MAX_STEPS.value,
            start_on_home_screen=_START_ON_HOME.value,
        )
        task_score = client.get_task_score(task_type=task_name, task_idx=task_idx)
        agent_successful = task_score if interaction_results.done else 0.0
        episode = {
            constants.EpisodeConstants.GOAL: goal,
            constants.EpisodeConstants.TASK_TEMPLATE: task_name,
            constants.EpisodeConstants.INSTANCE_ID: task_idx,
            constants.EpisodeConstants.IS_SUCCESSFUL: agent_successful,
            constants.EpisodeConstants.EPISODE_DATA: interaction_results.step_data,
            constants.EpisodeConstants.EPISODE_LENGTH: len(
                interaction_results.step_data.get(constants.STEP_NUMBER, [])
            ),
            constants.EpisodeConstants.RUN_TIME: time.time() - start,
            constants.EpisodeConstants.EXCEPTION_INFO: None,
            constants.EpisodeConstants.AUX_DATA: interaction_results.aux_data,
            constants.EpisodeConstants.AGENT_NAME: _AGENT_NAME.value,
            constants.EpisodeConstants.FINISH_DTIME: datetime.datetime.now(),
            constants.EpisodeConstants.SCREEN_CONFIG: {
                'width': env.logical_screen_size[0],
                'height': env.logical_screen_size[1],
                'orientation': 'landscape'
                if env.orientation == 1
                else 'portrait',
                'config_name': 'docker',
            },
            constants.EpisodeConstants.SEED: _TASK_RANDOM_SEED.value,
        }
        client.tear_down_task(task_type=task_name, task_idx=task_idx)
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception('Skipping failed task %s_%d: %s', task_name, task_idx, e)
        episode = _create_failed_result(
            task_name=task_name,
            goal=goal,
            instance_id=task_idx,
            exception=str(e),
            run_time=time.time() - start,
        )
      finally:
        try:
          client.reset(go_home=True)
        except Exception as e:  # pylint: disable=broad-exception-caught
          logging.warning('Reset failed after task %s_%d: %s', task_name, task_idx, e)

      checkpointer.save_episodes([episode], instance_name)

      if episode[constants.EpisodeConstants.EXCEPTION_INFO] is None:
        total += 1
        correct += float(episode[constants.EpisodeConstants.IS_SUCCESSFUL])

      print(
          f'[{task_name} #{task_idx}] success={episode[constants.EpisodeConstants.IS_SUCCESSFUL]} '
          f'completed={int(correct)}/{total}'
      )

  print(
      f'Finished running agent {_AGENT_NAME.value} on {_SUITE_FAMILY.value} '
      f'family. Wrote to {checkpoint_dir}.'
  )


def main(argv: Sequence[str]) -> None:
  del argv
  _main()


if __name__ == '__main__':
  app.run(main)
