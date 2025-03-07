# Copyright 2024 the LlamaFactory team.
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
import random
import subprocess
import sys
from enum import Enum, unique

from src.llamafactory import launcher
from src.llamafactory.api.app import run_api
from src.llamafactory.chat.chat_model import run_chat
from src.llamafactory.eval.evaluator import run_eval
from src.llamafactory.extras.env import VERSION, print_env
from src.llamafactory.extras.logging import get_logger
from src.llamafactory.extras.misc import get_device_count
from src.llamafactory.train.tuner import export_model, run_exp
from src.llamafactory.webui.interface import run_web_demo, run_web_ui


USAGE = (
    "-" * 70
    + "\n"
    + "| Usage:                                                             |\n"
    + "|   llamafactory-cli api -h: launch an OpenAI-style API server       |\n"
    + "|   llamafactory-cli chat -h: launch a chat interface in CLI         |\n"
    + "|   llamafactory-cli eval -h: evaluate models                        |\n"
    + "|   llamafactory-cli export -h: merge LoRA adapters and export model |\n"
    + "|   llamafactory-cli train -h: train models                          |\n"
    + "|   llamafactory-cli webchat -h: launch a chat interface in Web UI   |\n"
    + "|   llamafactory-cli webui: launch LlamaBoard                        |\n"
    + "|   llamafactory-cli version: show version info                      |\n"
    + "-" * 70
)

WELCOME = (
    "-" * 58
    + "\n"
    + "| Welcome to LLaMA Factory, version {}".format(VERSION)
    + " " * (21 - len(VERSION))
    + "|\n|"
    + " " * 56
    + "|\n"
    + "| Project page: https://github.com/hiyouga/LLaMA-Factory |\n"
    + "-" * 58
)

logger = get_logger(__name__)


@unique
class Command(str, Enum):
    API = "api"
    CHAT = "chat"
    ENV = "env"
    EVAL = "eval"
    EXPORT = "export"
    TRAIN = "train"
    WEBDEMO = "webchat"
    WEBUI = "webui"
    VER = "version"
    HELP = "help"


def main(command, yml):
    sys.argv.append(yml)

    if command == Command.API:
        run_api()
    elif command == Command.CHAT:
        run_chat()
    elif command == Command.ENV:
        print_env()
    elif command == Command.EVAL:
        run_eval()
    elif command == Command.EXPORT:
        export_model()
    elif command == Command.TRAIN:
        force_torchrun = os.environ.get("FORCE_TORCHRUN", "0").lower() in ["true", "1"]
        if force_torchrun or get_device_count() > 1:
            master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
            master_port = os.environ.get("MASTER_PORT", str(random.randint(20001, 29999)))
            logger.info("Initializing distributed tasks at: {}:{}".format(master_addr, master_port))
            subprocess.run(
                (
                    "torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                    "--master_addr {master_addr} --master_port {master_port} {file_name} {args}"
                ).format(
                    nnodes=os.environ.get("NNODES", "1"),
                    node_rank=os.environ.get("RANK", "0"),
                    nproc_per_node=os.environ.get("NPROC_PER_NODE", str(get_device_count())),
                    master_addr=master_addr,
                    master_port=master_port,
                    file_name=launcher.__file__,
                    args=" ".join(sys.argv[1:]),
                ),
                shell=True,
            )
        else:
            run_exp()
    elif command == Command.WEBDEMO:
        run_web_demo()
    elif command == Command.WEBUI:
        run_web_ui()
    elif command == Command.VER:
        print(WELCOME)
    elif command == Command.HELP:
        print(USAGE)
    else:
        raise NotImplementedError("Unknown command: {}".format(command))

if __name__ == '__main__':
    # command, yml = "train", "examples/rec/glm3/lora_pt_ml.yaml"
    # command, yml = "train", "examples/rec/glm3/lora_sft_ml.yaml"
    # command, yml = "train", "examples/rec/glm3/lora_pd_ml.yaml"
    # command, yml = "train", "examples/rec/glm3/lora_sft_dialogpedia.yaml"

    # command, yml = "train", "examples/rec/gemma2/lora_sft_ml.yaml"
    # command, yml = "train", "examples/rec/gemma2/lora_pd_ml.yaml"

    # command, yml = "train", "examples/rec/gemma2/lora_pt_yelp.yaml"
    # command, yml = "train", "examples/rec/gemma2/lora_sft_ml.yaml"
    command, yml = "train", "examples/rec/llama3/lora_predict_ml.yaml"
    main(command, yml)