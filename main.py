import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import click
import os

import scripts

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'


# os.environ["TORCH_DISTRIBUTED_DEBUG"] = 'DETAIL'
# os.environ["NCCL_DEBUG"] = 'INFO'


@click.group()
@click.pass_context
def main(ctx):
    """
    Training and evaluation
    """
    print("Mode:", ctx.invoked_subcommand)


@main.command()
@click.option(
    "-c",
    "--config-path",
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
# nohup python -u main.py train --config-path configs/voc12.yaml >nohup_logs/base_voc12_p1.log 2>&1 &
def train(config_path):
    scripts.op_train(config_path)


def cam(config_path):
    return NotImplementedError


if __name__ == "__main__":
    main()
