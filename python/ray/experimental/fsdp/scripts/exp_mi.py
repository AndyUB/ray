import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.info("Welcome to Downton Abbey!")


def run_command(command, env=None):
    """
    Run a shell command and return its exit code, stdout, and stderr.

    Args:
        command (list): Command to run as a list of arguments
        env (dict, optional): Environment variables to set

    Returns:
        tuple: (exit_code, stdout, stderr)
    """
    try:
        logger.info(f"Executing command: {' '.join(command)}")
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,  # Don't raise exception on non-zero exit
            env=env if env else os.environ.copy(),  # Pass environment
        )

        exit_code = process.returncode
        stdout = process.stdout
        stderr = process.stderr

        if stdout:
            logger.info(f"Command stdout: {stdout}")
        if stderr:
            logger.error(f"Command stderr: {stderr}")

        return exit_code, stdout, stderr
    except Exception as e:
        logger.error(f"Exception: {str(e)}")
        return -1, "", str(e)


def run_experiment(config):
    """
    Run a single experiment with the specified configuration.

    Args:
        config (dict): Experiment configuration

    Returns:
        tuple: (success, output_path, log_file)
    """
    # Set timezone
    os.environ["TZ"] = "America/Los_Angeles"

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Disable ray log deduplication
    os.environ["RAY_DEDUP_LOGS"] = "0"

    # Create output directory
    output_path = config["output_path"]
    os.makedirs(output_path, exist_ok=True)

    # Clean existing files
    for file in os.listdir(output_path):
        if file.endswith(".csv") or file.endswith(".log"):
            os.remove(os.path.join(output_path, file))

    logger.info(f"Running {output_path}...")

    # Set up experiment parameters
    batch_size = config["batch_size"]
    seq_len = config["seq_len"]
    num_partitions = config["num_partitions"]
    num_actors = config["num_actors"]
    num_iters = config["num_iters"]
    latency_prefix = timestamp
    log_file = os.path.join(output_path, f"{timestamp}.log")

    # Build command
    module_path = config["module_path"]
    command = [
        "python",
        "-m",
        module_path,
        "--batch-size",
        str(batch_size),
        "--seq-len",
        str(seq_len),
        "--num-partitions",
        str(num_partitions),
        "--num-actors",
        str(num_actors),
        "--num-iters",
        str(num_iters),
        "--output-path",
        output_path,
        "--latency-prefix",
        latency_prefix,
        "--tracing",
    ]

    # Redirect output to log file
    with open(log_file, "w") as f:
        process = subprocess.run(
            command,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )

    status = process.returncode

    if status != 0:
        logger.error(f"WA, status: {status}, log: {log_file}")
        return False, output_path, log_file

    logger.info(f"AC, log: {log_file}")
    return True, output_path, log_file


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Folder for experiments",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for experiments",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=1024,
        help="Sequence length for experiments",
    )
    parser.add_argument(
        "--num-partitions",
        type=int,
        default=18,
        help="Number of partitions",
    )
    parser.add_argument(
        "--num-actors",
        type=int,
        default=2,
        help="Number of actors",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=20,
        help="Number of iterations",
    )
    args = parser.parse_args()

    # Get current working directory for debugging
    current_dir = os.getcwd()
    logger.info(f"Current working directory: {current_dir}")

    # Define template variations
    variations = [
        # Ray configurations
        {
            "framework": "ray",
            "settings": [
                {"cc": "off", "ov": "off", "num_actors": 1},
                {"cc": "on", "ov": "off", "num_actors": args.num_actors},
                {"cc": "on", "ov": "on", "num_actors": args.num_actors},
            ],
        },
        # Torch configurations
        {
            "framework": "torch",
            "settings": [
                {"cc": "off", "fp": "on", "num_actors": 1},
                {"cc": "on", "fp": "on", "pf": "on", "num_actors": args.num_actors},
            ],
        },
    ]

    # Generate experiments from template
    experiments = []
    for variant in variations:
        framework = variant["framework"]
        for setting in variant["settings"]:
            # Build path components based on settings
            components = []

            # Add all available settings to path and module
            for key, value in setting.items():
                if key != "num_actors":  # num_actors is not part of the path
                    components.append(f"{key}_{value}")

            # Construct paths
            output_path = f"results/{args.folder}/llama3/{framework}/{'/'.join(components)}/exp_self"
            module_path = f"ray.experimental.fsdp.src.main.llama3.{framework}.{'.'.join(components)}"

            # Create experiment config
            experiment = {
                "output_path": output_path,
                "module_path": module_path,
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
                "num_partitions": args.num_partitions,
                "num_actors": setting["num_actors"],
                "num_iters": args.num_iters,
            }

            experiments.append(experiment)

    # Check if we're in the correct directory
    if not os.getcwd().endswith("python/ray/experimental/fsdp"):
        logger.error("Not running from python/ray/experimental/fsdp directory")
        sys.exit(1)

    # Run experiments sequentially
    for config in experiments:
        run_experiment(config)


if __name__ == "__main__":
    main()
