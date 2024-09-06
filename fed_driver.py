import os
import argparse as ap
import threading

from fed_utils import FedHelper


def run_script(os_system_command, exit_codes):
    exit_codes[os_system_command] = os.system(os_system_command)


def run_federated_learning(helper):
    fed_client_cmd = [f"python fed_client.py {helper.args_str} --cid={cid}" for cid in helper.clients]
    fed_server_cmd = [f"python fed_server.py {helper.args_str}"]
    
    commands = fed_client_cmd + fed_server_cmd
    for cmd in commands:
        print(cmd)

    threads = []
    exit_codes = {}
    threads = [threading.Thread(target=run_script, args=(cmd, exit_codes), daemon=True) for cmd in commands]
    for thd in threads:
        thd.start()
    for thd in threads:
        thd.join()
    for cmd in commands:
        assert exit_codes[cmd] == 0


if __name__ == "__main__":
    parser = ap.ArgumentParser(description="Creating Info. for all Setups in Fed_Driver.")
    parser.add_argument("--model", type=str, default="cnn_model")
    parser.add_argument("--cnum", type=int, default=3)
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--local_round", type=int, default=2)
    parser.add_argument("--global_round", type=int, default=4)
    parser.add_argument("--setup", type=str, default="same")
    parser.add_argument("--clients", type=eval, default=[0, 1, 2])
    args = parser.parse_args()

    assert all(0 <= cid < args.cnum for cid in args.clients)
    run_federated_learning(FedHelper(vars(args)))
