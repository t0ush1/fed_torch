import os
import pickle as pk
import numpy as np


BASIC_PORT = 50100
STOP_PORT = 50120
PORT_GAP = 25


def nparray_to_rpcio(nparray):
    byte_array_data = [x.tobytes() for x in nparray]
    byte_array_type = [str(x.dtype) for x in nparray]
    byte_array_shape = [str(x.shape) for x in nparray]
    return byte_array_data, byte_array_type, byte_array_shape


def rpcio_to_nparray(byte_data, byte_type, byte_shape):
    return [
        np.frombuffer(data, dtype=np.dtype(rtype)).reshape(eval(shape))
        for data, rtype, shape in zip(byte_data, byte_type, byte_shape)
    ]


class FedHelper:
    def __init__(self, config):
        self.model = config["model"]
        self.cnum = config["cnum"]
        self.dataset = config["dataset"]
        self.local_round = config["local_round"]
        self.global_round = config["global_round"]
        self.setup = config["setup"]
        self.clients = config["clients"]
        self.index = sum(map(lambda i: 1 << i, self.clients))
        self.args_str = " ".join(f"--{k}='{v}'" if type(v) is list else f"--{k}={v}" for k, v in config.items())
        self.config_str = f"{self.model}_{self.cnum}_{self.dataset}_{self.setup}"


class ServerHelper(FedHelper):
    def __init__(self, config):
        super().__init__(config)
        self.ports = [BASIC_PORT + cid for cid in self.clients]

    def load_test_data(self):
        file_path = f"./datasets/{self.dataset}/client_{self.cnum}_{self.setup}"
        with open(f"{file_path}/testX.pk", "rb") as file_x:
            data_x = pk.load(file_x)
        with open(f"{file_path}/testY.pk", "rb") as file_y:
            data_y = pk.load(file_y)
        return data_x, data_y

    def record_metrics(self, accuracy, loss, time):
        dir_path = "./rec_fed_metrics/"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        file_path = dir_path + self.config_str + ".rec"
        if not os.path.exists(file_path):
            print("Now we create the " + file_path)
            with open(file_path, "wb") as fout:
                record = {}
                pk.dump(record, fout)

        with open(file_path, "rb") as fin:
            record = pk.load(fin)
        record[self.index] = {"loss": loss, "acc": accuracy, "time": time}
        with open(file_path, "wb") as fout:
            pk.dump(record, fout)
        print("Record metrics:", record)


class ClientHelper(FedHelper):
    def __init__(self, config):
        super().__init__(config)
        self.cid = config["cid"]
        self.port = BASIC_PORT + self.cid

    def load_train_data(self):
        file_path = f"./datasets/{self.dataset}/client_{self.cnum}_{self.setup}"
        with open(f"{file_path}/client_trainX_{self.cid}.pk", "rb") as file_x:
            data_x = pk.load(file_x)
        with open(f"{file_path}/client_trainY_{self.cid}.pk", "rb") as file_y:
            data_y = pk.load(file_y)
        return data_x, data_y

    def record_grad(self, round, pre_params, cur_params, datasize):
        dir_path = f"./rec_fed_grad/{self.config_str}/"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # record the datasize of each client.
        if round == 0:
            file_path = dir_path + f"{self.cid}.datasize"
            with open(file_path, "wb") as fout:
                pk.dump(datasize, fout)

        # save the data in pickle form
        file_path = dir_path + f"{self.cid}_{round}.grad_rec"
        grad = [cur - pre for (pre, cur) in zip(pre_params, cur_params)]
        with open(file_path, "wb") as fout:
            pk.dump(grad, fout)
        print("Record grad:", file_path)
