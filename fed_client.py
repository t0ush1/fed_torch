import argparse as ap
import fed_proto_pb2
import fed_proto_pb2_grpc
import grpc
import numpy # 不加会出 bug，不懂
from concurrent import futures
import threading

from fed_models import DEFAULT_BATCH, SHAPE_DICT, MODEL_DICT
from fed_utils import ClientHelper, nparray_to_rpcio, rpcio_to_nparray


class DatasizeServicer(fed_proto_pb2_grpc.DatasizeServiceServicer):
    def __init__(self, dataset, cid):
        self.datasize = len(dataset[1])
        self.cid = cid

    def get_datasize(self, request, context):
        return fed_proto_pb2.datasize_reply(size=self.datasize)


class GradServicer(fed_proto_pb2_grpc.GradServiceServicer):
    def __init__(self, helper, dataset, model, cid, epoch):
        self.helper = helper
        self.dataset = dataset
        self.datasize = len(dataset[1])
        self.model = model
        self.cid = cid
        self.epoch = epoch
        self.batch = DEFAULT_BATCH
        self.round = 0  # record the global round we are in.

    def grad_descent(self, request, context):
        # Download the global model by grpc (rpcio-->np.array)
        print(f"The client#{self.cid} executes the local trainning with {self.epoch} epoches")
        byte_data = list(request.server_grad_para_data)
        byte_type = list(request.server_grad_para_type)
        byte_shape = list(request.server_grad_para_shape)
        global_params = rpcio_to_nparray(byte_data, byte_type, byte_shape)
        self.model.load_params(global_params)

        # Locally train the globally model
        self.model.fit(self.dataset, self.epoch, self.batch)
        local_params = self.model.get_params()

        # Record the grad of this client
        self.helper.record_grad(self.round, global_params, local_params, self.datasize)
        self.round += 1

        # reply the updates model by grpc (np.array-->rpcio)
        byte_data, byte_type, byte_shape = nparray_to_rpcio(local_params)
        print(f"Reply gradients to server by client#{self.cid}")
        return fed_proto_pb2.grad_reply(
            client_grad_para_data=byte_data, client_grad_para_type=byte_type, client_grad_para_shape=byte_shape
        )


class StopServicer(fed_proto_pb2_grpc.StopServiceServicer):
    def __init__(self, stop_event, cid):
        self.stop_event = stop_event
        self.cid = cid

    def stop(self, request, context):
        print(f"The client#{self.cid} received the stop request from {request.message}")
        self.stop_event.set()
        return fed_proto_pb2.stop_reply(message=f"The client {self.cid} has been stopped!")


def run_fed_client(helper):
    cid = helper.cid
    epoch = helper.local_round
    model = MODEL_DICT[helper.model](*SHAPE_DICT[helper.dataset])
    dataset = helper.load_train_data()
    stop_event = threading.Event()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    fed_proto_pb2_grpc.add_GradServiceServicer_to_server(GradServicer(helper, dataset, model, cid, epoch), server)
    fed_proto_pb2_grpc.add_DatasizeServiceServicer_to_server(DatasizeServicer(dataset, cid), server)
    fed_proto_pb2_grpc.add_StopServiceServicer_to_server(StopServicer(stop_event, cid), server)
    server.add_insecure_port(f"[::]:{helper.port}")

    print(f"Rpc server of client#{cid} is created.")
    server.start()
    stop_event.wait()
    print(f"The client#{cid} have been stoped.")


if __name__ == "__main__":
    parser = ap.ArgumentParser(description="Creating Info. for Comp. Shapley.")
    parser.add_argument("--model", type=str, default="linear_model")
    parser.add_argument("--cnum", type=int, default=3)
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--local_round", type=int, default=2)
    parser.add_argument("--global_round", type=int, default=4)
    parser.add_argument("--setup", type=str, default="same")
    parser.add_argument("--clients", type=eval, default=[0, 1, 2])
    parser.add_argument("--cid", type=int, default=0)
    args = parser.parse_args()
    
    run_fed_client(ClientHelper(vars(args)))
