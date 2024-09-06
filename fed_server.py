import argparse as ap
import fed_proto_pb2
import fed_proto_pb2_grpc
import grpc
import threading
import time
import numpy as np
from queue import Queue

from fed_utils import ServerHelper, nparray_to_rpcio, rpcio_to_nparray
from fed_models import MODEL_DICT, SHAPE_DICT


def get_params_from_client(grad_stub, grad_data, grad_type, grad_shape, cid, share_queue):
    print("Now we call the local train from client#{}".format(cid))
    local_params = grad_stub.grad_descent(
        fed_proto_pb2.grad_request(
            server_grad_para_data=grad_data, server_grad_para_type=grad_type, server_grad_para_shape=grad_shape
        )
    )
    print("Now we put local_params in share_queue of {}".format(cid))
    share_queue.put(local_params)


# here the fed_server calls the remote function of clients.
def run_fed_server(helper):
    clients = helper.clients
    global_model = MODEL_DICT[helper.model](*SHAPE_DICT[helper.dataset])
    global_round = helper.global_round
    print("Now the combination of client is ", clients)

    channels = [grpc.insecure_channel(f"localhost:{port}") for port in helper.ports]
    grad_stubs = [fed_proto_pb2_grpc.GradServiceStub(channel) for channel in channels]
    size_stubs = [fed_proto_pb2_grpc.DatasizeServiceStub(channel) for channel in channels]
    stop_stubs = [fed_proto_pb2_grpc.StopServiceStub(channel) for channel in channels]
    print("Build the channel to each client to execute the local trainning")

    test_dataset = helper.load_test_data()
    print("Loaded the test data")

    print("Globally train the federated model, i.e. execute the stub.grad_decendent")
    global_params = global_model.get_params()
    datasize_response = [stub.get_datasize(fed_proto_pb2.datasize_request(size=0)) for stub in size_stubs]
    np_datasize = np.array([r.size for r in datasize_response])
    tot_datasize = np.sum(np_datasize)
    grad_alpha = np_datasize / tot_datasize
    print(f"## The datasize from rpc is: {np_datasize} ## allsize is {tot_datasize}")

    share_queues = [Queue() for _ in clients]
    for round in range(global_round):
        grad_data, grad_type, grad_shape = nparray_to_rpcio(global_params)
        threadings = []
        for i, cid in enumerate(clients):
            thread = threading.Thread(
                target=get_params_from_client,
                args=(grad_stubs[i], grad_data, grad_type, grad_shape, cid, share_queues[i]),
                daemon=True,
                name=f"client#{cid}",
            )
            print(f"Start the Thread-{cid}")
            thread.start()
            threadings.append(thread)
        for thread in threadings:
            thread.join()
        print("All threadings joined")

        for sq in share_queues:
            assert sq.empty() == False
        rpcio_responses = [sq.get() for sq in share_queues]
        local_params_all = [
            rpcio_to_nparray(r.client_grad_para_data, r.client_grad_para_type, r.client_grad_para_shape)
            for r in rpcio_responses
        ]
        print("Have got the response of weights from client")

        # exec aggregation on each layer
        global_params = [np.zeros(layer.shape) for layer in global_params]
        for layer in range(len(global_params)):
            for i, local_params in enumerate(local_params_all):
                global_params[layer] += grad_alpha[i] * local_params[layer]

        # test the performance of global model.
        global_model.load_params(global_params)
        loss, acc = global_model.eval(test_dataset)
        print("Clients {}: Round: {}, Acc: {}, Loss: {}".format(clients, round, acc, loss))
    
    for stop_stub in stop_stubs:
        stop_stub.stop(fed_proto_pb2.stop_request(message="server"))

    loss, acc = global_model.eval(test_dataset)
    return acc, loss


if __name__ == "__main__":
    parser = ap.ArgumentParser(description="Creating Info. for all Setups in Fed_Driver.")
    parser.add_argument("--model", type=str, default="linear_model")
    parser.add_argument("--cnum", type=int, default=3)
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--local_round", type=int, default=2)
    parser.add_argument("--global_round", type=int, default=4)
    parser.add_argument("--setup", type=str, default="same")
    parser.add_argument("--clients", type=eval, default=[0, 1, 2])
    args = parser.parse_args()

    # waiting for the build-up of client
    seconds = 10
    for i in range(seconds):
        print(f"The Server is waiting for the client until {seconds - i} seconds.")
        time.sleep(1)

    # run the server in FL & record the running time of FL
    helper = ServerHelper(vars(args))
    begin_time = time.perf_counter()
    acc, loss = run_fed_server(helper)
    end_time = time.perf_counter()
    helper.record_metrics(acc, loss, end_time - begin_time)
