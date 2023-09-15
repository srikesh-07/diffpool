import argparse
import os
import random
import time

import dgl
import dgl.function as fn

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from data_utils import pre_process
from dgl import DGLGraph
from dgl.data import tu
from model.encoder import DiffPool

from sklearn.metrics import accuracy_score

global_train_time_per_epoch = []


def arg_parse():
    """
    argument parser
    """
    parser = argparse.ArgumentParser(description="DiffPool arguments")
    parser.add_argument("--dataset", dest="dataset", help="Input Dataset")
    parser.add_argument(
        "--pool_ratio", dest="pool_ratio", type=float, help="pooling ratio"
    )
    parser.add_argument(
        "--num_pool", dest="num_pool", type=int, help="num_pooling layer"
    )
    parser.add_argument(
        "--no_link_pred",
        dest="linkpred",
        action="store_false",
        help="switch of link prediction object",
    )
    parser.add_argument("--cuda", dest="cuda", type=int, help="switch cuda")
    parser.add_argument("--lr", dest="lr", type=float, help="learning rate")
    parser.add_argument(
        "--clip", dest="clip", type=float, help="gradient clipping"
    )
    parser.add_argument(
        "--batch-size", dest="batch_size", type=int, help="batch size"
    )
    parser.add_argument("--epochs", dest="epoch", type=int, help="num-of-epoch")
    parser.add_argument(
        "--train-ratio",
        dest="train_ratio",
        type=float,
        help="ratio of trainning dataset split",
    )
    parser.add_argument(
        "--test-ratio",
        dest="test_ratio",
        type=float,
        help="ratio of testing dataset split",
    )
    parser.add_argument(
        "--num_workers",
        dest="n_worker",
        type=int,
        help="number of workers when dataloading",
    )
    parser.add_argument(
        "--gc-per-block",
        dest="gc_per_block",
        type=int,
        help="number of graph conv layer per block",
    )
    parser.add_argument(
        "--bn",
        dest="bn",
        action="store_const",
        const=True,
        default=True,
        help="switch for bn",
    )
    parser.add_argument(
        "--dropout", dest="dropout", type=float, help="dropout rate"
    )
    parser.add_argument(
        "--bias",
        dest="bias",
        action="store_const",
        const=True,
        default=True,
        help="switch for bias",
    )
    parser.add_argument(
        "--save_dir",
        dest="save_dir",
        help="model saving directory: SAVE_DICT/DATASET",
    )
    parser.add_argument(
        "--load_epoch",
        dest="load_epoch",
        type=int,
        help="load trained model params from\
                         SAVE_DICT/DATASET/model-LOAD_EPOCH",
    )
    parser.add_argument(
        "--data_mode",
        dest="data_mode",
        help="data\
                        preprocessing mode: default, id, degree, or one-hot\
                        vector of degree number",
        choices=["default", "id", "deg", "deg_num"],
    )

    parser.set_defaults(
        dataset="ENZYMES",
        pool_ratio=0.15,
        num_pool=1,
        cuda=1,
        lr=1e-3,
        clip=2.0,
        batch_size=20,
        epoch=4000,
        train_ratio=0.7,
        test_ratio=0.1,
        n_worker=1,
        gc_per_block=3,
        dropout=0.0,
        method="diffpool",
        bn=True,
        bias=True,
        save_dir="./model_param",
        load_epoch=-1,
        data_mode="default",
    )
    return parser.parse_args()


def prepare_data(dataset, prog_args, train=False, pre_process=None):
    """
    preprocess TU dataset according to DiffPool's paper setting and load dataset into dataloader
    """
    if train:
        shuffle = True
    else:
        shuffle = False

    if pre_process:
        pre_process(dataset, prog_args)

    # dataset.set_fold(fold)
    return dgl.dataloading.GraphDataLoader(
        dataset,
        batch_size=prog_args.batch_size,
        shuffle=shuffle,
        num_workers=prog_args.n_worker,
    )


def graph_classify_task(prog_args):
    """
    perform graph classification task
    """

    dataset = tu.LegacyTUDataset(name=prog_args.dataset)
    train_size = int(prog_args.train_ratio * len(dataset))
    test_size = int(prog_args.test_ratio * len(dataset))
    val_size = int(len(dataset) - train_size - test_size)

    nodes = torch.zeros(len(dataset))
    for i in range(len(dataset)):
        nodes[i] = dataset[i].num_nodes

    _, ind = torch.sort(nodes, descending=True)

    head = 0
    med = 0
    tail = 0

    if prog_args.dataset == "PROTEINS":
        K = [0, 371, 742, 1113]
    elif prog_args.dataset == "PTC":
        K = [0, 115, 230, 344]
    elif prog_args.dataset == "IMDBBINARY":
        K = [0, 333, 666, 1000]
    elif prog_args.dataset == "DD":
        K = [0, 393, 785, 1178]
    elif prog_args.dataset == "FRANK":
        K =[0, 1445, 2890, 4337]

    for i in ind[K[0]:K[1]]:
        head += 1
        setattr(dataset[i], 'graph_group', 2)
    for i in ind[K[1]:K[2]]:
        med += 1
        setattr(dataset[i], 'graph_group', 1)
    for i in ind[K[2]:K[3]]:
        tail += 1
        setattr(dataset[i], 'graph_group', 0)

    dataset_train, dataset_val, dataset_test = torch.utils.data.random_split(
        dataset, (train_size, val_size, test_size)
    )
    train_dataloader = prepare_data(
        dataset_train, prog_args, train=True, pre_process=pre_process
    )
    val_dataloader = prepare_data(
        dataset_val, prog_args, train=False, pre_process=pre_process
    )
    test_dataloader = prepare_data(
        dataset_test, prog_args, train=False, pre_process=pre_process
    )
    input_dim, label_dim, max_num_node = dataset.statistics()
    print("++++++++++STATISTICS ABOUT THE DATASET")
    print("dataset feature dimension is", input_dim)
    print("dataset label dimension is", label_dim)
    print("the max num node is", max_num_node)
    print("number of graphs is", len(dataset))
    # assert len(dataset) % prog_args.batch_size == 0, "training set not divisible by batch size"

    hidden_dim = 64  # used to be 64
    embedding_dim = 64

    # calculate assignment dimension: pool_ratio * largest graph's maximum
    # number of nodes  in the dataset
    assign_dim = int(max_num_node * prog_args.pool_ratio)
    print("++++++++++MODEL STATISTICS++++++++")
    print("model hidden dim is", hidden_dim)
    print("model embedding dim for graph instance embedding", embedding_dim)
    print("initial batched pool graph dim is", assign_dim)
    activation = F.relu

    # initialize model
    # 'diffpool' : diffpool
    model = DiffPool(
        input_dim,
        hidden_dim,
        embedding_dim,
        label_dim,
        activation,
        prog_args.gc_per_block,
        prog_args.dropout,
        prog_args.num_pool,
        prog_args.linkpred,
        prog_args.batch_size,
        "meanpool",
        assign_dim,
        prog_args.pool_ratio,
    )

    if prog_args.load_epoch >= 0 and prog_args.save_dir is not None:
        model.load_state_dict(
            torch.load(
                prog_args.save_dir
                + "/"
                + prog_args.dataset
                + "/model.iter-"
                + str(prog_args.load_epoch)
            )
        )

    print("model init finished")
    print("MODEL:::::::", prog_args.method)
    if prog_args.cuda:
        model = model.cuda()

    logger = train(
        train_dataloader, model, prog_args, val_dataset=val_dataloader
    )
    result = evaluate(test_dataloader, model, prog_args, logger)
    print("Test  acc {:.2f}, head_acc {:.2f}, med_acc {:.2f}, tail_acc {:.2f}".format(result["acc"],
                                                                                            result["head_acc"],
                                                                                            result["med_acc"],
                                                                                            result["tail_acc"]))
    return logger, result


def train(dataset, model, prog_args, same_feat=True, val_dataset=None, seed=None):
    """
    training function
    """
    dir = os.path.join(prog_args.save_dir, str(seed), prog_args.dataset)
    if not os.path.exists(dir):
        os.makedirs(dir)
    dataloader = dataset
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.001
    )
    early_stopping_logger = {"best_epoch": -1, "val_acc": -1}

    if prog_args.cuda > 0:
        torch.cuda.set_device(0)
    for epoch in range(prog_args.epoch):
        begin_time = time.time()
        model.train()
        accum_correct = 0
        total = 0
        print("\nEPOCH ###### {} ######".format(epoch))
        computation_time = 0.0
        for batch_idx, (batch_graph, graph_labels) in enumerate(dataloader):
            for key, value in batch_graph.ndata.items():
                batch_graph.ndata[key] = value.float()
            graph_labels = graph_labels.long()
            if torch.cuda.is_available():
                batch_graph = batch_graph.to(torch.cuda.current_device())
                graph_labels = graph_labels.cuda()

            model.zero_grad()
            compute_start = time.time()
            ypred = model(batch_graph)
            indi = torch.argmax(ypred, dim=1)
            correct = torch.sum(indi == graph_labels).item()
            accum_correct += correct
            total += graph_labels.size()[0]
            loss = model.loss(ypred, graph_labels)
            loss.backward()
            batch_compute_time = time.time() - compute_start
            computation_time += batch_compute_time
            nn.utils.clip_grad_norm_(model.parameters(), prog_args.clip)
            optimizer.step()

        train_accu = accum_correct / total
        print(
            "train accuracy for this epoch {} is {:.2f}%".format(
                epoch, train_accu * 100
            )
        )
        elapsed_time = time.time() - begin_time
        print(
            "loss {:.4f} with epoch time {:.4f} s & computation time {:.4f} s ".format(
                loss.item(), elapsed_time, computation_time
            )
        )
        global_train_time_per_epoch.append(elapsed_time)
        if val_dataset is not None:
            result = evaluate(val_dataset, model, prog_args)
            print("validation  acc {:.2f}, head_acc {:.2f}, med_acc {:.2f}, tail_acc {:.2f}".format(result["acc"],
                                                                                                    result["head_acc"],
                                                                                                    result["med_acc"],
                                                                                                    result["tail_acc"]))
            if (
                result["acc"] >= early_stopping_logger["val_acc"]
                and result["acc"] <= train_accu
            ):
                early_stopping_logger.update(best_epoch=epoch, val_acc=result["acc"])
                if prog_args.save_dir is not None:
                    torch.save(
                        model.state_dict(),
                        prog_args.save_dir
                        + "/"
                        + prog_args.dataset
                        + "/model.iter-"
                        + str(early_stopping_logger["best_epoch"]),
                    )
            print(
                "best epoch is EPOCH {}, val_acc is {:.2f}%".format(
                    early_stopping_logger["best_epoch"],
                    early_stopping_logger["val_acc"] * 100,
                )
            )
        torch.cuda.empty_cache()
    return early_stopping_logger


def evaluate(dataloader, model, prog_args, logger=None):
    """
    evaluate function
    """
    if logger is not None and prog_args.save_dir is not None:
        model.load_state_dict(
            torch.load(
                prog_args.save_dir
                + "/"
                + prog_args.dataset
                + "/model.iter-"
                + str(logger["best_epoch"])
            )
        )
    model.eval()
    graph_preds = {0: [], 1: [], 2:[]}
    graph_correct = {0: [], 1:[], 2:[]}

    correct_label = 0
    with torch.no_grad():
        for batch_idx, (batch_graph, graph_labels) in enumerate(dataloader):
            for key, value in batch_graph.ndata.items():
                batch_graph.ndata[key] = value.float()
            graph_labels = graph_labels.long()
            for graph, label in zip(batch_graph, graph_labels):
                graph_correct[graph.graph_group].append(label)
            if torch.cuda.is_available():
                batch_graph = batch_graph.to(torch.cuda.current_device())
                graph_labels = graph_labels.cuda()
            ypred = model(batch_graph)
            indi = torch.argmax(ypred, dim=1)
            for index, graph in enumerate(batch_graph):
                graph_preds[graph.graph_group].append(indi[index].cpu().item())
            correct = torch.sum(indi == graph_labels)
            correct_label += correct.item()
    result = correct_label / (len(dataloader) * prog_args.batch_size)
    return dict(acc=result,
                head_acc=accuracy_score(graph_correct[2], graph_preds[2]),
                med_acc=accuracy_score(graph_correct[1], graph_preds[1]),
                tail_acc=accuracy_score(graph_correct[0], graph_preds[0]))



def main():
    """
    main
    """
    prog_args = arg_parse()

    SEEDS = [0, 1, 2, 3, 4]

    test_record = torch.zeros(len(SEEDS))
    valid_record = torch.zeros(len(SEEDS))
    tail_record = torch.zeros(len(SEEDS))
    medium_record = torch.zeros(len(SEEDS))
    head_record = torch.zeros(len(SEEDS))

    for seed in SEEDS:
        dgl.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        logger, test_result = graph_classify_task(prog_args)

        print(
            "Train time per epoch: {:.4f}".format(
                sum(global_train_time_per_epoch) / len(global_train_time_per_epoch)
            )
        )
        print(
            "Max memory usage: {:.4f}".format(
                torch.cuda.max_memory_allocated(0) / (1024 * 1024)
            )
        )
        valid_record[seed] = logger["val_acc"]
        test_record[seed] = test_result["acc"]
        head_record[seed] = test_result["head_acc"]
        medium_record[seed] = test_record["med_acc"]
        tail_record[seed] = test_record["tail_acc"]

    with open("metrics.txt", "a") as txt_file:
        txt_file.write(f"Dataset: {prog_args.dataset} \n"
                       # f"Alpha: {args.alpha}, \n"
                       # f"Mu: {args.mu1}, \n"
                       f"Valid Mean: {round(valid_record.mean().item(), 4)} \n"
                       f"Std Valid Mean: {round(valid_record.std().item(), 4)} \n"
                       f"Test Mean: {round(test_record.mean().item(), 4)} \n"
                       f"Std Test Mean: {round(test_record.std().item(), 4)} \n"
                       f"Head Mean: {round(head_record.mean().item(), 4)} \n"
                       f"Std Head Mean: {round(head_record.std().item(), 4)} \n"
                       f"Medium Mean: {round(medium_record.mean().item(), 4)} \n"
                       f"Std Medium Mean: {round(medium_record.std().item(), 4)} \n"
                       f"Tail Mean: {round(tail_record.mean().item(), 4)} \n"
                       f"Std Tail Mean: {round(tail_record.std().item(), 4)} \n\n"
                       )


if __name__ == "__main__":
    main()
