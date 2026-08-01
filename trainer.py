import sys
import logging
import copy
import time

import torch
from utils import factory
from utils.data_manager import DataManager
from utils.logging_utils import canonical_log_dir
from utils.toolkit import count_parameters
import os
import random
import numpy as np


def _log_and_print(message):
    print(message)
    logging.info(message)


def _sync_device(device):
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _safe_mean(values):
    values = [value for value in values if value is not None and not np.isnan(value)]
    if len(values) == 0:
        return 0.0
    return float(np.mean(values))


def _get_epoch_count(args, model, task_id):
    run_epoch = getattr(model, "run_epoch", None)
    if isinstance(run_epoch, (int, float)) and run_epoch > 0:
        return int(run_epoch)

    model_name = args["model_name"].lower()
    if model_name == "ease":
        if task_id == 0 or args["init_cls"] == args["increment"]:
            return int(args.get("init_epochs", 1))
        return int(args.get("later_epochs") or args.get("init_epochs", 1))
    if model_name == "foster":
        if task_id == 0:
            return int(args.get("init_epochs", 1))
        return int(args.get("boosting_epochs", 0)) + int(args.get("compression_epochs", 0))
    if model_name == "hidep":
        epochs = int(args.get("tii_epochs", args.get("original_epochs", args.get("epochs", 0))))
        epochs += int(args.get("epochs", 0))
        if task_id > 0 and not args.get("not_train_ca", False):
            epochs += int(args.get("crct_epochs", 0))
        return max(epochs, 1)

    if "tuned_epoch" in args:
        return int(args["tuned_epoch"])
    if task_id == 0:
        return int(args.get("init_epoch", args.get("init_epochs", args.get("epochs", 1))))
    return int(args.get("epochs", args.get("later_epochs", args.get("init_epoch", 1))))


def _get_eval_image_count(model):
    test_loader = getattr(model, "test_loader", None)
    if test_loader is None or not hasattr(test_loader, "dataset"):
        return 0
    return len(test_loader.dataset)


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    backbone_name = args["backbone_type"]
    if backbone_name.startswith("pretrained_"):
        backbone_name = backbone_name[len("pretrained_") :]

    log_dir = canonical_log_dir(args["model_name"])
    logs_name = "logs/{}".format(log_dir)

    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}_{}_{}_{}_{}".format(
        log_dir,
        args["dataset"],
        backbone_name,
        init_cls,
        args["increment"],
        args["seed"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        force=True,
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )

    args["nb_classes"] = data_manager.nb_classes  # update args
    args["nb_tasks"] = data_manager.nb_tasks
    model = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_curve_with_task, cnn_curve_task = {"top1": []}, {"top1": []}
    cnn_matrix, nme_matrix = [], []

    total_train_time = 0.0
    total_test_time = 0.0
    total_eval_images = 0
    train_seconds_per_epoch = []
    inference_milliseconds_per_image = []

    is_bilora = args["model_name"].lower() == "bilora"

    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        _sync_device(args["device"][0])
        start_time = time.perf_counter()
        model.incremental_train(data_manager)
        _sync_device(args["device"][0])

        train_end_time = time.perf_counter()
        task_train_time = train_end_time - start_time
        total_train_time += task_train_time
        task_epochs = _get_epoch_count(args, model, task)
        task_train_time_per_epoch = task_train_time / max(task_epochs, 1)
        train_seconds_per_epoch.append(task_train_time_per_epoch)
        # print('Time for task {}: {}'.format(task, total_time))

        _sync_device(args["device"][0])
        eval_start_time = time.perf_counter()
        if is_bilora:
            cnn_accy, cnn_accy_with_task, nme_accy, cnn_accy_task = model.eval_task()
        else:
            cnn_accy, nme_accy = model.eval_task()
        _sync_device(args["device"][0])
        eval_end_time = time.perf_counter()
        task_test_time = eval_end_time - eval_start_time
        total_test_time += task_test_time
        task_eval_images = _get_eval_image_count(model)
        total_eval_images += task_eval_images
        if task_eval_images > 0:
            task_ms_per_image = task_test_time * 1000.0 / task_eval_images
            inference_milliseconds_per_image.append(task_ms_per_image)
        else:
            task_ms_per_image = np.nan

        timing_info = (
            "Timing Task {} => Train {:.2f}s, Epochs {}, Train {:.4f}s/epoch, "
            "Inference {:.2f}s, Images {}, Inference {:.4f}ms/image"
        ).format(
            task,
            task_train_time,
            task_epochs,
            task_train_time_per_epoch,
            task_test_time,
            task_eval_images,
            task_ms_per_image,
        )
        logging.info(timing_info)
        print(timing_info)
        model.after_task()

        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if "-" in key]
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            nme_keys = [key for key in nme_accy["grouped"].keys() if "-" in key]
            nme_values = [nme_accy["grouped"][key] for key in nme_keys]
            nme_matrix.append(nme_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])

            if is_bilora:
                cnn_curve_with_task["top1"].append(cnn_accy_with_task["top1"])
                cnn_curve_task["top1"].append(cnn_accy_task)

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            if is_bilora:
                logging.info(
                    "CNN top1 with task curve: {}".format(cnn_curve_with_task["top1"])
                )
                logging.info("CNN top1 task curve: {}".format(cnn_curve_task["top1"]))
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top5 curve: {}".format(nme_curve["top5"]))

            # print('Average Accuracy (CNN):', round(sum(cnn_curve["top1"]) / len(cnn_curve["top1"]), 2))
            # print('Average Accuracy (NME):', round(sum(nme_curve["top1"]) / len(nme_curve["top1"]), 2))

            logging.info(
                "Average Accuracy (CNN): {}".format(
                    round(sum(cnn_curve["top1"]) / len(cnn_curve["top1"]), 2)
                )
            )
            logging.info(
                "Average Accuracy (NME): {}".format(
                    round(sum(nme_curve["top1"]) / len(nme_curve["top1"]), 2)
                )
            )
            # logging.info("Train Time: {}".format(model.train_time))
            # logging.info("Test Time: {} \n".format(model.test_time))
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if "-" in key]
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            if is_bilora:
                cnn_curve_with_task["top1"].append(cnn_accy_with_task["top1"])
                cnn_curve_task["top1"].append(cnn_accy_task)

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            if is_bilora:
                logging.info(
                    "CNN top1 with task curve: {}".format(cnn_curve_with_task["top1"])
                )
                logging.info("CNN top1 task curve: {}".format(cnn_curve_task["top1"]))
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))

            logging.info(
                "Average Accuracy (CNN): {}".format(
                    round(sum(cnn_curve["top1"]) / len(cnn_curve["top1"]), 2)
                )
            )
            # logging.info("Train Time: {}".format(model.train_time))
            # logging.info("Test Time: {} \n".format(model.test_time))
    # print(
    #     "Finished {}_init{}_inc{}: {} seed={}  ".format(
    #         args["dataset"],
    #         args["init_cls"],
    #         args["increment"],
    #         args["backbone_type"],
    #         args["seed"],
    #     )
    # )
    separator = "=" * 80
    print(separator)
    print(
        "Finished {}_init{}_inc{}: {}  seed={} ".format(
            args["dataset"],
            args["init_cls"],
            args["increment"],
            args["backbone_type"],
            args["seed"],
        )
    )
    print("Last Accuracy: {}".format(round(cnn_curve["top1"][-1], 2)))
    print(
        "Average Accuracy (CNN): {}".format(
            round(sum(cnn_curve["top1"]) / len(cnn_curve["top1"]), 2)
        )
    )
    print("PD: {:.2f}".format(cnn_curve["top1"][0] - cnn_curve["top1"][-1]))
    # print("Total Train Time: {:.2f}s".format(total_train_time))
    # print("Total Inference Time: {:.2f}s".format(total_test_time))
    # print(
    #     "Average Training Time per Epoch per Incremental Task: {:.4f}s".format(
    #         _safe_mean(train_seconds_per_epoch)
    #     )
    # )
    # if total_eval_images > 0:
    #     print(
    #         "Average Inference Time: {:.4f}ms/image".format(
    #             total_test_time * 1000.0 / total_eval_images
    #         )
    #     )
    # print(
    #     "Mean Task Inference Time: {:.4f}ms/image".format(
    #         _safe_mean(inference_milliseconds_per_image)
    #     )
    # )
    # logging.info("Total Train Time: {:.2f}s".format(total_train_time))
    # logging.info("Total Inference Time: {:.2f}s".format(total_test_time))
    # logging.info(
    #     "Average Training Time per Epoch per Incremental Task: {:.4f}s".format(
    #         _safe_mean(train_seconds_per_epoch)
    #     )
    # )
    # if total_eval_images > 0:
    #     logging.info(
    #         "Average Inference Time: {:.4f}ms/image".format(
    #             total_test_time * 1000.0 / total_eval_images
    #         )
    #     )
    # logging.info(
    #     "Mean Task Inference Time: {:.4f}ms/image".format(
    #         _safe_mean(inference_milliseconds_per_image)
    #     )
    # )
    if len(cnn_matrix) > 0:
        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(cnn_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)
        np_acctable = np_acctable.T
        forgetting = np.mean(
            (np.max(np_acctable, axis=1) - np_acctable[:, task])[:task]
        )
        print("Accuracy Matrix (CNN):")
        print(str(np_acctable))
        print("Forgetting (CNN): {}".format(forgetting))

    if len(nme_matrix) > 0:
        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(nme_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)
        np_acctable = np_acctable.T
        forgetting = np.mean(
            (np.max(np_acctable, axis=1) - np_acctable[:, task])[:task]
        )
        print("Accuracy Matrix (NME):")
        print(str(np_acctable))
        print("Forgetting (NME): {}".format(forgetting))

    print(separator)


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus

    if gpus and gpus[0].type == "cuda":
        torch.cuda.set_device(gpus[0])


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
