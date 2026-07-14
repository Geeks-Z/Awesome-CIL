import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbone.bilora_fft import Attention_FFT, SiNet
from models.base import BaseLearner
from utils.schedulers import CosineSchedule
from utils.toolkit import accuracy, tensor2numpy


def _collect_trainable_names(network, task_index):
    trainable_names = []
    for name, param in network.named_parameters():
        is_trainable = (
            f"classifier_pool.{task_index}." in name
            or name.endswith(f"coef_k.{task_index}")
            or name.endswith(f"coef_v.{task_index}")
        )
        param.requires_grad_(is_trainable)
        if is_trainable:
            trainable_names.append(name)
    return sorted(trainable_names)


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)

        if args["net_type"] == "sip":
            self._network = SiNet(args)
        else:
            raise ValueError("Unknown net: {}.".format(args["net_type"]))

        for module in self._network.modules():
            if isinstance(module, Attention_FFT):
                module.init_param()

        self.args = args
        self.optim = args["optim"]
        self.EPSILON = args["EPSILON"]
        self.init_epoch = args["init_epoch"]
        self.init_lr = args["init_lr"]
        self.init_lr_decay = args["init_lr_decay"]
        self.init_weight_decay = args["init_weight_decay"]
        self.epochs = args["epochs"]
        self.lrate = args["lrate"]
        self.lrate_decay = args["lrate_decay"]
        self.batch_size = args["batch_size"]
        self.weight_decay = args["weight_decay"]
        self.num_workers = args["num_workers"]
        self.lamb = args["lamb"]
        self.lame = args["lame"]
        self.total_sessions = args["total_sessions"]
        self.dataset = args["dataset"]

        self.topk = args.get("topk", 5)
        self.class_num = self._network.class_num
        self.debug = False

        self.all_keys = []
        self.feature_list = []
        self.project_type = []

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)

        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        self._train(self.train_loader, self.test_loader)
        self.clustering(self.train_loader)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        task_index = (
            self._network.module.numtask - 1
            if isinstance(self._network, nn.DataParallel)
            else self._network.numtask - 1
        )
        enabled = _collect_trainable_names(self._network, task_index)

        with torch.no_grad():
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                self._network(inputs, get_cur_feat=True)

        # logging.info("Parameters to be updated: %s", enabled)
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        if self._cur_task == 0:
            if self.optim == "sgd":
                optimizer = optim.SGD(
                    self._network.parameters(),
                    momentum=0.9,
                    lr=self.init_lr,
                    weight_decay=self.init_weight_decay,
                )
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer, T_max=self.init_epoch
                )
            elif self.optim == "adam":
                optimizer = optim.Adam(
                    self._network.parameters(),
                    lr=self.init_lr,
                    weight_decay=self.init_weight_decay,
                    betas=(0.9, 0.999),
                )
                scheduler = CosineSchedule(optimizer=optimizer, K=self.init_epoch)
            else:
                raise Exception
            self.run_epoch = self.init_epoch
            self.train_function(train_loader, test_loader, optimizer, scheduler)
        else:
            if self.optim == "sgd":
                optimizer = optim.SGD(
                    self._network.parameters(),
                    momentum=0.9,
                    lr=self.lrate,
                    weight_decay=self.weight_decay,
                )
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer, T_max=self.epochs
                )
            elif self.optim == "adam":
                optimizer = optim.Adam(
                    self._network.parameters(),
                    lr=self.lrate,
                    weight_decay=self.weight_decay,
                    betas=(0.9, 0.999),
                )
                scheduler = CosineSchedule(optimizer=optimizer, K=self.epochs)
            else:
                raise Exception
            self.run_epoch = self.epochs
            self.train_function(train_loader, test_loader, optimizer, scheduler)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        return

    def train_function(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.run_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.eval()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                mask = (targets >= self._known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask) - self._known_classes

                logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                if self.debug and i > 10:
                    break

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.run_epoch,
                losses / len(train_loader),
                train_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)

    def clustering(self, dataloader):
        features = []
        for _, inputs, targets in dataloader:
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            mask = (targets >= self._known_classes).nonzero().view(-1)
            inputs = torch.index_select(inputs, 0, mask)
            with torch.no_grad():
                if isinstance(self._network, nn.DataParallel):
                    feature = self._network.module.extract_vector(inputs)
                else:
                    feature = self._network.extract_vector(inputs)
            feature = feature / feature.norm(dim=-1, keepdim=True)
            features.append(feature)
        features = torch.cat(features, 0).cpu().detach().numpy()
        clustering = KMeans(n_clusters=5, random_state=0).fit(features)
        self.all_keys.append(
            torch.tensor(clustering.cluster_centers_).to(feature.device)
        )

    def eval_task(self):
        y_pred, y_pred_with_task, y_true, y_pred_task, y_true_task = self._eval_cnn(
            self.test_loader
        )
        cnn_accy = self._evaluate(y_pred, y_true)
        cnn_accy_with_task = self._evaluate(y_pred_with_task, y_true)
        cnn_accy_task = round((y_pred_task == y_true_task).float().mean().item(), 4)
        return cnn_accy, cnn_accy_with_task, None, cnn_accy_task

    def _evaluate(self, y_pred, y_true):
        ret = {}
        top1_pred = y_pred.T[0] if y_pred.ndim > 1 else y_pred
        grouped = accuracy(
            top1_pred,
            y_true,
            self._known_classes,
            self.args["init_cls"],
            self.args["increment"],
        )
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        if y_pred.ndim > 1:
            ret["top{}".format(self.topk)] = np.around(
                (y_pred.T == np.tile(y_true, (y_pred.shape[1], 1))).sum()
                * 100
                / len(y_true),
                decimals=2,
            )
        else:
            ret["top{}".format(self.topk)] = ret["top1"]
        if "top5" not in ret:
            ret["top5"] = ret["top{}".format(self.topk)]
        return ret

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        y_pred_with_task = []
        y_pred_task, y_true_task = [], []
        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            with torch.no_grad():
                if isinstance(self._network, nn.DataParallel):
                    outputs = self._network.module.interface(inputs)
                else:
                    outputs = self._network.interface(inputs)

            predicts = torch.topk(
                outputs,
                k=min(self.topk, outputs.shape[1]),
                dim=1,
                largest=True,
                sorted=True,
            )[1]
            y_pred_task.append((predicts[:, 0] // self.class_num).cpu())
            y_true_task.append((targets // self.class_num).cpu())

            outputs_with_task = torch.zeros(
                outputs.shape[0],
                self.class_num,
                device=outputs.device,
                dtype=outputs.dtype,
            )
            task_ids = targets // self.class_num
            for index, task_id in enumerate(task_ids.tolist()):
                begin = self.class_num * task_id
                end = begin + self.class_num
                outputs_with_task[index] = outputs[index, begin:end]
            predicts_with_task = (
                outputs_with_task.argmax(dim=1) + task_ids * self.class_num
            )

            y_pred.append(predicts.cpu().numpy())
            y_pred_with_task.append(predicts_with_task.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return (
            np.concatenate(y_pred),
            np.concatenate(y_pred_with_task),
            np.concatenate(y_true),
            torch.cat(y_pred_task),
            torch.cat(y_true_task),
        )

    def _compute_accuracy_domain(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]

            predicts = torch.max(outputs, dim=1)[1]
            correct += (
                (predicts % self.class_num).cpu() == (targets % self.class_num)
            ).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)
