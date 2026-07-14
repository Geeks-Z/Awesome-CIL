import logging
import math

import numpy as np
import torch
from sklearn.cluster import KMeans
from timm.models import create_model
from torch import nn, optim
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import backbone.vit_hidep  # noqa: F401
from models.base import BaseLearner
from utils.toolkit import accuracy, tensor2numpy


num_workers = 8


def _normalize_backbone_name(name):
    name = name.lower()
    aliases = {
        "pretrained_vit_b16_224": "vit_base_patch16_224",
        "pretrained_vit_b16_224_in21k": "vit_base_patch16_224_in21k",
        "pretrained_vit_large_patch16_224": "vit_large_patch16_224",
        "pretrained_vit_base_patch16_224_dino": "vit_base_patch16_224_dino",
        "pretrained_vit_base_patch16_224_sam": "vit_base_patch16_224_sam",
    }
    name = aliases.get(name, name)
    if name.endswith("_hidep"):
        name = name[: -len("_hidep")]
    return aliases.get(name, name)


def _get(args, key, default):
    return args[key] if key in args else default


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.batch_size = args["batch_size"]
        self.num_workers = _get(args, "num_workers", num_workers)
        self.nb_classes = args["nb_classes"]
        self.init_cls = args["init_cls"]
        self.inc = args["increment"]
        self.topk = _get(args, "topk", 5)

        self.class_mask = []
        self.target_task_map = {}
        self.cls_mean = {}
        self.cls_cov = {}

        self.original_model = self._build_original_model()
        self._network = self._build_prompt_model()
        self._apply_freeze(self.original_model)
        self._apply_freeze(self._network)

    def _build_original_model(self):
        name = _normalize_backbone_name(
            self.args.get("original_backbone_type", self.args.get("original_model", self.args["backbone_type"]))
        )
        return create_model(
            name,
            pretrained=self.args.get("pretrained", True),
            num_classes=self.nb_classes,
            drop_rate=self.args.get("drop", 0.0),
            drop_path_rate=self.args.get("drop_path", 0.0),
            drop_block_rate=None,
            mlp_structure=self.args.get("original_model_mlp_structure", [2]),
        )

    def _build_prompt_model(self):
        name = _normalize_backbone_name(self.args.get("model", self.args["backbone_type"]))
        return create_model(
            name,
            pretrained=self.args.get("pretrained", True),
            num_classes=self.nb_classes,
            drop_rate=self.args.get("drop", 0.0),
            drop_path_rate=self.args.get("drop_path", 0.0),
            drop_block_rate=None,
            prompt_length=self.args.get("length", 5),
            embedding_key=self.args.get("embedding_key", "cls"),
            prompt_init=self.args.get("prompt_key_init", "uniform"),
            prompt_pool=self.args.get("prompt_pool", True),
            prompt_key=self.args.get("prompt_key", False),
            pool_size=self.args.get("size", self.args["nb_tasks"]),
            top_k=self.args.get("top_k", 1),
            batchwise_prompt=self.args.get("batchwise_prompt", False),
            prompt_key_init=self.args.get("prompt_key_init", "uniform"),
            head_type=self.args.get("head_type", "token"),
            use_prompt_mask=self.args.get("use_prompt_mask", True),
            use_g_prompt=self.args.get("use_g_prompt", False),
            g_prompt_length=self.args.get("g_prompt_length", 5),
            g_prompt_layer_idx=self.args.get("g_prompt_layer_idx", []),
            use_prefix_tune_for_g_prompt=self.args.get("use_prefix_tune_for_g_prompt", False),
            use_e_prompt=self.args.get("use_e_prompt", True),
            e_prompt_layer_idx=self.args.get("e_prompt_layer_idx", [0, 1, 2, 3, 4]),
            use_prefix_tune_for_e_prompt=self.args.get("use_prefix_tune_for_e_prompt", True),
            same_key_value=self.args.get("same_key_value", False),
        )

    def _apply_freeze(self, model):
        freeze = tuple(self.args.get("freeze", ["blocks", "patch_embed", "cls_token", "norm", "pos_embed"]))
        if not freeze:
            return
        for name, param in model.named_parameters():
            if name.startswith(freeze):
                param.requires_grad = False

    def _task_range(self, task_id):
        if task_id == 0:
            return 0, self.init_cls
        start = self.init_cls + (task_id - 1) * self.inc
        return start, start + self.inc

    def _update_class_mask(self):
        while len(self.class_mask) <= self._cur_task:
            start, end = self._task_range(len(self.class_mask))
            classes = list(range(start, end))
            self.class_mask.append(classes)
            for cls_id in classes:
                self.target_task_map[cls_id] = len(self.class_mask) - 1

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._update_class_mask()
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train"
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        train_eval_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="test"
        )
        self.train_loader_for_stats = DataLoader(
            train_eval_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

        self.original_model.to(self._device)
        self._network.to(self._device)
        self._train_original(self.train_loader)
        self._transfer_prompt_params()
        self._train_prompt(self.train_loader)
        self._apply_prompt_momentum()
        self._compute_mean(self.train_loader_for_stats)
        if self._cur_task > 0 and not self.args.get("not_train_ca", False):
            self._train_task_adaptive_prediction()

    def _make_optimizer(self, params, lr, weight_decay=0.0):
        opt_name = self.args.get("opt", self.args.get("optimizer", "adam")).lower()
        if opt_name == "sgd":
            return optim.SGD(params, lr=lr, momentum=self.args.get("momentum", 0.9), weight_decay=weight_decay)
        if opt_name == "adamw":
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        return optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=tuple(self.args.get("opt_betas", (0.9, 0.999))))

    def _make_scheduler(self, optimizer, epochs):
        sched = self.args.get("sched", self.args.get("scheduler", "constant"))
        if sched == "constant":
            return None
        if sched == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=self.args.get("min_lr", 1e-5)
            )
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.args.get("milestones", [10]),
            gamma=self.args.get("decay_rate", self.args.get("lr_decay", 0.1)),
        )

    def _mask_logits(self, logits, task_id, current_only):
        if not self.args.get("train_mask", True):
            return logits
        if current_only:
            mask = self.class_mask[task_id]
        else:
            mask = []
            for idx in range(task_id + 1):
                mask.extend(self.class_mask[idx])
        not_mask = np.setdiff1d(np.arange(self.nb_classes), mask)
        not_mask = torch.tensor(not_mask, dtype=torch.int64, device=logits.device)
        return logits.index_fill(dim=1, index=not_mask, value=float("-inf"))

    def _prompt_id_from_original(self, inputs, task_id):
        output = self.original_model(inputs)
        logits = self._mask_logits(output["logits"], task_id, current_only=False)
        pred = torch.max(logits, dim=1)[1]
        prompt_id = torch.tensor(
            [self.target_task_map[int(cls_id)] for cls_id in pred.detach().cpu()],
            device=inputs.device,
            dtype=torch.long,
        )
        return prompt_id.unsqueeze(-1)

    def _train_original(self, train_loader):
        epochs = self.args.get("tii_epochs", self.args.get("original_epochs", self.args.get("epochs", 5)))
        lr = self.args.get("tii_lr", self.args.get("lr", 5e-4))
        optimizer = self._make_optimizer(
            filter(lambda p: p.requires_grad, self.original_model.parameters()),
            lr=lr,
            weight_decay=self.args.get("weight_decay", 0.0),
        )
        scheduler = self._make_scheduler(optimizer, epochs)
        criterion = nn.CrossEntropyLoss().to(self._device)

        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self.original_model.train()
            losses, correct, total = 0.0, 0, 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                output = self.original_model(inputs)
                logits = self._mask_logits(output["logits"], self._cur_task, current_only=True)
                loss = criterion(logits, targets)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.original_model.parameters(), self.args.get("clip_grad", 1.0))
                optimizer.step()

                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = "TII Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task, epoch + 1, epochs, losses / len(train_loader), train_acc
            )
            prog_bar.set_description(info)
        logging.info(info)

    def _prompt_params(self):
        if self.args.get("larger_prompt_lr", False):
            prompt_params = [
                p for name, p in self._network.named_parameters() if "prompt" in name and p.requires_grad
            ]
            base_params = [
                p for name, p in self._network.named_parameters() if "prompt" not in name and p.requires_grad
            ]
            return [
                {"params": prompt_params, "lr": self.args.get("lr", 5e-4), "weight_decay": self.args.get("weight_decay", 0.0)},
                {
                    "params": base_params,
                    "lr": self.args.get("lr", 5e-4) * 0.1,
                    "weight_decay": self.args.get("weight_decay", 0.0),
                },
            ]
        return filter(lambda p: p.requires_grad, self._network.parameters())

    def _transfer_prompt_params(self):
        if self._cur_task == 0 or not self.args.get("prompt_pool", True):
            return
        top_k = self.args.get("top_k", 1)
        prev_start = (self._cur_task - 1) * top_k
        prev_end = self._cur_task * top_k
        cur_start = prev_end
        cur_end = (self._cur_task + 1) * top_k
        if cur_end > self.args.get("size", self.args["nb_tasks"]):
            return

        with torch.no_grad():
            if self.args.get("shared_prompt_pool", True):
                if self.args.get("use_prefix_tune_for_e_prompt", True):
                    self._network.e_prompt.prompt[:, :, cur_start:cur_end].copy_(
                        self._network.e_prompt.prompt[:, :, prev_start:prev_end]
                    )
                else:
                    self._network.e_prompt.prompt[:, cur_start:cur_end].copy_(
                        self._network.e_prompt.prompt[:, prev_start:prev_end]
                    )
            if self.args.get("shared_prompt_key", False):
                self._network.e_prompt.prompt_key[cur_start:cur_end].copy_(
                    self._network.e_prompt.prompt_key[prev_start:prev_end]
                )

    def _apply_prompt_momentum(self):
        momentum = self.args.get("prompt_momentum", 0.01)
        if momentum <= 0 or self._cur_task == 0 or not self.args.get("use_prefix_tune_for_e_prompt", True):
            return
        top_k = self.args.get("top_k", 1)
        cur_start = self._cur_task * top_k
        cur_end = (self._cur_task + 1) * top_k
        if cur_end > self.args.get("size", self.args["nb_tasks"]):
            return
        with torch.no_grad():
            current = self._network.e_prompt.prompt[:, :, cur_start:cur_end].detach().clone()
            previous = self._network.e_prompt.prompt[:, :, :cur_start].detach().clone().mean(dim=2, keepdim=True)
            self._network.e_prompt.prompt[:, :, cur_start:cur_end].copy_(
                (1 - momentum) * current + momentum * previous
            )

    def _train_prompt(self, train_loader):
        epochs = self.args.get("epochs", 5)
        optimizer = self._make_optimizer(
            self._prompt_params(), lr=self.args.get("lr", 5e-4), weight_decay=self.args.get("weight_decay", 0.0)
        )
        scheduler = self._make_scheduler(optimizer, epochs)
        criterion = nn.CrossEntropyLoss().to(self._device)

        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            self.original_model.eval()
            losses, correct, total = 0.0, 0, 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                with torch.no_grad():
                    prompt_id = self._prompt_id_from_original(inputs, self._cur_task)

                output = self._network(
                    inputs,
                    task_id=self._cur_task,
                    prompt_id=prompt_id,
                    train=True,
                    prompt_momentum=self.args.get("prompt_momentum", 0.01),
                )
                logits = self._mask_logits(output["logits"], self._cur_task, current_only=True)
                loss = criterion(logits, targets)
                loss = loss + self._orth_loss(output["pre_logits"])

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._network.parameters(), self.args.get("clip_grad", 1.0))
                optimizer.step()

                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = "HiDeP Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task, epoch + 1, epochs, losses / len(train_loader), train_acc
            )
            prog_bar.set_description(info)
        logging.info(info)

    def _orth_loss(self, features):
        reg = self.args.get("reg", 0.01)
        if reg == 0:
            return features.new_tensor(0.0)
        sample_mean = []
        for value in self.cls_mean.values():
            if isinstance(value, list):
                sample_mean.extend(value)
            else:
                sample_mean.append(value)
        if sample_mean:
            means = torch.stack([v.to(features.device).float() for v in sample_mean], dim=0)
            matrix = torch.cat([means, features], dim=0)
        else:
            matrix = features
        sim = torch.matmul(matrix, matrix.t()) / 0.8
        labels = torch.arange(sim.shape[0], device=features.device)
        return reg * F.cross_entropy(sim, labels)

    @torch.no_grad()
    def _compute_mean(self, train_loader):
        self._network.eval()
        features, labels = [], []
        for _, inputs, targets in train_loader:
            inputs = inputs.to(self._device)
            output = self._network(inputs, task_id=self._cur_task, train=True)
            features.append(output["pre_logits"].detach().cpu())
            labels.append(targets.cpu())
        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)

        for cls_id in self.class_mask[self._cur_task]:
            cls_features = features[labels == cls_id]
            if cls_features.numel() == 0:
                continue
            if self.args.get("ca_storage_efficient_method", "multi-centroid") == "multi-centroid":
                n_clusters = min(self.args.get("n_centroids", 10), len(cls_features))
                if n_clusters <= 1:
                    self.cls_mean[cls_id] = [cls_features.mean(dim=0).to(self._device)]
                    self.cls_cov[cls_id] = [cls_features.var(dim=0, unbiased=False).to(self._device) + 1e-4]
                    continue
                kmeans = KMeans(n_clusters=n_clusters, n_init=10)
                np_features = cls_features.numpy()
                cluster_labels = kmeans.fit_predict(np_features)
                means, variances = [], []
                for cluster_id in range(n_clusters):
                    cluster_data = cls_features[cluster_labels == cluster_id]
                    means.append(cluster_data.mean(dim=0).to(self._device))
                    variances.append(cluster_data.var(dim=0, unbiased=False).to(self._device) + 1e-4)
                self.cls_mean[cls_id] = means
                self.cls_cov[cls_id] = variances
            else:
                self.cls_mean[cls_id] = cls_features.mean(dim=0).to(self._device)
                self.cls_cov[cls_id] = torch.cov(cls_features.T).to(self._device) + (
                    torch.eye(cls_features.shape[1], device=self._device) * 1e-4
                )

    def _train_task_adaptive_prediction(self):
        param_list = [
            p for name, p in self._network.named_parameters() if p.requires_grad and "prompt" not in name
        ]
        if not param_list:
            return
        optimizer = optim.SGD(
            [{"params": param_list, "lr": self.args.get("ca_lr", 0.005), "weight_decay": self.args.get("weight_decay", 0.0)}],
            lr=self.args.get("ca_lr", 0.005),
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.get("crct_epochs", 30))
        criterion = nn.CrossEntropyLoss().to(self._device)

        old_class_count = sum(len(self.class_mask[i]) for i in range(self._cur_task))
        num_sampled_pcls = self.batch_size * 5
        for _ in range(self.args.get("crct_epochs", 30)):
            sampled_data, sampled_label = [], []
            for task_id in range(self._cur_task + 1):
                for cls_id in self.class_mask[task_id]:
                    means = self.cls_mean.get(cls_id)
                    variances = self.cls_cov.get(cls_id)
                    if means is None:
                        continue
                    if not isinstance(means, list):
                        means = [means]
                        variances = [torch.diag(variances)]
                    for mean, var in zip(means, variances):
                        if torch.mean(var.float()) == 0:
                            continue
                        cov = torch.diag(var.float()) + 1e-4 * torch.eye(mean.shape[0], device=self._device)
                        dist = MultivariateNormal(mean.float().to(self._device), cov)
                        sampled_data.append(dist.sample(sample_shape=(num_sampled_pcls,)))
                        sampled_label.extend([cls_id] * num_sampled_pcls)
            if not sampled_data:
                return
            inputs = torch.cat(sampled_data, dim=0).float().to(self._device)
            targets = torch.tensor(sampled_label, dtype=torch.long, device=self._device)
            indexes = torch.randperm(inputs.size(0), device=self._device)
            inputs, targets = inputs[indexes], targets[indexes]

            for batch_id in range(max(1, old_class_count)):
                start = batch_id * num_sampled_pcls
                end = start + num_sampled_pcls
                if start >= inputs.shape[0]:
                    break
                output = self._network(inputs[start:end], fc_only=True)
                logits = self._mask_logits(output["logits"], self._cur_task, current_only=False)
                loss = criterion(logits, targets[start:end])
                if not math.isfinite(loss.item()):
                    raise ValueError("Non-finite classifier alignment loss: {}".format(loss.item()))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)
        return cnn_accy, None

    def _evaluate(self, y_pred, y_true):
        ret = {}
        top1_pred = y_pred.T[0] if y_pred.ndim > 1 else y_pred
        grouped = accuracy(top1_pred, y_true, self._known_classes, self.args["init_cls"], self.args["increment"])
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        if y_pred.ndim > 1:
            ret["top{}".format(self.topk)] = np.around(
                (y_pred.T == np.tile(y_true, (y_pred.shape[1], 1))).sum() * 100 / len(y_true), decimals=2
            )
        else:
            ret["top{}".format(self.topk)] = ret["top1"]
        if "top5" not in ret:
            ret["top5"] = ret["top{}".format(self.topk)]
        return ret

    def _eval_cnn(self, loader):
        self._network.eval()
        self.original_model.eval()
        y_pred, y_true = [], []
        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            with torch.no_grad():
                prompt_id = self._prompt_id_from_original(inputs, self._cur_task)
                output = self._network(inputs, task_id=self._cur_task, prompt_id=prompt_id)
                logits = self._mask_logits(output["logits"], self._cur_task, current_only=False)
            predicts = torch.topk(logits, k=min(self.topk, logits.shape[1]), dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)
            with torch.no_grad():
                prompt_id = self._prompt_id_from_original(inputs, self._cur_task)
                outputs = model(inputs, task_id=self._cur_task, prompt_id=prompt_id)["logits"]
                outputs = self._mask_logits(outputs, self._cur_task, current_only=False)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts == targets).cpu().sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)
