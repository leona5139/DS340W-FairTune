import os

os.environ["TORCH_HOME"] = os.path.dirname(os.getcwd())

import datetime
import itertools
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torchvision
import yaml
from pprint import pprint
from torch import nn
from torch.utils.data.dataloader import default_collate

from utilities import presets
from utilities import transforms
from utilities.utils import *
from utilities.training_utils import *
from parse_args import *
from utilities.sampler import RASampler

############################################################################
# Fairness-aware sensitivity ranking (SPT/GPS-style one-shot scorer) + a
# nested k-sweep. Replaces search_mask.py's Optuna/TPE search over the
# 36-bit auto_peft1 mask space with: one gradient pass to rank the 36 mask
# components by sensitivity, then only len(args.k_schedule) full trainings
# (nested masks M_1 subset M_2 subset ... subset M_36) instead of 20-50+
# blind trials. See fairtune-speedup-plan.md for the design.
#
# The mask this script produces is saved in the exact same format/location
# as search_mask.py's ("<model>/<dataset>/Optuna_Masks/<sens_attribute>/
# *.npy", auto_peft1's 36-bit np.int8 layout) so finetune_with_mask.py and
# orchestration/common.sh's discover_mask() need zero changes.
#
# `create_model`/`define_dataloaders` below are copied verbatim from
# search_mask.py (its `create_results_df` isn't needed here -- results are
# collected directly into the k-sweep table instead of a per-row CSV), and
# the eval-dispatch block inside run_k_sweep_trial() is copied verbatim from
# search_mask.py's objective() (post-training evaluation section). This is
# deliberate duplication, not an
# oversight: search_mask.py is the TPE baseline this method is benchmarked
# against, and it must stay provably unmodified. Do not "fix" the unweighted
# criterion below (class weights are computed but never passed into
# nn.CrossEntropyLoss) -- that mirrors a pre-existing bug in
# search_mask.py's objective(), and replicating it keeps the comparison
# between this script and search_mask.py apples-to-apples.
############################################################################

COMPONENT_TYPE_NAMES = {0: "layernorm", 1: "attention", 2: "mlp"}


def create_model(args):
    print("Creating model")
    print("TUNING METHOD: ", args.tuning_method)
    model = utils.get_timm_model(args.model, num_classes=args.num_classes)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in [
            "class_token",
            "position_embedding",
            "relative_position_bias_table",
        ]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))

    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay
        if len(custom_keys_weight_decay) > 0
        else None,
    )

    return model, parameters


def define_dataloaders(args):
    with open("config.yaml") as file:
        yaml_data = yaml.safe_load(file)

    print("Creating dataset")
    (
        dataset,
        dataset_val,
        dataset_test,
        train_sampler,
        val_sampler,
        test_sampler,
    ) = get_fairness_data(args, yaml_data)

    args.num_classes = len(dataset.classes)
    print("DATASET: ", args.dataset)
    print("Size of training dataset: ", len(dataset))
    print("Size of validation dataset: ", len(dataset_val))
    print("Size of test dataset: ", len(dataset_test))
    print("Number of classes: ", args.num_classes)
    pprint(dataset.class_to_idx)

    collate_fn = None
    mixup_transforms = get_mixup_transforms(args)

    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)

        def collate_fn(batch):
            return mixupcutmix(*default_collate(batch))

    print("Creating data loaders")

    if args.dataset == "papila":
        drop_last = False
    else:
        drop_last = True

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    return data_loader, data_loader_val, data_loader_test


def _to_float(x):
    try:
        return float(x.item())
    except AttributeError:
        return float(x)


def _is_higher_better(objective_metric):
    if objective_metric in ("acc_diff", "auc_diff", "max_loss"):
        return False
    elif objective_metric in ("min_acc", "min_auc", "overall_acc", "overall_auc"):
        return True
    else:
        raise NotImplementedError(f"Objective metric not implemented: {objective_metric}")


def _component_params(model, block_idx, bit_idx):
    """Returns the parameters belonging to mask component (block_idx, bit_idx),
    using the exact auto_peft1 mapping (utilities/utils.py:304-339):
    bit_idx 0 = norm1+norm2, 1 = attn, 2 = mlp."""
    block = model.blocks[block_idx]
    if bit_idx == 0:
        return list(block.norm1.parameters()) + list(block.norm2.parameters())
    elif bit_idx == 1:
        return list(block.attn.parameters())
    elif bit_idx == 2:
        return list(block.mlp.parameters())
    else:
        raise ValueError(f"bit_idx must be 0, 1, or 2 (auto_peft1 format); got {bit_idx}")


def compute_sensitivity_scores(args, model, data_loader, device):
    """One-shot component sensitivity scoring (SPT/GPS-style Taylor
    sensitivity, computed on worst-group loss by default -- the
    fairness-aware piece this method adds on top of vanilla SPT/GPS).

    Returns (ranking, scores_df): `ranking` is a list of the 36 auto_peft1
    component indices sorted best-to-worst by score (or a random permutation
    if args.rank_method == "random"); `scores_df` has one row per component.
    """
    num_blocks = len(model.blocks)
    mask_length = num_blocks * 3

    if args.rank_method == "random":
        rng = np.random.RandomState(args.sensitivity_seed)
        ranking = rng.permutation(mask_length).tolist()
        rank_of = {component_idx: rank + 1 for rank, component_idx in enumerate(ranking)}
        rows = []
        for idx in range(mask_length):
            block_idx, bit_idx = idx // 3, idx % 3
            param_count = sum(p.numel() for p in _component_params(model, block_idx, bit_idx))
            rows.append(
                {
                    "component_idx": idx,
                    "block_idx": block_idx,
                    "bit_idx": bit_idx,
                    "component_type": COMPONENT_TYPE_NAMES[bit_idx],
                    "raw_score": float("nan"),
                    "param_count": param_count,
                    "normalized_score": float("nan"),
                    "rank": rank_of[idx],
                }
            )
        return ranking, pd.DataFrame(rows)

    model.eval()
    model.zero_grad()
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, reduction="none")

    num_available = len(data_loader)
    num_batches = (
        args.sensitivity_num_batches if args.sensitivity_num_batches is not None else num_available
    )
    if num_batches > num_available:
        print(
            f"WARNING: --sensitivity_num_batches={num_batches} exceeds one epoch "
            f"(len(data_loader)={num_available}); wrapping, "
            f"{num_batches - num_available} batch(es) will be reused."
        )
        batch_iter = itertools.islice(itertools.cycle(data_loader), num_batches)
    else:
        batch_iter = itertools.islice(data_loader, num_batches)

    print(
        f"Computing {args.rank_method} sensitivity scores over {num_batches} batch(es) "
        f"({num_available} batch(es)/epoch available)"
    )

    batches_used = 0
    for image, target, sens_attr in batch_iter:
        image, target = image.to(device), target.to(device)
        output = model(image)
        per_sample_loss = criterion(output, target)

        if args.rank_method == "worst_group_sensitivity":
            # data/papila.py's __getitem__ returns a raw string ('M'/'F')
            # for gender, a raw int for age/intersectional -- default_collate
            # turns these into list[str] or a LongTensor respectively.
            # Normalizing to a plain list lets us group by raw value with no
            # per-attribute dispatch, unlike the rest of this codebase.
            sens_attr_list = (
                sens_attr.tolist() if torch.is_tensor(sens_attr) else list(sens_attr)
            )
            group_means = []
            for group in sorted(set(sens_attr_list), key=str):
                group_mask = torch.tensor(
                    [g == group for g in sens_attr_list], device=per_sample_loss.device
                )
                group_means.append(per_sample_loss[group_mask].mean())
            scoring_loss = torch.stack(group_means).max()
        elif args.rank_method == "erm_sensitivity":
            scoring_loss = per_sample_loss.mean()
        else:
            raise NotImplementedError(f"Unknown rank_method: {args.rank_method}")

        # No zero_grad() between batches: gradients accumulate (sum) across
        # the whole calibration set. A constant scale factor (sum vs.
        # average) can't change the resulting ranking, only the raw_score
        # magnitude -- see fairtune-speedup-plan.md's normalization ablation.
        scoring_loss.backward()
        batches_used += 1

    print(
        f"Sensitivity scoring pass complete ({batches_used} batches, "
        "gradients summed -- not averaged -- across batches)."
    )

    rows = []
    for idx in range(mask_length):
        block_idx, bit_idx = idx // 3, idx % 3
        params = _component_params(model, block_idx, bit_idx)
        raw_score = sum(
            ((p.grad * p.data) ** 2).sum().item() for p in params if p.grad is not None
        )
        param_count = sum(p.numel() for p in params)

        if args.sensitivity_normalize == "none":
            normalized_score = raw_score
        elif args.sensitivity_normalize == "param_count":
            normalized_score = raw_score / param_count
        elif args.sensitivity_normalize == "sqrt_param_count":
            normalized_score = raw_score / (param_count**0.5)
        else:
            raise NotImplementedError(
                f"Unknown sensitivity_normalize: {args.sensitivity_normalize}"
            )

        rows.append(
            {
                "component_idx": idx,
                "block_idx": block_idx,
                "bit_idx": bit_idx,
                "component_type": COMPONENT_TYPE_NAMES[bit_idx],
                "raw_score": raw_score,
                "param_count": param_count,
                "normalized_score": normalized_score,
            }
        )

    scores_df = pd.DataFrame(rows)
    scores_df["rank"] = (
        scores_df["normalized_score"].rank(ascending=False, method="first").astype(int)
    )
    ranking = scores_df.sort_values("rank")["component_idx"].tolist()

    model.zero_grad()
    return ranking, scores_df


def run_k_sweep_trial(args, ranking, k, data_loader, data_loader_val, device):
    """Trains one nested mask M_k (the top-k components by sensitivity
    score) to completion and evaluates it on the val set. Mirrors
    search_mask.py's objective() -- same model/optimizer/criterion
    construction, same training loop, same per-sens_attribute evaluation
    dispatch and objective_metric computation -- minus everything
    Optuna-specific (no trial.suggest_float("lr", ...), no
    trial.report/should_prune) and minus the redundant in-loop
    per-epoch evaluation+checkpointing objective() also does (which never
    fed into its returned value -- only the single post-training evaluation
    below does)."""
    mask_k = np.zeros(len(ranking), dtype=np.int8)
    mask_k[list(ranking[:k])] = 1
    print(f"k={k} | mask: {mask_k.tolist()}")

    model, parameters = create_model(args)
    utils.get_masked_model(model, "auto_peft1", mask=mask_k)

    trainable_params, all_param = utils.check_tunable_params(model, True)
    trainable_percentage = 100 * trainable_params / all_param

    model.to(device)

    # OL3I and Papila are highly imbalanced datasets. search_mask.py's
    # objective() computes class weights here but never passes them into
    # the criterion below -- a pre-existing bug, replicated deliberately
    # (see module docstring) to keep this a fair comparison against TPE.
    if args.compute_cw:
        if args.dataset == "ol3i":
            weight = torch.tensor([0.04361966711306677, 0.9563803328869332])
        elif args.dataset == "papila":
            weight = torch.tensor([0.20714285714285716, 0.7928571428571428])
        else:
            raise NotImplementedError("Class weights not calculated for this dataset")
        weight = weight.to(device)
        print("Using CW Loss with weights: ", weight)
    else:
        weight = None

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, reduction="none")
    ece_criterion = utils.ECELoss()
    optimizer = get_optimizer(args, parameters)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    lr_scheduler = get_lr_scheduler(args, optimizer)

    print(f"Start training (k={k})")
    start_time = time.time()
    model_ema = None

    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch_fairness(
            model,
            criterion,
            ece_criterion,
            optimizer,
            data_loader,
            device,
            epoch,
            args,
            model_ema,
            scaler,
        )
        lr_scheduler.step()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    # Obtaining the performance on val set
    print("Training finished | Evaluating on the val set")

    if args.sens_attribute == "gender":
        (
            val_acc,
            val_male_acc,
            val_female_acc,
            val_auc,
            val_male_auc,
            val_female_auc,
            val_loss,
            val_max_loss,
        ) = evaluate_fairness_gender(
            model,
            criterion,
            ece_criterion,
            data_loader_val,
            args=args,
            device=device,
        )

        max_acc = max(val_male_acc, val_female_acc)
        min_acc = min(val_male_acc, val_female_acc)
        acc_diff = abs(max_acc - min_acc)

        max_auc = max(val_male_auc, val_female_auc)
        min_auc = min(val_male_auc, val_female_auc)
        auc_diff = abs(max_auc - min_auc)

        print("\n")
        print("Val Male Accuracy: ", val_male_acc)
        print("Val Female Accuracy: ", val_female_acc)
        print("Difference in sub-group performance: ", acc_diff)
        print("\n")
        print("Val Male AUC: ", val_male_auc)
        print("Val Female AUC: ", val_female_auc)
        print("Difference in sub-group performance (AUC): ", auc_diff)

    elif args.sens_attribute == "skin_type":
        if args.skin_type == "multi":
            (
                val_acc,
                val_acc_type0,
                val_acc_type1,
                val_acc_type2,
                val_acc_type3,
                val_acc_type4,
                val_acc_type5,
                val_auc,
                val_auc_type0,
                val_auc_type1,
                val_auc_type2,
                val_auc_type3,
                val_auc_type4,
                val_auc_type5,
                val_loss,
                val_max_loss,
            ) = evaluate_fairness_skin_type(
                model,
                criterion,
                ece_criterion,
                data_loader_val,
                args=args,
                device=device,
            )

            max_acc = max(
                val_acc_type0,
                val_acc_type1,
                val_acc_type2,
                val_acc_type3,
                val_acc_type4,
                val_acc_type5,
            )
            min_acc = min(
                val_acc_type0,
                val_acc_type1,
                val_acc_type2,
                val_acc_type3,
                val_acc_type4,
                val_acc_type5,
            )
            acc_diff = abs(max_acc - min_acc)

            max_auc = max(
                val_auc_type0,
                val_auc_type1,
                val_auc_type2,
                val_auc_type3,
                val_auc_type4,
                val_auc_type5,
            )
            min_auc = min(
                val_auc_type0,
                val_auc_type1,
                val_auc_type2,
                val_auc_type3,
                val_auc_type4,
                val_auc_type5,
            )
            auc_diff = abs(max_auc - min_auc)

            print("\n")
            print("Val Type 0 Accuracy: ", val_acc_type0)
            print("Val Type 1 Accuracy: ", val_acc_type1)
            print("Val Type 2 Accuracy: ", val_acc_type2)
            print("Val Type 3 Accuracy: ", val_acc_type3)
            print("Val Type 4 Accuracy: ", val_acc_type4)
            print("Val Type 5 Accuracy: ", val_acc_type5)
            print("Difference in sub-group performance (Accuracy): ", acc_diff)

            print("\n")
            print("Val Type 0 AUC: ", val_auc_type0)
            print("Val Type 1 AUC: ", val_auc_type1)
            print("Val Type 2 AUC: ", val_auc_type2)
            print("Val Type 3 AUC: ", val_auc_type3)
            print("Val Type 4 AUC: ", val_auc_type4)
            print("Val Type 5 AUC: ", val_auc_type5)
            print("Difference in sub-group performance (AUC): ", auc_diff)

        elif args.skin_type == "binary":
            (
                val_acc,
                val_acc_type0,
                val_acc_type1,
                val_auc,
                val_auc_type0,
                val_auc_type1,
                val_loss,
                val_max_loss,
            ) = evaluate_fairness_skin_type_binary(
                model,
                criterion,
                ece_criterion,
                data_loader_val,
                args=args,
                device=device,
            )

            max_acc = max(val_acc_type0, val_acc_type1)
            min_acc = min(val_acc_type0, val_acc_type1)
            acc_diff = abs(max_acc - min_acc)

            max_auc = max(val_auc_type0, val_auc_type1)
            min_auc = min(val_auc_type0, val_auc_type1)
            auc_diff = abs(max_auc - min_auc)

            print("\n")
            print("Val Type 0 Accuracy: ", val_acc_type0)
            print("Val Type 1 Accuracy: ", val_acc_type1)
            print("Difference in sub-group performance (Accuracy): ", acc_diff)

            print("\n")
            print("Overall Val AUC: ", val_auc)
            print("Val Type 0 AUC: ", val_auc_type0)
            print("Val Type 1 AUC: ", val_auc_type1)
            print("Difference in sub-group performance (AUC): ", auc_diff)

    elif args.sens_attribute == "age":
        if args.age_type == "multi":
            (
                val_acc,
                acc_age0_avg,
                acc_age1_avg,
                acc_age2_avg,
                acc_age3_avg,
                acc_age4_avg,
                val_auc,
                auc_age0_avg,
                auc_age1_avg,
                auc_age2_avg,
                auc_age3_avg,
                auc_age4_avg,
                val_loss,
                val_max_loss,
            ) = evaluate_fairness_age(
                model,
                criterion,
                ece_criterion,
                data_loader_val,
                args=args,
                device=device,
            )

            max_acc = max(
                acc_age0_avg, acc_age1_avg, acc_age2_avg, acc_age3_avg, acc_age4_avg
            )
            min_acc = min(
                acc_age0_avg, acc_age1_avg, acc_age2_avg, acc_age3_avg, acc_age4_avg
            )
            acc_diff = abs(max_acc - min_acc)

            max_auc = max(
                auc_age0_avg, auc_age1_avg, auc_age2_avg, auc_age3_avg, auc_age4_avg
            )
            min_auc = min(
                auc_age0_avg, auc_age1_avg, auc_age2_avg, auc_age3_avg, auc_age4_avg
            )
            auc_diff = abs(max_auc - min_auc)

            print("\n")
            print("val Age Group 0 Accuracy: ", acc_age0_avg)
            print("val Age Group 1 Accuracy: ", acc_age1_avg)
            print("val Age Group 2 Accuracy: ", acc_age2_avg)
            print("val Age Group 3 Accuracy: ", acc_age3_avg)
            print("val Age Group 4 Accuracy: ", acc_age4_avg)
            print("Difference in sub-group performance (Accuracy): ", acc_diff)

            print("\n")
            print("val Age Group 0 AUC: ", auc_age0_avg)
            print("val Age Group 1 AUC: ", auc_age1_avg)
            print("val Age Group 2 AUC: ", auc_age2_avg)
            print("val Age Group 3 AUC: ", auc_age3_avg)
            print("val Age Group 4 AUC: ", auc_age4_avg)
            print("Difference in sub-group performance (AUC): ", auc_diff)

        elif args.age_type == "binary":
            (
                val_acc,
                acc_age0_avg,
                acc_age1_avg,
                val_auc,
                auc_age0_avg,
                auc_age1_avg,
                val_loss,
                val_max_loss,
            ) = evaluate_fairness_age_binary(
                model,
                criterion,
                ece_criterion,
                data_loader_val,
                args=args,
                device=device,
            )

            max_acc = max(acc_age0_avg, acc_age1_avg)
            min_acc = min(acc_age0_avg, acc_age1_avg)
            acc_diff = abs(max_acc - min_acc)

            max_auc = max(auc_age0_avg, auc_age1_avg)
            min_auc = min(auc_age0_avg, auc_age1_avg)
            auc_diff = abs(max_auc - min_auc)

            print("\n")
            print("val Age Group 0 Accuracy: ", acc_age0_avg)
            print("val Age Group 1 Accuracy: ", acc_age1_avg)
            print("Difference in sub-group performance (Accuracy): ", acc_diff)

            print("\n")
            print("val Age Group 0 AUC: ", auc_age0_avg)
            print("val Age Group 1 AUC: ", auc_age1_avg)
            print("Difference in sub-group performance (AUC): ", auc_diff)

        else:
            raise NotImplementedError(
                "Age type not supported. Choose from 'multi' or 'binary'"
            )

    elif args.sens_attribute == "race":
        if args.cal_equiodds:
            (
                val_acc,
                acc_race0_avg,
                acc_race1_avg,
                val_auc,
                auc_race0_avg,
                auc_race1_avg,
                val_loss,
                val_max_loss,
                equiodds_diff,
                equiodds_ratio,
                dpd,
                dpr,
            ) = evaluate_fairness_race_binary(
                model,
                criterion,
                ece_criterion,
                data_loader_val,
                args=args,
                device=device,
            )
        else:
            (
                val_acc,
                acc_race0_avg,
                acc_race1_avg,
                val_auc,
                auc_race0_avg,
                auc_race1_avg,
                val_loss,
                val_max_loss,
            ) = evaluate_fairness_race_binary(
                model,
                criterion,
                ece_criterion,
                data_loader_val,
                args=args,
                device=device,
            )

        max_acc = max(acc_race0_avg, acc_race1_avg)
        min_acc = min(acc_race0_avg, acc_race1_avg)
        acc_diff = abs(max_acc - min_acc)

        max_auc = max(auc_race0_avg, auc_race1_avg)
        min_auc = min(auc_race0_avg, auc_race1_avg)
        auc_diff = abs(max_auc - min_auc)

        print("\n")
        print("val Race Group 0 Accuracy: ", acc_race0_avg)
        print("val Race Group 1 Accuracy: ", acc_race1_avg)
        print("Difference in sub-group performance (Accuracy): ", acc_diff)

        print("\n")
        print("val Race Group 0 AUC: ", auc_race0_avg)
        print("val Race Group 1 AUC: ", auc_race1_avg)
        print("Difference in sub-group performance (AUC): ", auc_diff)

    elif args.sens_attribute == "age_sex":
        assert args.dataset == "chexpert"
        if args.cal_equiodds:
            (
                val_acc,
                val_acc_type0,
                val_acc_type1,
                val_acc_type2,
                val_acc_type3,
                val_auc,
                val_auc_type0,
                val_auc_type1,
                val_auc_type2,
                val_auc_type3,
                val_loss,
                val_max_loss,
                equiodds_diff,
                equiodds_ratio,
                dpd,
                dpr,
            ) = evaluate_fairness_age_sex(
                model,
                criterion,
                ece_criterion,
                data_loader_val,
                args=args,
                device=device,
            )
        else:
            (
                val_acc,
                val_acc_type0,
                val_acc_type1,
                val_acc_type2,
                val_acc_type3,
                val_auc,
                val_auc_type0,
                val_auc_type1,
                val_auc_type2,
                val_auc_type3,
                val_loss,
                val_max_loss,
            ) = evaluate_fairness_age_sex(
                model,
                criterion,
                ece_criterion,
                data_loader_val,
                args=args,
                device=device,
            )

        max_acc = max(val_acc_type0, val_acc_type1, val_acc_type2, val_acc_type3)
        min_acc = min(val_acc_type0, val_acc_type1, val_acc_type2, val_acc_type3)
        acc_diff = abs(max_acc - min_acc)

        max_auc = max(val_auc_type0, val_auc_type1, val_auc_type2, val_auc_type3)
        min_auc = min(val_auc_type0, val_auc_type1, val_auc_type2, val_auc_type3)
        auc_diff = abs(max_auc - min_auc)

        print("\n")
        print("val AgeSex Group 0 Accuracy: ", val_acc_type0)
        print("val AgeSex Group 1 Accuracy: ", val_acc_type1)
        print("val AgeSex Group 2 Accuracy: ", val_acc_type2)
        print("val AgeSex Group 3 Accuracy: ", val_acc_type3)
        print("Difference in sub-group performance (Accuracy): ", acc_diff)

        print("\n")
        print("val AgeSex Group 0 AUC: ", val_auc_type0)
        print("val AgeSex Group 1 AUC: ", val_auc_type1)
        print("val AgeSex Group 2 AUC: ", val_auc_type2)
        print("val AgeSex Group 3 AUC: ", val_auc_type3)

    elif args.sens_attribute == "intersectional":
        if args.cal_equiodds:
            (
                val_acc,
                val_acc_groups,
                val_auc,
                val_auc_groups,
                val_loss,
                val_max_loss,
                val_group_counts,
                equiodds_diff,
                equiodds_ratio,
                dpd,
                dpr,
            ) = evaluate_fairness_intersectional(
                model,
                criterion,
                ece_criterion,
                data_loader_val,
                args=args,
                device=device,
            )
        else:
            (
                val_acc,
                val_acc_groups,
                val_auc,
                val_auc_groups,
                val_loss,
                val_max_loss,
                val_group_counts,
            ) = evaluate_fairness_intersectional(
                model,
                criterion,
                ece_criterion,
                data_loader_val,
                args=args,
                device=device,
            )

        max_acc = max(val_acc_groups)
        min_acc = min(val_acc_groups)
        acc_diff = abs(max_acc - min_acc)

        max_auc = max(val_auc_groups)
        min_auc = min(val_auc_groups)
        auc_diff = abs(max_auc - min_auc)

        print("\n")
        print("val Group Accuracies: ", val_acc_groups)
        print("val Group (correct, count): ", val_group_counts)
        print("Difference in sub-group performance (Accuracy): ", acc_diff)

        print("\n")
        print("val Group AUCs: ", val_auc_groups)

    else:
        raise NotImplementedError("Sensitive attribute not implemented")

    print("Val overall accuracy: ", val_acc)
    print("Val Max Accuracy: ", round(max_acc, 3))
    print("Val Min Accuracy: ", round(min_acc, 3))
    print("Val Accuracy Difference: ", round(acc_diff, 3))
    print("Val loss: ", round(torch.mean(val_loss).item(), 3))
    print("Val max loss: ", round(val_max_loss.item(), 3))

    print("Val overall AUC: ", val_auc)
    print("Val Max AUC: ", round(max_auc, 3))
    print("Val Min AUC: ", round(min_auc, 3))
    print("Val AUC Difference: ", round(auc_diff, 3))

    if args.objective_metric == "acc_diff":
        metric_value = acc_diff
    elif args.objective_metric == "auc_diff":
        metric_value = auc_diff
    elif args.objective_metric == "min_acc":
        metric_value = min_acc
    elif args.objective_metric == "min_auc":
        metric_value = min_auc
    elif args.objective_metric == "max_loss":
        metric_value = val_max_loss
    elif args.objective_metric == "overall_acc":
        metric_value = val_acc
    elif args.objective_metric == "overall_auc":
        metric_value = val_auc
    else:
        raise NotImplementedError("Objective metric not implemented")

    return {
        "k": k,
        "objective_metric_value": _to_float(metric_value),
        "val_acc": _to_float(val_acc),
        "max_acc": _to_float(max_acc),
        "min_acc": _to_float(min_acc),
        "acc_diff": _to_float(acc_diff),
        "val_auc": _to_float(val_auc),
        "max_auc": _to_float(max_auc),
        "min_auc": _to_float(min_auc),
        "auc_diff": _to_float(auc_diff),
        "trainable_params": trainable_params,
        "trainable_percentage": trainable_percentage,
        "wall_clock_seconds": total_time,
    }


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    assert args.tuning_method == "auto_peft1", (
        "search_mask_sensitivity.py only supports --tuning_method auto_peft1 "
        "(the 36-bit block x {layernorm, attention, mlp} mask format used "
        "by finetune_with_mask.py and utilities.utils.auto_peft1); got: "
        f"{args.tuning_method}"
    )

    # No --seed exists anywhere else in this codebase; a single seed here
    # covers both the calibration-batch order and (since the train loader's
    # RandomSampler shares the global RNG stream, with no per-epoch state of
    # its own) the training-data order across the whole k-sweep.
    random.seed(args.sensitivity_seed)
    np.random.seed(args.sensitivity_seed)
    torch.manual_seed(args.sensitivity_seed)

    device = torch.device(args.device)

    args.train_fscl_classifier = False
    args.train_fscl_encoder = False

    if args.dev_mode:
        args.disable_checkpointing = True

    if "auc" in args.objective_metric:
        args.use_metric = "auc"

    args.distributed = False
    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    print("Building dataloaders (shared across the whole k-sweep, unlike search_mask.py's "
          "Optuna objective(), which rebuilds them every trial)")
    data_loader, data_loader_val, data_loader_test = define_dataloaders(args)

    print(f"\n=== Scoring the 36 mask components (rank_method={args.rank_method}) ===")
    scoring_model, _ = create_model(args)
    scoring_model.to(device)
    ranking, scores_df = compute_sensitivity_scores(args, scoring_model, data_loader, device)
    del scoring_model

    scores_savedir = os.path.join(args.model, args.dataset, "Sensitivity_Scores", args.sens_attribute)
    os.makedirs(scores_savedir, exist_ok=True)
    scores_path = os.path.join(scores_savedir, f"scores_{args.rank_method}.csv")
    scores_df.to_csv(scores_path, index=False)
    print(f"Saved sensitivity scores to {scores_path}")
    print("Component ranking (best to worst):", ranking)

    sweep_rows = []
    for k in args.k_schedule:
        print(f"\n=== k-sweep: k={k} (rank_method={args.rank_method}, objective_metric={args.objective_metric}) ===")
        sweep_rows.append(run_k_sweep_trial(args, ranking, k, data_loader, data_loader_val, device))

    sweep_df = pd.DataFrame(sweep_rows)
    results_savedir = os.path.join(args.model, args.dataset, "Sensitivity_Results", args.sens_attribute)
    os.makedirs(results_savedir, exist_ok=True)
    sweep_path = os.path.join(
        results_savedir, f"k_sweep_{args.rank_method}_{args.objective_metric}.csv"
    )
    sweep_df.to_csv(sweep_path, index=False)
    print(f"Saved k-sweep results to {sweep_path}")
    print(sweep_df.to_string(index=False))

    higher_is_better = _is_higher_better(args.objective_metric)
    best_idx = (
        sweep_df["objective_metric_value"].idxmax()
        if higher_is_better
        else sweep_df["objective_metric_value"].idxmin()
    )
    best_row = sweep_df.loc[best_idx]
    best_k = int(best_row["k"])
    best_value = best_row["objective_metric_value"]
    print(
        f"\nBest k: {best_k} ({args.objective_metric}={best_value:.4f}, "
        f"{'higher' if higher_is_better else 'lower'} is better)"
    )

    best_mask = np.zeros(len(ranking), dtype=np.int8)
    best_mask[list(ranking[:best_k])] = 1

    mask_savedir = os.path.join(args.model, args.dataset, "Optuna_Masks", args.sens_attribute)
    os.makedirs(mask_savedir, exist_ok=True)
    mask_path = os.path.join(
        mask_savedir,
        f"sensitivity_{args.rank_method}_best_mask_{args.objective_metric}_{best_value}.npy",
    )
    if not args.dev_mode:
        np.save(mask_path, best_mask)
        print(f"Saved best mask (k={best_k}) to {mask_path}")
    else:
        print(f"--dev_mode: not saving mask (would have saved to {mask_path})")
