"""Two-stage training with held-out validation and epoch-level restart."""

from collections import defaultdict
from pathlib import Path
import hashlib
import json
import math
import random

import numpy as np
import torch

from .encoder import CompactEvidencePCA
from .evaluation import macro
from .io import load_npz, save_checkpoint, save_npz, sha256, write_json
from .objective import components
from .refiner import FreeG
from .runtime import Predictor, configure_torch, tensors


def training_steps(rows, config, stage, epoch):
    seed = config["seed"] + 1009 * epoch
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[row["anatomy_id"]].append(row)
    if stage == "encoder":
        for group in groups.values():
            rng.shuffle(group)
        interleaved = [
            groups[name][index]
            for index in range(max(map(len, groups.values())))
            for name in sorted(groups)
            if index < len(groups[name])
        ]
        size = config["encoder_effective_batch"]
        return [
            [[row] for row in interleaved[start : start + size]]
            for start in range(0, len(interleaved), size)
        ]

    size = config["refiner_microbatch"]
    microbatches = {}
    for name, group in sorted(groups.items()):
        rng.shuffle(group)
        microbatches[name] = [
            group[start : start + size] for start in range(0, len(group), size)
        ]
    interleaved = [
        microbatches[name][index]
        for index in range(max(map(len, microbatches.values())))
        for name in sorted(microbatches)
        if index < len(microbatches[name])
    ]
    size = config["refiner_accumulation"]
    return [
        interleaved[start : start + size] for start in range(0, len(interleaved), size)
    ]


def prepare_inputs(train, validation, folder, config, device, encoder):
    moments = defaultdict(list)
    parent_groups = defaultdict(list)
    predictor = Predictor(encoder, config, device) if encoder is not None else None
    for data in (train, validation):
        for row in data.rows:
            anatomy = data.geometry(row)
            path = folder / "inputs" / f"{row['sample_id']}.npz"
            if not path.exists():
                points = data.points(row)
                if predictor is None:
                    inputs = anatomy.encoder_inputs(points)
                else:
                    _, fitted = predictor.upstream(anatomy, points)
                    inputs = anatomy.refiner_inputs(points, fitted)
                    inputs["raw"] = inputs["raw"].astype(np.float16)
                save_npz(path, **inputs)
            if predictor is not None and data is train:
                raw = load_npz(path)["raw"].astype(np.float64)
                parent = (row["anatomy_id"], row["parent_id"])
                moments[parent].append((raw.mean(0), (raw**2).mean(0)))
    if predictor is None:
        return None
    for (name, _), parent in moments.items():
        parent_groups[name].append(np.mean(parent, axis=0))
    mean, second_moment = np.mean(
        [np.mean(values, axis=0) for values in parent_groups.values()], axis=0
    )
    scale = np.sqrt(np.maximum(second_moment - mean**2, 0)).clip(
        config["normalization_minimum_scale"]
    )
    result = dict(mean=mean.tolist(), scale=scale.tolist(), training_only=True)
    write_json(folder / "normalization.json", result)
    return result


def forward(model, stage, data, rows, folder, device):
    geometry = data.geometry(rows[0])
    inputs = [load_npz(folder / "inputs" / f"{row['sample_id']}.npz") for row in rows]
    x = tensors(
        {key: np.stack([value[key] for value in inputs]) for key in inputs[0]}, device
    )
    if stage == "encoder":
        graph = geometry.encoder_graph(device)
        return model(x, graph)["displacement_mm"]
    x["raw"] = x["raw"].float()
    return model(x, geometry.refiner_graph(device))[0]


def objective(model, stage, data, rows, folder, device, config):
    predictions = forward(model, stage, data, rows, folder, device)
    targets = torch.as_tensor(
        np.stack([data.target(row) for row in rows]), device=device
    )
    identity = torch.tensor([row["is_identity"] for row in rows], device=device)
    return components(predictions, targets, identity, config["loss"])["total"]


@torch.inference_mode()
def validate(model, stage, dataset, folder, device, config):
    model.eval()
    records = []
    for row in dataset.rows:
        loss = float(objective(model, stage, dataset, [row], folder, device, config)[0])
        if not math.isfinite(loss):
            raise FloatingPointError("Nonfinite validation objective.")
        records.append(row | {"metrics": {"loss": loss}})
    primary = set(config["primary_validation_anatomies"])
    selected = [row for row in records if row["anatomy_id"] in primary]
    return {
        "selection_loss": macro(selected, "loss"),
        "validation_loss": macro(records, "loss"),
    }


def training_signature(train, validation, settings, encoder, device):
    paths = set()
    for data in (train, validation):
        for row in data.rows:
            paths.update(
                data.root / row[key]
                for key in ("anatomy", "points", "target")
                if row[key]
            )
    identity = {
        "config": settings,
        "device": str(device),
        "torch": torch.__version__,
        "training": train.rows,
        "validation": validation.rows,
        "files": [sha256(path) for path in sorted(paths)],
        "encoder": sha256(encoder) if encoder is not None else None,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def train_stage(stage, train, validation, output, config, device, encoder=None):
    device = configure_torch(device)
    train_names = {row["anatomy_id"] for row in train.rows}
    val_names = {row["anatomy_id"] for row in validation.rows}
    ids = {row["sample_id"] for row in train.rows}
    if (
        train_names & val_names
        or ids & {row["sample_id"] for row in validation.rows}
        or any(row["partition"] != "train" for row in train.rows)
        or any(row["partition"] != "validation" for row in validation.rows)
    ):
        raise ValueError(
            "Use disjoint training and validation anatomies and observations."
        )
    if (
        not set(config["primary_validation_anatomies"])
        or not set(config["primary_validation_anatomies"]) <= val_names
    ):
        raise ValueError(
            "All configured checkpoint-selection anatomies must be present in validation."
        )
    if stage == "refiner" and encoder is None:
        raise ValueError("Refiner training requires a frozen encoder checkpoint.")

    signature = training_signature(train, validation, config, encoder, device)
    folder = Path(output) / stage
    protocol = folder / "protocol.json"
    if protocol.exists():
        if json.loads(protocol.read_text())["signature"] != signature:
            raise ValueError(
                f"The data, model or configuration changed. Choose a new run directory: {folder}"
            )
    else:
        write_json(protocol, {"signature": signature, "config": config})

    normalization = prepare_inputs(train, validation, folder, config, device, encoder)
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = (CompactEvidencePCA() if stage == "encoder" else FreeG(normalization)).to(
        device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config[f"{stage}_weight_decay"],
    )
    maximum = config[f"{stage}_epochs"]
    if maximum < 1:
        raise ValueError("The training budget must be at least one epoch.")

    resume = folder / "resume.pt"
    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history, best = state["history"], state["best"]
    else:
        report = validate(model, stage, validation, folder, device, config)
        history = [dict(epoch=0, **report)]
        best = dict(
            selected_epoch=0,
            score=report["selection_loss"],
            model={
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
        )

    for epoch in range(history[-1]["epoch"], maximum):
        fraction = epoch / max(1, maximum - 1)
        floor, initial = config["minimum_learning_rate"], config["learning_rate"]
        rate = floor + 0.5 * (initial - floor) * (1 + math.cos(math.pi * fraction))
        for group in optimizer.param_groups:
            group["lr"] = rate
        model.train()
        train_total = 0.0
        for update in training_steps(train.rows, config, stage, epoch):
            count = sum(map(len, update))
            optimizer.zero_grad(set_to_none=True)
            for batch in update:
                value = objective(
                    model, stage, train, batch, folder, device, config
                ).sum()
                if not torch.isfinite(value):
                    raise FloatingPointError("Nonfinite training objective.")
                (value / count).backward()
                train_total += float(value.detach())
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config[f"{stage}_gradient_clip"],
                error_if_nonfinite=True,
            )
            optimizer.step()
        report = validate(model, stage, validation, folder, device, config)
        history.append(
            dict(
                epoch=epoch + 1,
                learning_rate=rate,
                train_loss=train_total / len(train.rows),
                **report,
            )
        )
        if report["selection_loss"] < best["score"]:
            best = dict(
                selected_epoch=epoch + 1,
                score=report["selection_loss"],
                model={
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
            )
        save_checkpoint(
            resume,
            dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                best=best,
                history=history,
            ),
        )
        print(
            f"{stage} epoch {epoch + 1}/{maximum} | validation {report['selection_loss']:.5f}",
            flush=True,
        )

    path = folder / "best.pt"
    if not path.exists():
        save_checkpoint(
            path,
            best
            | dict(
                normalization=normalization,
                seed=seed,
                protocol=signature,
                test_used_for_selection=False,
            ),
        )
    write_json(folder / "history.json", history)
    return path
