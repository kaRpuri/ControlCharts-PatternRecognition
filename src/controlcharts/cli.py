"""Command line entry point: ``controlcharts <command>``."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from . import data as data_mod
from . import plots
from .config import Config
from .evaluate import evaluate, format_report
from .model import load_model
from .predict import analyze
from .preprocess import prepare
from .train import save_model, train

logger = logging.getLogger("controlcharts")

DEFAULT_CONFIG = "configs/default.yaml"


def _load_dataset(config: Config) -> data_mod.Dataset:
    """Load the configured dataset, with a clear error if it is missing."""
    path = Path(config.paths.dataset)
    if not path.is_file():
        raise FileNotFoundError(
            f"dataset not found at {path}; generate one with `controlcharts generate`"
        )
    return data_mod.load_csv(path)


def command_generate(args: argparse.Namespace) -> None:
    """Generate a synthetic dataset and write it as CSV."""
    dataset = data_mod.generate(
        samples_per_class=args.samples_per_class, length=args.length, seed=args.seed
    )
    path = data_mod.save_csv(dataset, args.output)
    print(
        json.dumps(
            {
                "output": str(path),
                "series": len(dataset),
                "samples_per_class": args.samples_per_class,
                "length": args.length,
                "seed": args.seed,
            },
            indent=2,
        )
    )


def command_train(args: argparse.Namespace) -> None:
    """Train the classifier and write the model plus training figures."""
    config = Config.from_yaml(args.config)
    dataset = _load_dataset(config)

    training = config.training
    if args.epochs is not None:
        training = type(training)(**{**vars(training), "epochs": args.epochs})
    if args.seed is not None:
        training = type(training)(**{**vars(training), "seed": args.seed})

    result = train(dataset, training)
    model_path = save_model(result.model, args.model or config.paths.model)

    assets = Path(config.paths.assets)
    plots.plot_training_curves(result.history, assets / "training_curves.png")
    plots.plot_example_patterns(dataset, assets / "example_patterns.png")

    print(
        json.dumps(
            {
                "model": str(model_path),
                "train_samples": len(result.split.x_train),
                "test_samples": len(result.split.x_test),
                "epochs_run": result.epochs_run,
                "seed": training.seed,
                "train_accuracy": round(result.final_accuracy, 4),
                "validation_accuracy": round(result.final_val_accuracy, 4),
            },
            indent=2,
        )
    )


def command_evaluate(args: argparse.Namespace) -> None:
    """Score the saved model on the test split and write the confusion matrix."""
    config = Config.from_yaml(args.config)
    dataset = _load_dataset(config)
    split = prepare(dataset, test_size=config.training.test_size, seed=config.training.seed)

    model = load_model(args.model or config.paths.model)
    evaluation = evaluate(model, split.x_test, split.y_test)

    assets = Path(config.paths.assets)
    plots.plot_confusion_matrix(evaluation, assets / "confusion_matrix.png")

    print(format_report(evaluation))
    print()
    print(
        json.dumps(
            {
                "accuracy": round(evaluation.accuracy, 4),
                "test_samples": int(len(evaluation.true_indices)),
                "per_class_recall": {
                    k: round(v, 4) for k, v in evaluation.per_class_accuracy().items()
                },
            },
            indent=2,
        )
    )


def command_predict(args: argparse.Namespace) -> None:
    """Classify a CSV of process measurements and report the findings."""
    config = Config.from_yaml(args.config)
    model = load_model(args.model or config.paths.model)

    usl, lsl = args.usl, args.lsl
    if args.spec:
        if args.spec not in config.specifications:
            raise KeyError(
                f"no specification named {args.spec!r}; known: {sorted(config.specifications)}"
            )
        spec = config.specifications[args.spec]
        usl, lsl = spec.usl, spec.lsl

    report = analyze(model, args.csv, usl=usl, lsl=lsl)

    if args.plot:
        plots.plot_control_charts(report, args.plot, name=args.spec or "")

    payload: dict[str, object] = {
        "source": report.source,
        "subgroups": int(len(report.chart.subgroup_means)),
        "subgroup_size": report.chart.subgroup_size,
        "center_x": round(report.chart.center_x, 6),
        "center_r": round(report.chart.center_r, 6),
        "ucl_x": round(report.chart.ucl_x, 6),
        "lcl_x": round(report.chart.lcl_x, 6),
        "sigma": round(report.chart.sigma, 6),
        "patterns": {
            p.chart: {"pattern": p.pattern, "confidence": round(p.confidence, 4)}
            for p in report.predictions
        },
        "in_control": report.in_control,
        "findings": report.violations,
    }
    if report.capability is not None:
        payload["cp"] = round(report.capability.cp, 4)
        payload["cpk"] = round(report.capability.cpk, 4)

    print(json.dumps(payload, indent=2))


def build_parser() -> argparse.ArgumentParser:
    """Assemble the argument parser for every command."""
    parser = argparse.ArgumentParser(
        prog="controlcharts",
        description="Classify control chart patterns that signal an out-of-control process.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging verbosity (default: WARNING)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="write a synthetic dataset")
    generate.add_argument("--output", default="data/generated.csv", help="destination CSV")
    generate.add_argument("--samples-per-class", type=int, default=100, help="series per class")
    generate.add_argument("--length", type=int, default=60, help="points per series")
    generate.add_argument("--seed", type=int, default=0, help="generator seed")
    generate.set_defaults(func=command_generate)

    train_cmd = subparsers.add_parser("train", help="train the classifier")
    train_cmd.add_argument("--config", default=DEFAULT_CONFIG, help="config file")
    train_cmd.add_argument("--model", help="override the output model path")
    train_cmd.add_argument("--epochs", type=int, help="override the epoch count")
    train_cmd.add_argument("--seed", type=int, help="override the seed")
    train_cmd.set_defaults(func=command_train)

    evaluate_cmd = subparsers.add_parser("evaluate", help="score the saved model")
    evaluate_cmd.add_argument("--config", default=DEFAULT_CONFIG, help="config file")
    evaluate_cmd.add_argument("--model", help="override the model path")
    evaluate_cmd.set_defaults(func=command_evaluate)

    predict = subparsers.add_parser("predict", help="classify a CSV of measurements")
    predict.add_argument("csv", help="CSV of process measurements")
    predict.add_argument("--config", default=DEFAULT_CONFIG, help="config file")
    predict.add_argument("--model", help="override the model path")
    predict.add_argument("--spec", help="named specification from the config")
    predict.add_argument("--usl", type=float, help="upper specification limit")
    predict.add_argument("--lsl", type=float, help="lower specification limit")
    predict.add_argument("--plot", help="also write the control chart figure here")
    predict.set_defaults(func=command_predict)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to a command.

    Args:
        argv: Argument list, or None to read from the process arguments.

    Returns:
        Process exit status.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )
    args.func(args)
    return 0
