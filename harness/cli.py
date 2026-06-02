# Meta-Harness 命令行入口模块。
# 提供 benchmark、run、preflight、report 四个子命令，
# 用于单独评测 agent、启动完整进化实验、检查 solver 连接，以及查看运行结果。

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from harness.eval import DatasetSpec, evaluate_system
from harness.llm import StubLLM, build_solver_client, preflight_solver
from harness.loop import _official_eval_config, load_config, run_harness
from harness.official_eval import evaluate_system_official


# 解析命令行参数，并根据子命令分发到对应处理函数。
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tcharness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # benchmark：只评测一个指定 agent，不启动完整 proposer/进化循环。
    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--system", default="agents/fewshot_all.py")
    benchmark.add_argument("--split", default="val", choices=["train", "val", "test"])
    benchmark.add_argument("--config", default="configs/experiment.yaml")
    benchmark.add_argument("--stub", action="store_true")
    benchmark.add_argument("--max-workers", type=int, default=None)
    benchmark.add_argument("--compact-eval", action="store_true")

    # run：启动完整 Meta-Harness 外层循环。
    run = subparsers.add_parser("run")
    run.add_argument("--config", default="configs/experiment.yaml")

    # preflight：只检查 solver API 是否可用。
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--config", default="configs/experiment.yaml")

    # report：读取已有 run 目录下的 frontier 和 leaderboard。
    report = subparsers.add_parser("report")
    report.add_argument("--run", required=True)

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "benchmark":
        return _benchmark(args)
    if args.command == "run":
        result = run_harness(args.config)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "preflight":
        return _preflight(args)
    if args.command == "report":
        return _report(args)
    return 1


# 执行一次独立 benchmark，并把评测结果以 JSON 打印出来。
def _benchmark(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    project_root = config_path.resolve().parents[1]
    config = load_config(config_path)

    # 将配置里的数据集路径解析成 evaluator 需要的 DatasetSpec。
    datasets = [
        DatasetSpec(
            name=str(row["name"]),
            train_path=str(Path(str(row["train"])) if Path(str(row["train"])).is_absolute() else project_root / row["train"]),
            val_path=str(Path(str(row["val"])) if Path(str(row["val"])).is_absolute() else project_root / row["val"]),
            test_path=str(Path(str(row["test"])) if Path(str(row["test"])).is_absolute() else project_root / row["test"]),
        )
        for row in config.get("datasets", [])
    ]

    # stub 模式用于快速测试流程；正常模式使用配置中的 solver。
    llm = StubLLM('{"label": "unknown"}') if args.stub else build_solver_client(config)
    max_workers = args.max_workers or int(
        config.get("evaluation", {}).get(
            "max_workers",
            config.get("inner_loop", {}).get("max_workers", config.get("benchmark", {}).get("concurrency", 1)),
        )
    )

    # compact_eval 使用旧评测器；默认使用 official-like evaluator。
    if args.compact_eval:
        result = evaluate_system(Path(args.system), datasets, args.split, llm, {}, max_workers=max_workers)
    else:
        result = evaluate_system_official(
            Path(args.system),
            datasets,
            args.split,
            llm,
            _official_eval_config(config, project_root=project_root, run_dir=project_root / "runs" / "_benchmark"),
            max_workers=max_workers,
        )

    # 只输出摘要字段，避免把完整 trace 全部打印到终端。
    print(
        json.dumps(
            {
                "system_name": result.system_name,
                "split": result.split,
                "per_dataset": result.per_dataset,
                "average": result.average,
                "summary": result.summary,
                "errors": result.errors,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


# 打印指定 run 目录下保存的 frontier 和 leaderboard。
def _report(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    frontier = run_dir / "frontier.json"
    leaderboard = run_dir / "leaderboard.csv"
    if frontier.exists():
        print(frontier.read_text(encoding="utf-8"))
    if leaderboard.exists():
        print(leaderboard.read_text(encoding="utf-8"))
    return 0


# 检查 solver 端点是否可访问，并用退出码表示检查结果。
def _preflight(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    result = preflight_solver(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())