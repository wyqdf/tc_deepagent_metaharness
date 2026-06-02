# Meta-Harness proposer 模块。
# 本文件负责调用 DeepAgent 生成候选 agent，并通过受限 backend 控制其读写权限。
# DeepAgent 只负责提出候选；候选是否有效由外层 loop 和 evaluator 评测决定。

from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    from deepagents.backends import LocalShellBackend as _BaseLocalShellBackend
    from deepagents.backends.protocol import (
        EditResult,
        ExecuteResponse,
        GlobResult,
        GrepResult,
        LsResult,
        ReadResult,
        WriteResult,
    )
except ImportError:  # pragma: no cover - real proposer raises a clearer error later
    _BaseLocalShellBackend = object
    EditResult = ExecuteResponse = GlobResult = GrepResult = LsResult = ReadResult = WriteResult = object


_ALLOWED_HARNESS_READS = {
    "harness/__init__.py",
    "harness/agent_protocol.py",
    "harness/llm.py",
    "harness/memory.py",
}
_BANNED_READ_PREFIXES = (
    "runs/",
    "logs/",
    "results/",
    "tc_deepagent_metaharness_backup_pre_clean_",
)
_BANNED_READ_FILES = {
    "AGENT.md",
}
_BANNED_COMMAND_HINTS = (
    "rm ",
    "rm\t",
    "rm\n",
    "mv ",
    "cp ",
    "cat >",
    "set-content",
    "out-file",
    "tee ",
    ">>",
    "1>",
    "2>",
    "del ",
)
_WRITE_PREFIXES = ("agents/", "runs/")
_BANNED_CANDIDATE_CODE_SNIPPETS = (
    "os.remove",
    "os.unlink",
    "shutil.rmtree",
    "subprocess.",
    "popen(",
    "system(",
)


# 候选 agent 的结构化记录，包含名称、文件路径、设计假设、源码和元信息。
@dataclass
class CandidateProposal:
    name: str
    path: str
    hypothesis: str
    source_code: str
    manifest: dict[str, Any] = field(default_factory=dict)


# DeepAgent 的项目级包装器：负责启动 proposer、保存候选、读取 pending_eval，并限制写入边界。
class DeepAgentProposer:
    # 初始化 proposer 的模型、输出目录、运行环境、项目根目录和重试参数。
    def __init__(
        self,
        model: str,
        output_dir: str | Path,
        env: Mapping[str, str] | None = None,
        dry_run: bool = False,
        root_dir: str | Path | None = None,
        allowed_run_dir: str | Path | None = None,
        max_retries: int = 2,
        retry_sleep_seconds: float = 10.0,
    ):
        self.model = model
        self.output_dir = Path(output_dir)
        self.env = dict(env or os.environ)
        self.dry_run = dry_run
        self.root_dir = Path(root_dir) if root_dir is not None else Path.cwd()
        self.allowed_run_dir = Path(allowed_run_dir).resolve() if allowed_run_dir is not None else None
        self.max_retries = int(self.env.get("DEEPAGENT_MAX_RETRIES", max_retries))
        self.retry_sleep_seconds = float(self.env.get("DEEPAGENT_RETRY_SLEEP_SECONDS", retry_sleep_seconds))

    # 旧版单候选入口：调用一次 DeepAgent，并从回复中解析一个候选。
    def propose(self, context: str, iteration: int) -> CandidateProposal:
        if self.dry_run:
            return self._dry_run_proposal(iteration)
        response = self._run_deepagent(context, iteration)
        return self._proposal_from_response(response, iteration)

    # 正式多候选入口：每轮生成候选文件和 pending_eval.json，并返回候选列表。
    def propose_official(
        self,
        task_prompt: str,
        iteration: int,
        pending_eval_path: str | Path,
        trace_path: str | Path | None = None,
        response_path: str | Path | None = None,
        messages_path: str | Path | None = None,
    ) -> list[CandidateProposal]:
        # 每次正式生成前先定位本轮的候选清单文件。
        pending_path = Path(pending_eval_path)
        # 清理上一轮残留，避免 loop 读到旧候选。
        if pending_path.exists():
            pending_path.unlink()
        if trace_path is not None and Path(trace_path).exists():
            Path(trace_path).unlink()
        # dry-run 不调用真实 DeepAgent，只写两个最小候选测试流程。
        if self.dry_run:
            proposals = [self._dry_run_proposal(iteration, index=idx) for idx in range(1, 3)]
            self._write_pending_eval(iteration, proposals, pending_path)
            response_text = "Dry-run proposer generated candidates: " + ", ".join(
                proposal.name for proposal in proposals
            )
            if response_path is not None:
                Path(response_path).parent.mkdir(parents=True, exist_ok=True)
                Path(response_path).write_text(response_text, encoding="utf-8")
            if messages_path is not None:
                Path(messages_path).parent.mkdir(parents=True, exist_ok=True)
                Path(messages_path).write_text(
                    json.dumps(
                        {
                            "dry_run": True,
                            "iteration": iteration,
                            "response": response_text,
                            "candidates": [proposal.name for proposal in proposals],
                        },
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            if trace_path is not None:
                _append_jsonl(
                    Path(trace_path),
                    {
                        "event": "dry_run_proposer",
                        "iteration": iteration,
                        "candidates": [proposal.name for proposal in proposals],
                    },
                )
            return self.load_pending_eval(pending_path, iteration)

        # 真实模式下让 DeepAgent 自己写候选文件和 pending_eval.json。
        response = self._run_deepagent_with_retries(
            task_prompt,
            iteration,
            official_mode=True,
            trace_path=Path(trace_path) if trace_path is not None else None,
            response_path=Path(response_path) if response_path is not None else None,
            messages_path=Path(messages_path) if messages_path is not None else None,
            allowed_run_dir=pending_path.parent,
        )
        if response_path is not None and not Path(response_path).exists():
            Path(response_path).parent.mkdir(parents=True, exist_ok=True)
            Path(response_path).write_text(response, encoding="utf-8")
        return self.load_pending_eval(pending_path, iteration)

    # 带重试的 DeepAgent 调用封装，用于处理模型或 backend 的临时失败。
    def _run_deepagent_with_retries(
        self,
        context: str,
        iteration: int,
        official_mode: bool = False,
        trace_path: Path | None = None,
        response_path: Path | None = None,
        messages_path: Path | None = None,
        allowed_run_dir: Path | None = None,
    ) -> str:
        # 总尝试次数 = 首次调用 + 配置的重试次数。
        attempts = max(1, self.max_retries + 1)
        last_exc: Exception | None = None
        # 每次尝试都会写 trace，便于定位失败原因。
        for attempt in range(1, attempts + 1):
            if trace_path is not None:
                _append_jsonl(
                    trace_path,
                    {
                        "event": "proposer_attempt_started",
                        "iteration": iteration,
                        "attempt": attempt,
                        "max_attempts": attempts,
                    },
                )
            try:
                return self._run_deepagent(
                    context,
                    iteration,
                    official_mode=official_mode,
                    trace_path=trace_path,
                    response_path=response_path,
                    messages_path=messages_path,
                    allowed_run_dir=allowed_run_dir,
                    attempt=attempt,
                )
            except Exception as exc:
                last_exc = exc
                if trace_path is not None:
                    _append_jsonl(
                        trace_path,
                        {
                            "event": "proposer_attempt_failed",
                            "iteration": iteration,
                            "attempt": attempt,
                            "max_attempts": attempts,
                            "error_type": exc.__class__.__name__,
                            "error": str(exc),
                            "will_retry": attempt < attempts,
                        },
                    )
                if attempt >= attempts:
                    raise
                if self.retry_sleep_seconds > 0:
                    time.sleep(self.retry_sleep_seconds)
        raise RuntimeError("DeepAgent proposer failed") from last_exc

    # 从 pending_eval.json 读取候选清单，并加载候选源码。
    def load_pending_eval(self, pending_eval_path: str | Path, iteration: int) -> list[CandidateProposal]:
        pending_path = Path(pending_eval_path)
        # pending_eval.json 是 proposer 和外层 loop 的交接文件。
        data = json.loads(pending_path.read_text(encoding="utf-8"))
        proposals = []
        for item in data.get("candidates", []):
            # 兼容 file 和 path 两种候选文件字段。
            file_value = item.get("file") or item.get("path")
            if not file_value:
                raise KeyError("pending_eval candidate is missing required file/path field")
            raw_path = Path(str(file_value))
            file_path = raw_path if raw_path.is_absolute() else self.root_dir / raw_path
            # 读取候选源码，后续 loop 会基于这个文件进行验证和评测。
            source = file_path.read_text(encoding="utf-8")
            try:
                rel_file = str(file_path.relative_to(self.root_dir))
            except ValueError:
                rel_file = str(file_path)
            proposals.append(
                CandidateProposal(
                    name=str(item["name"]),
                    path=str(file_path),
                    hypothesis=str(item.get("hypothesis", "")),
                    source_code=source,
                    manifest={
                        "iteration": data.get("iteration", iteration),
                        "name": item.get("name"),
                        "file": item.get("file") or rel_file,
                        "path": item.get("path") or rel_file,
                        "hypothesis": item.get("hypothesis", ""),
                        "axis": item.get("axis", ""),
                        "base_system": item.get("base_system", ""),
                        "components": item.get("components", []),
                    },
                )
            )
        return proposals

    # 把 CandidateProposal 的源码写入配置的候选输出目录。
    def write_candidate(self, proposal: CandidateProposal) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        target = self._unique_candidate_path(self.output_dir / Path(proposal.path).name)
        target.write_text(proposal.source_code, encoding="utf-8")
        proposal.path = str(target)
        return target

    # 如果目标文件已存在，则追加数字后缀避免覆盖。
    def _unique_candidate_path(self, target: Path) -> Path:
        if not target.exists():
            return target
        stem = target.stem
        suffix = target.suffix
        parent = target.parent
        counter = 2
        while True:
            candidate = parent / f"{stem}__{counter}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1

    # 生成一个最小可运行的 dry-run 候选，用于测试外层流程。
    def _dry_run_proposal(self, iteration: int, index: int | None = None) -> CandidateProposal:
        suffix = f"_{index}" if index is not None else ""
        name = f"dry_run_candidate_{iteration:03d}{suffix}"
        source = (
            '"""Dry-run generated official-like candidate."""\n\n'
            "import json\n"
            "from harness.agent_protocol import BaseAgentMemory, extract_json_field\n\n"
            "class DryRunCandidate(BaseAgentMemory):\n"
            "    def __init__(self, llm):\n"
            "        super().__init__(llm)\n"
            "        self.examples = []\n\n"
            "    def predict(self, input):\n"
            "        prompt = 'Answer the following question.\\n\\n' + input + "
            "'\\n\\nReturn JSON: {\"final_answer\": \"...\"}'\n"
            "        response = self.call_llm(prompt)\n"
            "        return extract_json_field(response, 'final_answer'), {'full_response': response}\n\n"
            "    def learn_from_batch(self, batch_results):\n"
            "        for row in batch_results:\n"
            "            self.examples.append({'input': row['input'], 'target': row['ground_truth']})\n\n"
            "    def get_state(self):\n"
            "        return json.dumps({'examples': self.examples}, ensure_ascii=False)\n\n"
            "    def set_state(self, state):\n"
            "        self.examples = json.loads(state).get('examples', [])\n"
        )
        return CandidateProposal(
            name=name,
            path=f"{name}.py",
            hypothesis="Dry-run candidate uses direct prompting with official-like agent protocol.",
            source_code=source,
            manifest={
                "name": name,
                "model": self.model,
                "dry_run": True,
                "uses_auth_token_env": "ANTHROPIC_AUTH_TOKEN",
                "uses_base_url_env": "ANTHROPIC_BASE_URL",
            },
        )

    # 把候选列表写成外层 loop 期望的 pending_eval.json 格式。
    def _write_pending_eval(
        self,
        iteration: int,
        proposals: Sequence[CandidateProposal],
        pending_path: Path,
    ) -> None:
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        candidates = []
        for proposal in proposals:
            written = self.write_candidate(proposal)
            try:
                rel_file = str(written.relative_to(self.root_dir))
            except ValueError:
                rel_file = str(written)
            candidates.append(
                {
                    "name": proposal.name,
                    "file": rel_file,
                    "hypothesis": proposal.hypothesis,
                    "axis": proposal.manifest.get("axis", "dry_run"),
                    "base_system": proposal.manifest.get("base_system", "no_memory"),
                    "components": proposal.manifest.get("components", ["dry_run"]),
                }
            )
        pending_path.write_text(
            json.dumps({"iteration": iteration, "candidates": candidates}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # 真正创建 DeepAgent、执行 task prompt，并保存回复和工具调用记录。
    def _run_deepagent(
        self,
        context: str,
        iteration: int,
        official_mode: bool = False,
        trace_path: Path | None = None,
        response_path: Path | None = None,
        messages_path: Path | None = None,
        allowed_run_dir: Path | None = None,
        attempt: int = 1,
    ) -> str:
        try:
            from deepagents import create_deep_agent
            from deepagents.backends import LocalShellBackend
        except ImportError as exc:
            raise RuntimeError("DeepAgent dependencies are missing. Run `uv sync`.") from exc

        # 临时注入 DeepAgent 所需环境变量，结束后必须恢复。
        old_env = os.environ.copy()
        os.environ.update(self._deepagent_env())
        try:
            # backend 负责限制 DeepAgent 的文件系统和 shell 权限。
            backend_kwargs = {
                "root_dir": str(self.root_dir),
                "allowed_run_dir": str(allowed_run_dir) if allowed_run_dir is not None else None,
                "virtual_mode": False,
                "inherit_env": True,
                "timeout": int(self.env.get("DEEPAGENT_SHELL_TIMEOUT", "120")),
                "max_output_bytes": int(self.env.get("DEEPAGENT_MAX_OUTPUT_BYTES", "200000")),
            }
            if trace_path is not None:
                backend = _LoggingLocalShellBackend(
                    trace_path=trace_path,
                    write_guard=self._write_guard,
                    **backend_kwargs,
                )
            else:
                backend = _LoggingLocalShellBackend(
                    trace_path=None,
                    write_guard=self._write_guard,
                    **backend_kwargs,
                )
            if trace_path is not None:
                _append_jsonl(
                    trace_path,
                    {
                        "event": "proposer_session_started",
                        "iteration": iteration,
                        "official_mode": official_mode,
                        "attempt": attempt,
                        "model": self.model,
                        "root_dir": str(self.root_dir),
                        "skills": [],
                        "allowed_reads": sorted(_ALLOWED_HARNESS_READS),
                        "banned_read_prefixes": list(_BANNED_READ_PREFIXES),
                        "banned_read_files": sorted(_BANNED_READ_FILES),
                        "banned_shell_hints": list(_BANNED_COMMAND_HINTS),
                    },
                )
            # 创建真正的 DeepAgent，模型、backend 和 system prompt 都在这里接入。
            agent = create_deep_agent(
                model=self._make_anthropic_model(),
                backend=backend,
                system_prompt=self._system_prompt(iteration, official_mode=official_mode),
            )
            # 把 loop 构造的 task prompt 交给 DeepAgent 执行。
            result = agent.invoke({"messages": [{"role": "user", "content": context}]})
            response = _extract_text(result)
            if messages_path is not None:
                messages_path.parent.mkdir(parents=True, exist_ok=True)
                messages_path.write_text(
                    json.dumps(_serialize_for_json(result), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            if response_path is not None:
                response_path.parent.mkdir(parents=True, exist_ok=True)
                response_path.write_text(response, encoding="utf-8")
            if trace_path is not None:
                _append_jsonl(
                    trace_path,
                    {
                        "event": "proposer_session_finished",
                        "iteration": iteration,
                        "response_chars": len(response),
                        "response_preview": _preview(response),
                        "inspected_files": _extract_inspected_paths(trace_path),
                    },
                )
            return response
        except Exception as exc:
            if trace_path is not None:
                _append_jsonl(
                    trace_path,
                    {
                        "event": "proposer_session_failed",
                        "iteration": iteration,
                        "attempt": attempt,
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                    },
                )
            raise
        # 无论成功或失败，都恢复调用前的环境变量。
        finally:
            os.environ.clear()
            os.environ.update(old_env)

    # 构造 DeepAgent 运行时使用的环境变量。
    def _deepagent_env(self) -> dict[str, str]:
        env = dict(self.env)
        env.setdefault("ANTHROPIC_BASE_URL", "https://doro.lol")
        if env.get("anthropic_auth_token") and not env.get("ANTHROPIC_AUTH_TOKEN"):
            env["ANTHROPIC_AUTH_TOKEN"] = env["anthropic_auth_token"]
        self.env = env
        return env

    # 根据环境变量创建 proposer 使用的 ChatAnthropic 模型。
    def _make_anthropic_model(self) -> Any:
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise RuntimeError("DeepAgent Claude proposer needs langchain-anthropic.") from exc

        env = self._deepagent_env()
        auth_token = env.get("ANTHROPIC_AUTH_TOKEN")
        base_url = env.get("ANTHROPIC_BASE_URL", "https://doro.lol")
        kwargs = {"model": self.model, "temperature": 0.0, "base_url": base_url}
        if auth_token:
            return _make_chat_anthropic_with_auth_token(ChatAnthropic, auth_token, **kwargs)
        raise RuntimeError("DeepAgent proposer needs ANTHROPIC_AUTH_TOKEN.")

    # 生成 DeepAgent 的系统提示词，约束其候选生成方式和写入规则。
    def _system_prompt(self, iteration: int, official_mode: bool = False) -> str:
        # 非 official 模式只要求返回一个候选。
        if not official_mode:
            return (
                "You are the DeepAgent proposer for a compact Meta-Harness text classification project. "
                f"This is iteration {iteration}. Return exactly one candidate. "
                f"The workspace root is {self.root_dir}. You may inspect project files and run safe read-only commands. "
                "The candidate must be Python code defining a BaseAgentMemory subclass. "
                "Do not modify files outside agents/."
            )
        # official 模式要求两个候选、prototype、pending_eval 和禁止数据集硬编码。
        return (
            "You are the DeepAgent proposer for a compact Meta-Harness text classification project. "
            f"This is iteration {iteration}. Work in the main session. Do not delegate. "
            f"The workspace root is {self.root_dir}. Read source files, result files, traces, and logs yourself. "
            "You MUST prototype core logic with temporary scripts in /tmp/ before writing final candidates. "
            "You do NOT run the full benchmark. The outer loop handles validation and benchmarking. "
            "You MUST design exactly 2 new memory-system candidates every iteration: one exploitation and one exploration. "
            "Avoid parameter-only variants; change a real mechanism such as memory content, retrieval, prompt organization, "
            "learning strategy, or verification. Do not hardcode dataset names or dataset-specific hints in candidate code. "
            "Write candidate Python files only under agents/. "
            f"Candidate file names MUST include an iteration-specific prefix such as iter{iteration:03d}_ and must not overwrite existing files. "
            "Each candidate must define a BaseAgentMemory subclass implementing predict(input), "
            "learn_from_batch(batch_results), get_state(), and set_state(). "
            "Finally write pending_eval.json exactly at the path specified by the task prompt. "
            "The JSON must contain iteration and a candidates array with name, file, hypothesis, axis, base_system, components. "
            "Return a short final message naming the candidates and the files you inspected."
        )

    # 旧版解析逻辑：从 DeepAgent 文本回复中提取 manifest 和 Python 代码。
    def _proposal_from_response(self, response: str, iteration: int) -> CandidateProposal:
        manifest = _extract_manifest(response)
        source = _extract_python_code(response)
        name = str(manifest.get("name") or f"deepagent_candidate_{iteration:03d}")
        return CandidateProposal(
            name=name,
            path=f"{name}.py",
            hypothesis=str(manifest.get("hypothesis") or "DeepAgent generated memory candidate."),
            source_code=source,
            manifest=manifest,
        )

    # 写文件前的安全检查，防止候选写到非法路径或包含危险代码。
    def _write_guard(self, file_path: str, content: str) -> bool:
        rel = self._relative_tool_path(file_path)
        # /tmp 只用于临时原型实验。
        if rel.startswith("/tmp/"):
            return True
        # 正式候选只能写 agents，运行产物只能写当前 run 目录。
        if not rel.startswith("agents/") and not self._is_allowed_run_path(rel):
            return False
        if self._is_allowed_run_path(rel):
            return True
        lowered = content.lower()
        # 拒绝带删除、系统命令或 subprocess 的候选代码。
        if any(snippet in lowered for snippet in _BANNED_CANDIDATE_CODE_SNIPPETS):
            return False
        return True

    # 把工具传入路径转换为相对于项目根目录的规范路径。
    def _relative_tool_path(self, path: str | Path | None) -> str:
        if path is None:
            return ""
        raw = Path(str(path))
        resolved = (self.root_dir / raw).resolve() if not raw.is_absolute() else raw.resolve()
        try:
            return resolved.relative_to(self.root_dir).as_posix()
        except ValueError:
            return resolved.as_posix()

    # 判断路径是否位于当前允许访问的 run 目录中。
    def _is_allowed_run_path(self, rel: str) -> bool:
        if self.allowed_run_dir is None:
            return rel.startswith("runs/")
        try:
            allowed = self.allowed_run_dir.relative_to(self.root_dir).as_posix()
        except ValueError:
            return False
        return rel == allowed or rel.startswith(f"{allowed}/")


# 包装 ChatAnthropic，使其使用 auth_token 参数而不是默认 api_key。
def _make_chat_anthropic_with_auth_token(chat_cls: Any, auth_token: str, **kwargs: Any) -> Any:
    from pydantic import Field

    # 为 ChatAnthropic 增加 anthropic_auth_token 字段的内部子类。
    class ChatAnthropicWithAuthToken(chat_cls):
        anthropic_auth_token: str = Field(exclude=True, repr=False)

        # 生成底层 Anthropic 客户端参数，并把 token 放入 auth_token 字段。
        @cached_property
        def _client_params(self) -> dict[str, Any]:
            params = super()._client_params.copy()
            params.pop("api_key", None)
            params["auth_token"] = self.anthropic_auth_token
            return params

    return ChatAnthropicWithAuthToken(anthropic_auth_token=auth_token, **kwargs)


# 受限 LocalShellBackend：拦截 DeepAgent 的文件、搜索和 shell 操作，并写入审计 trace。
class _LoggingLocalShellBackend(_BaseLocalShellBackend):

    # 初始化受限 backend 的 trace 路径、项目根目录、允许的 run 目录和写入守卫。
    def __init__(
        self,
        *args: Any,
        trace_path: str | Path | None = None,
        write_guard: Callable[[str, str], bool] | None = None,
        **kwargs: Any,
    ) -> None:
        allowed_run_dir = kwargs.pop("allowed_run_dir", None)
        self._trace_path = Path(trace_path) if trace_path is not None else None
        self._root_dir = Path(kwargs.get("root_dir", Path.cwd())).resolve()
        self._allowed_run_dir = (
            Path(allowed_run_dir).resolve() if allowed_run_dir is not None else None
        )
        self._write_guard_fn = write_guard
        super().__init__(*args, **kwargs)

    # 执行 shell 命令前先做权限检查，并记录执行结果。
    def execute(self, command: str, *, timeout: int | None = None) -> Any:
        # 命令被拒绝时不抛异常，而是返回 Access denied 给 DeepAgent。
        if self._command_denied(command):
            result = ExecuteResponse(
                output=(
                    f"Access denied by proposer information boundary for execute: {command}. "
                    "Use the source/result paths listed in the proposer prompt."
                ),
                exit_code=1,
                truncated=False,
            )
            self._record("execute", {"command": command, "timeout": timeout}, result)
            return result
        result = self._normalize_execute_result(super().execute(command, timeout=timeout))
        self._record("execute", {"command": command, "timeout": timeout}, result)
        return result

    # 读取文件前检查读权限，并记录读取行为。
    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> Any:
        # 读权限失败时返回受控错误，并写入 trace。
        if self._read_denied(file_path):
            result = ReadResult(
                error=(
                    f"Access denied by proposer information boundary for read: {file_path}. "
                    "Use the source/result paths listed in the proposer prompt."
                ),
                file_data=None,
            )
            self._record("read", {"file_path": file_path, "offset": offset, "limit": limit}, result)
            return result
        result = self._normalize_read_result(super().read(file_path, offset=offset, limit=limit))
        self._record("read", {"file_path": file_path, "offset": offset, "limit": limit}, result)
        return result

    # 写文件前检查写权限和内容安全，并记录写入行为。
    def write(self, file_path: str, content: str) -> Any:
        # 写入前同时检查路径权限和内容安全。
        if self._write_denied(file_path) or (self._write_guard_fn is not None and not self._write_guard_fn(file_path, content)):
            result = WriteResult(
                error=(
                    f"Access denied by proposer information boundary for write: {file_path}. "
                    "Write candidate files only under agents/ or run artifacts under runs/."
                ),
                path=None,
            )
            self._record(
                "write",
                {"file_path": file_path, "content_chars": len(content), "content_preview": _preview(content)},
                result,
            )
            return result
        result = self._normalize_write_result(super().write(file_path, content))
        self._record(
            "write",
            {"file_path": file_path, "content_chars": len(content), "content_preview": _preview(content)},
            result,
        )
        return result

    # 编辑文件前检查写权限和内容安全，并记录编辑行为。
    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> Any:
        if self._write_denied(file_path) or (self._write_guard_fn is not None and not self._write_guard_fn(file_path, new_string)):
            result = EditResult(
                error=(
                    f"Access denied by proposer information boundary for edit: {file_path}. "
                    "Write candidate files only under agents/ or run artifacts under runs/."
                ),
                path=None,
                occurrences=None,
            )
            self._record(
                "edit",
                {
                    "file_path": file_path,
                    "old_chars": len(old_string),
                    "new_chars": len(new_string),
                    "replace_all": replace_all,
                    "old_preview": _preview(old_string),
                    "new_preview": _preview(new_string),
                },
                result,
            )
            return result
        result = self._normalize_edit_result(super().edit(file_path, old_string, new_string, replace_all=replace_all))
        self._record(
            "edit",
            {
                "file_path": file_path,
                "old_chars": len(old_string),
                "new_chars": len(new_string),
                "replace_all": replace_all,
                "old_preview": _preview(old_string),
                "new_preview": _preview(new_string),
            },
            result,
        )
        return result

    # 列目录后过滤禁止路径，并记录结果。
    def ls(self, path: str) -> Any:
        result = self._normalize_ls_result(super().ls(path))
        result = self._filter_path_result(result)
        self._record("ls", {"path": path}, result)
        return result

    # glob 搜索后过滤禁止路径，并记录结果。
    def glob(self, pattern: str, path: str = "/") -> Any:
        result = self._normalize_glob_result(super().glob(pattern, path=path))
        result = self._filter_path_result(result)
        self._record("glob", {"pattern": pattern, "path": path}, result)
        return result

    # grep 搜索前检查搜索范围，结果中继续过滤禁止路径。
    def grep(self, pattern: str, path: str | None = None, glob: str | None = None) -> Any:
        # 禁止无范围或越界搜索，防止 DeepAgent 扫描整个项目。
        if self._search_denied(path):
            result = GrepResult(
                error=(
                    f"Access denied by proposer information boundary for grep: {path or '<workspace>'}. "
                    "Use the source/result paths listed in the proposer prompt."
                ),
                matches=None,
            )
            self._record("grep", {"pattern": pattern, "path": path, "glob": glob}, result)
            return result
        result = self._normalize_grep_result(super().grep(pattern, path=path, glob=glob))
        result = self._filter_path_result(result)
        self._record("grep", {"pattern": pattern, "path": path, "glob": glob}, result)
        return result

    # 原始 grep 搜索接口，同样受搜索范围限制。
    def grep_raw(self, pattern: str, path: str | None = None, glob: str | None = None) -> Any:
        if self._search_denied(path):
            result = (
                f"Access denied by proposer information boundary for grep_raw: {path or '<workspace>'}. "
                "Use the source/result paths listed in the proposer prompt."
            )
            self._record("grep_raw", {"pattern": pattern, "path": path, "glob": glob}, result)
            return result
        result = super().grep_raw(pattern, path=path, glob=glob)
        result = self._filter_path_result(result)
        self._record("grep_raw", {"pattern": pattern, "path": path, "glob": glob}, result)
        return result

    # glob_info 的受限包装版本。
    def glob_info(self, pattern: str, path: str = "/") -> Any:
        result = self._normalize_glob_result(super().glob_info(pattern, path=path))
        result = self._filter_path_result(result)
        self._record("glob_info", {"pattern": pattern, "path": path}, result)
        return result

    # ls_info 的受限包装版本。
    def ls_info(self, path: str) -> Any:
        result = self._normalize_ls_result(super().ls_info(path))
        result = self._filter_path_result(result)
        self._record("ls_info", {"path": path}, result)
        return result

    # 把一次 backend 工具调用压缩为 JSONL 审计事件。
    def _record(self, tool: str, args: Mapping[str, Any], result: Any) -> None:
        if self._trace_path is None:
            return
        # trace 中只保存摘要，避免日志文件过大。
        summary = _summarize_backend_result(result)
        payload = {
            "event": "backend_call",
            "tool": tool,
            "args": dict(args),
            "result": summary,
            "result_preview": _backend_result_preview(summary),
        }
        for key in ("file_path", "path", "command", "pattern"):
            value = args.get(key)
            if value is not None:
                payload[key] = value
        _append_jsonl(
            self._trace_path,
            payload,
        )

    # 把 backend 工具路径转换为相对于项目根目录的路径。
    def _relative_tool_path(self, path: str | Path | None) -> str:
        if path is None:
            return ""
        raw = Path(str(path))
        resolved = (self._root_dir / raw).resolve() if not raw.is_absolute() else raw.resolve()
        try:
            return resolved.relative_to(self._root_dir).as_posix()
        except ValueError:
            return resolved.as_posix()

    # 返回当前允许访问的 run 目录相对路径。
    def _allowed_run_rel(self) -> str | None:
        if self._allowed_run_dir is None:
            return None
        try:
            return self._allowed_run_dir.relative_to(self._root_dir).as_posix()
        except ValueError:
            return None

    # 判断路径是否落在允许访问的 run 目录内。
    def _is_allowed_run_path(self, path: str | Path | None) -> bool:
        rel = self._relative_tool_path(path)
        allowed = self._allowed_run_rel()
        if allowed is None:
            return False
        return rel == allowed or rel.startswith(f"{allowed}/")

    # 判断路径是否指向 runs 根目录或其子路径。
    def _is_root_run_path(self, rel: str) -> bool:
        return rel == "runs" or rel.startswith("runs/")

    # 读权限判断：只放行白名单源码和当前 run 目录。
    def _read_denied(self, path: str | Path | None) -> bool:
        rel = self._relative_tool_path(path)
        # 当前 run 目录下的日志和产物允许读取。
        if self._is_allowed_run_path(path):
            return False
        # 禁止直接读取整个 runs 根目录，避免偷看其他实验。
        if self._is_root_run_path(rel):
            return True
        if any(rel.startswith(prefix) for prefix in _BANNED_READ_PREFIXES):
            return True
        if Path(rel).name in _BANNED_READ_FILES:
            return True
        # harness 源码只允许读取白名单文件。
        return rel.startswith("harness/") and rel not in _ALLOWED_HARNESS_READS

    # 写权限判断：只允许写 agents、/tmp 或当前 run 目录。
    def _write_denied(self, path: str | Path | None) -> bool:
        rel = self._relative_tool_path(path)
        return not (rel.startswith("agents/") or rel.startswith("/tmp/") or self._is_allowed_run_path(path))

    # 搜索权限判断，禁止无范围全局搜索。
    def _search_denied(self, path: str | None) -> bool:
        if path is None:
            return True
        rel = self._relative_tool_path(path)
        if self._is_allowed_run_path(path):
            return False
        return rel in {"", ".", "harness"} or self._read_denied(path)

    # shell 命令级安全判断，防止读敏感源码、改评测器或危险写入。
    def _command_denied(self, command: str) -> bool:
        lowered = command.lower()
        allowed_rel = self._allowed_run_rel()
        allowed_abs = self._allowed_run_dir.as_posix().lower() if self._allowed_run_dir else ""
        # 显式禁止访问外部指令文件和备份目录。
        if "agent.md" in lowered or "tc_deepagent_metaharness_backup_pre_clean" in lowered:
            return True
        # 访问 runs/logs/results 时必须限定在当前允许的 run 目录。
        if "runs/" in lowered or "/runs/" in lowered:
            allowed_rel_l = allowed_rel.lower() if allowed_rel else ""
            if allowed_rel_l not in lowered and allowed_abs not in lowered:
                return True
        if re.search(r"(^|[\s/])logs/", lowered) or re.search(r"(^|[\s/])results/", lowered):
            allowed_rel_l = allowed_rel.lower() if allowed_rel else ""
            if allowed_abs not in lowered and allowed_rel_l not in lowered:
                return True
        # 下面的规则只针对可能修改 agents 或 runs 的命令。
        touches_project_outputs = (
            "agents/" in lowered
            or "/agents/" in lowered
            or "runs/" in lowered
            or "/runs/" in lowered
        )
        if touches_project_outputs and any(
            hint in lowered
            for hint in (
                "os.remove",
                ".unlink",
                "unlink(",
                "shutil.rmtree",
                "rmtree(",
                ".write(",
                ".write_text",
                ".write_bytes",
                "open(",
                "with open",
            )
        ) and re.search(r"['\"]\s*[wa+x]", lowered):
            return True
        if ("agents/" in lowered or "/agents/" in lowered or "runs/" in lowered or "/runs/" in lowered) and (
            "cat >" in lowered or ">>" in lowered or "1>" in lowered or "2>" in lowered or "<<" in lowered
        ):
            return True
        # 禁止通过 shell 查看或修改评测器和核心控制文件。
        denied_names = [
            "official_eval",
            "evaluators",
            "harness/data.py",
            "harness/eval.py",
            "harness/loop.py",
            "harness/proposer.py",
            "harness/store.py",
            "harness/cli.py",
        ]
        if any(name in lowered for name in denied_names):
            return True
        if touches_project_outputs and any(hint in lowered for hint in _BANNED_COMMAND_HINTS):
            return True
        # 禁止用通用文本工具大范围扫描 harness。
        if re.search(r"\b(rg|grep|find|cat|sed|awk)\b", lowered) and re.search(r"\bharness\b", lowered):
            return True
        return False

    # 过滤 ls/glob/grep 结果中的禁止路径。
    def _filter_path_result(self, result: Any) -> Any:
        # 某些 backend 会直接返回列表，先按路径过滤。
        if isinstance(result, list):
            return [item for item in result if not self._path_item_denied(item)]
        if not any(hasattr(result, key) for key in ("entries", "matches")):
            return result
        # 标准结果对象通常把路径放在 entries 或 matches 里。
        for key in ("entries", "matches"):
            values = getattr(result, key, None)
            if isinstance(values, list):
                setattr(result, key, [item for item in values if not self._path_item_denied(item)])
        return result

    # 判断某个搜索结果条目是否应该被隐藏。
    def _path_item_denied(self, item: Any) -> bool:
        if isinstance(item, Mapping):
            path = item.get("path") or item.get("file_path") or item.get("file")
        else:
            path = getattr(item, "path", None) or getattr(item, "file_path", None) or getattr(item, "file", None)
        if not path:
            return False
        rel = self._relative_tool_path(path)
        if self._is_allowed_run_path(path):
            return False
        if self._is_root_run_path(rel):
            return True
        if any(rel.startswith(prefix) for prefix in _BANNED_READ_PREFIXES):
            return True
        if Path(rel).name in _BANNED_READ_FILES:
            return True
        return self._read_denied(path)

    # 把 execute 的不同返回格式统一为 ExecuteResponse。
    def _normalize_execute_result(self, result: Any) -> ExecuteResponse:
        if isinstance(result, ExecuteResponse):
            return result
        if isinstance(result, Mapping):
            return ExecuteResponse(
                output=str(result.get("output", "")),
                exit_code=result.get("exit_code"),
                truncated=bool(result.get("truncated", False)),
            )
        if isinstance(result, str):
            return ExecuteResponse(output=result, exit_code=None, truncated=False)
        output = getattr(result, "output", "")
        exit_code = getattr(result, "exit_code", None)
        truncated = bool(getattr(result, "truncated", False))
        return ExecuteResponse(output=str(output), exit_code=exit_code, truncated=truncated)

    # 把 read 的不同返回格式统一为 ReadResult。
    def _normalize_read_result(self, result: Any) -> ReadResult:
        if isinstance(result, ReadResult):
            return result
        if isinstance(result, Mapping):
            return ReadResult(error=result.get("error"), file_data=result.get("file_data"))
        return ReadResult(error=None, file_data={"content": str(result), "encoding": "utf-8"})

    # 把 write 的不同返回格式统一为 WriteResult。
    def _normalize_write_result(self, result: Any) -> WriteResult:
        if isinstance(result, WriteResult):
            return result
        if isinstance(result, Mapping):
            return WriteResult(error=result.get("error"), path=result.get("path"))
        return WriteResult(error=None, path=str(result) if result is not None else None)

    # 把 edit 的不同返回格式统一为 EditResult。
    def _normalize_edit_result(self, result: Any) -> EditResult:
        if isinstance(result, EditResult):
            return result
        if isinstance(result, Mapping):
            return EditResult(
                error=result.get("error"),
                path=result.get("path"),
                occurrences=result.get("occurrences"),
            )
        return EditResult(error=None, path=str(result) if result is not None else None, occurrences=None)

    # 把 ls 的不同返回格式统一为 LsResult。
    def _normalize_ls_result(self, result: Any) -> LsResult:
        if isinstance(result, LsResult):
            return result
        if isinstance(result, Mapping):
            return LsResult(error=result.get("error"), entries=result.get("entries"))
        if isinstance(result, list):
            return LsResult(error=None, entries=result)
        return LsResult(error=None, entries=[])

    # 把 glob 的不同返回格式统一为 GlobResult。
    def _normalize_glob_result(self, result: Any) -> GlobResult:
        if isinstance(result, GlobResult):
            return result
        if isinstance(result, Mapping):
            return GlobResult(error=result.get("error"), matches=result.get("matches"))
        if isinstance(result, list):
            return GlobResult(error=None, matches=result)
        return GlobResult(error=None, matches=[])

    # 把 grep 的不同返回格式统一为 GrepResult。
    def _normalize_grep_result(self, result: Any) -> GrepResult:
        if isinstance(result, GrepResult):
            return result
        if isinstance(result, Mapping):
            return GrepResult(error=result.get("error"), matches=result.get("matches"))
        if isinstance(result, list):
            return GrepResult(error=None, matches=result)
        return GrepResult(error=None, matches=[])

    # backend 内部写入守卫，限制候选写入和重定向类内容。
    def _write_guard(self, file_path: str, content: str) -> bool:
        rel = self._relative_tool_path(file_path)
        if not rel.startswith("agents/") and not rel.startswith("runs/"):
            return False
        if rel.startswith("runs/"):
            return True
        lowered = content.lower()
        if any(hint in lowered for hint in _BANNED_COMMAND_HINTS):
            return False
        if ">" in lowered and "write_candidate" not in lowered:
            return False
        return True


# 从 proposer trace 中提取 DeepAgent 检查过的路径、命令和搜索模式。
def _extract_inspected_paths(trace_path: Path | None) -> list[str]:
    if trace_path is None or not trace_path.exists():
        return []
    paths: list[str] = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") != "backend_call":
            continue
        for key in ("file_path", "path", "command", "pattern"):
            value = event.get(key)
            if not isinstance(value, str) or not value:
                value = (event.get("args") or {}).get(key)
            if isinstance(value, str) and value:
                paths.append(value)
    return paths


# 从 backend 调用摘要中提取最适合展示的一小段预览。
def _backend_result_preview(summary: Mapping[str, Any]) -> str:
    for key in ("error", "output_preview", "content_preview", "matches_preview", "entries_preview"):
        value = summary.get(key)
        if value:
            return str(value)
    return ""


# 从 DeepAgent 返回对象中提取最终文本回复。
def _extract_text(raw_result: Any) -> str:
    if isinstance(raw_result, dict):
        messages = raw_result.get("messages") or []
        if messages:
            content = getattr(messages[-1], "content", "")
            if isinstance(content, list):
                return "\n".join(str(getattr(item, "text", item)) for item in content)
            return str(content)
    return str(raw_result)


# 从 DeepAgent 回复中提取 Python 候选代码块。
def _extract_python_code(response: str) -> str:
    match = re.search(r"```python\s*([\s\S]*?)\s*```", response)
    if not match:
        match = re.search(r"```\s*([\s\S]*?)\s*```", response)
    if match:
        return match.group(1).strip() + "\n"
    if "class " in response and "BaseAgentMemory" in response:
        return response.strip() + "\n"
    raise ValueError("DeepAgent response did not contain candidate Python code.")


# 从 DeepAgent 回复中提取 JSON manifest。
def _extract_manifest(response: str) -> dict[str, Any]:
    match = re.search(r"```json\s*([\s\S]*?)\s*```", response)
    if not match:
        return {}
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


# 向 JSONL 文件追加一条带 UTC 时间戳的事件。
def _append_jsonl(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **dict(event)}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_serialize_for_json(payload), ensure_ascii=False) + "\n")


# 截断长文本，避免 trace 或日志过大。
def _preview(value: Any, limit: int = 4000) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


# 把 backend 原始返回值压缩成适合审计记录的摘要。
def _summarize_backend_result(result: Any) -> dict[str, Any]:
    # 先统一序列化，再抽取关键字段做摘要。
    data = _serialize_for_json(result)
    if not isinstance(data, dict):
        return {"value_preview": _preview(data)}

    summary: dict[str, Any] = {}
    for key in ["error", "path", "exit_code", "truncated", "occurrences"]:
        if key in data:
            summary[key] = data[key]
    # 命令输出可能很长，只记录长度和预览。
    if "output" in data:
        output = str(data.get("output", ""))
        summary["output_chars"] = len(output)
        summary["output_preview"] = _preview(output)
    if "entries" in data:
        entries = data.get("entries") or []
        summary["entry_count"] = len(entries)
        summary["entries_preview"] = entries[:20]
    if "matches" in data:
        matches = data.get("matches") or []
        summary["match_count"] = len(matches)
        summary["matches_preview"] = matches[:20]
    # 文件内容也只记录长度和预览，避免 trace 过大。
    if "file_data" in data and data["file_data"]:
        file_data = data["file_data"]
        if isinstance(file_data, Mapping):
            content = str(file_data.get("content", ""))
            summary["encoding"] = file_data.get("encoding")
            summary["content_chars"] = len(content)
            summary["content_preview"] = _preview(content)
    return summary


# 递归转换对象，使其可以安全写入 JSON。
def _serialize_for_json(value: Any) -> Any:
    # dataclass 先转 dict，再继续递归处理。
    if is_dataclass(value) and not isinstance(value, type):
        return _serialize_for_json(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _serialize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_for_json(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # 兼容 Pydantic v2 对象。
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _serialize_for_json(model_dump())
        except Exception:
            pass
    # 兼容 Pydantic v1 或类似对象。
    dict_method = getattr(value, "dict", None)
    if callable(dict_method):
        try:
            return _serialize_for_json(dict_method())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _serialize_for_json(vars(value))
        except Exception:
            pass
    return str(value)
