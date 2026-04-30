from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Sequence

from ai_trader.self_improvement.proposal import (
    PromptProposal,
    Proposal,
    ProposalStatus,
    WeightProposal,
    assert_safe_target,
)


Runner = Callable[[Sequence[str]], subprocess.CompletedProcess]


class GitProposalStore:
    def __init__(self, *, repo_root: Path = Path("."), runner: Runner | None = None) -> None:
        self._repo_root = repo_root
        self._runner = runner or self._run

    def submit(self, proposal: Proposal) -> None:
        self._assert_safe(proposal)
        if proposal.status not in {ProposalStatus.PENDING, ProposalStatus.APPROVED}:
            raise ValueError(f"Proposal status is not submittable: {proposal.status}")

        branch = f"ai-proposal/{proposal.proposal_id[:8]}"
        proposal_path = self._repo_root / "proposals" / "pending" / f"{proposal.proposal_id}.json"

        self._runner(["git", "checkout", "-b", branch])
        proposal_path.parent.mkdir(parents=True, exist_ok=True)
        proposal_path.write_text(proposal.model_dump_json(indent=2), encoding="utf-8")

        if isinstance(proposal, PromptProposal):
            self._apply_prompt_proposal(proposal)

        self._runner(["git", "add", "-A"])
        self._runner(["git", "commit", "-m", self._commit_message(proposal)])
        self._runner(["git", "push", "origin", "HEAD"])
        self._runner(
            [
                "gh",
                "pr",
                "create",
                "--title",
                self._pr_title(proposal),
                "--body",
                self._pr_body(proposal),
                "--label",
                "ai-proposal",
            ]
        )

    def _apply_prompt_proposal(self, proposal: PromptProposal) -> None:
        target_path = self._resolve_target_path(proposal.target_module)
        if target_path is None or not target_path.exists():
            return
        text = target_path.read_text(encoding="utf-8")
        if proposal.current_text not in text:
            return
        target_path.write_text(text.replace(proposal.current_text, proposal.proposed_text, 1), encoding="utf-8")

    def _resolve_target_path(self, target_module: str) -> Path | None:
        module_name = target_module.split(":", 1)[0]
        if module_name.endswith(".py") or "/" in module_name or "\\" in module_name:
            return self._repo_root / module_name
        return self._repo_root / "src" / "ai_trader" / f"{module_name.replace('.', '/')}.py"

    def _assert_safe(self, proposal: Proposal) -> None:
        if isinstance(proposal, PromptProposal):
            for value in (proposal.target_module, proposal.current_text, proposal.proposed_text):
                assert_safe_target(value)
        elif isinstance(proposal, WeightProposal):
            assert_safe_target(proposal.target_field)

    def _commit_message(self, proposal: Proposal) -> str:
        target = proposal.target_module if isinstance(proposal, PromptProposal) else proposal.target_field
        rationale = proposal.rationale.replace("\n", " ")[:60]
        return f"[AI Proposal] {target}: {rationale}"

    def _pr_title(self, proposal: Proposal) -> str:
        target = proposal.target_module if isinstance(proposal, PromptProposal) else proposal.target_field
        return f"AI Proposal: {target}"

    def _pr_body(self, proposal: Proposal) -> str:
        outcome = proposal.source_outcome
        return (
            f"## Original Trade Thesis\n{'; '.join(outcome.trade_plan.thesis)}\n\n"
            f"## Actual Outcome\nTicker: {outcome.ticker}\n"
            f"PnL: {outcome.pnl_pct:.2%}\nExit: {outcome.exit_reason} on {outcome.actual_exit_date}\n\n"
            f"## Rationale\n{proposal.rationale}\n\n"
            "## Expected Improvement\n"
            f"{proposal.expected_improvement if isinstance(proposal, PromptProposal) else 'Weight adjustment'}\n\n"
            "**Human approval required before merge. Never auto-merge AI proposals.**"
        )

    def _run(self, command: Sequence[str]) -> subprocess.CompletedProcess:
        return subprocess.run(command, cwd=self._repo_root, check=True, text=True, capture_output=True)

