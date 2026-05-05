"""Once-per-session tool approval.

Per the design choice: when a tool call arrives, we ask the user once. If
they approve, we add the fully-qualified tool name to a set; subsequent
calls to that exact tool in this session run without prompting. Closing
the session forgets all approvals — there's no persistence to disk.

The approval prompt itself is rendered by the chat loop (which has the
console); this module only tracks state and exposes the policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ApprovalDecision(str, Enum):
    APPROVE_ONCE = "approve_once"
    APPROVE_SESSION = "approve_session"
    DENY = "deny"


@dataclass
class ApprovalState:
    approved: set[str] = field(default_factory=set)
    """Fully-qualified tool names approved for the rest of the session."""

    auto_approve_all: bool = False
    """If true, skip the prompt entirely (e.g. ``--yolo`` mode). Off by
    default; only the user can flip this on."""

    def is_approved(self, fq_name: str) -> bool:
        return self.auto_approve_all or fq_name in self.approved

    def record(self, fq_name: str, decision: ApprovalDecision) -> None:
        if decision == ApprovalDecision.APPROVE_SESSION:
            self.approved.add(fq_name)
        # APPROVE_ONCE doesn't add to the set; the next call re-prompts.
