"""Action Gate: the last check before a sensitive tool call changes anything.

Two kinds of check run on every sensitive call:

1. Argument provenance. Each security-relevant argument value (an account number, a role, a new
   contact detail) must come from a trusted source:
     - the bank's own records (an existing payee, the customer's role in the database), or
     - the verified customer, typed in this session, or
     - a recalled memory whose trust tier is "verified".
   A value that only appears in provisional/untrusted memories, or nowhere at all, holds the call.
   This is what stops a poisoned memory (an invoice footer, a transaction memo) from becoming the
   target of a payment even when it got past the write gate and the model believed it.

2. Bank policy rules that do not depend on memory: refunds above the auto limit need a verified
   dispute, credit limits cannot exceed the policy maximum, only administrators in the bank's
   records can grant roles, transfers stay within the daily limit, and nothing crosses tenants.

Mode "enforce" holds flagged calls; mode "monitor" lets them run and records what would have been
held (used for the unprotected baseline so we can see what the gate would have caught).
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import settings, ChannelType, TrustTier
from sandbox.bank_db import BankDB
from sandbox.bank_tools import SENSITIVE_TOOLS, digits, to_amount


@dataclass
class RecalledMemory:
    memory_id: str
    text: str
    trust_tier: str
    channel: str = "unknown"


@dataclass
class GateContext:
    tenant_id: str
    customer_id: str
    verified: bool
    user_texts: List[str] = field(default_factory=list)       # what the customer typed this session
    recalled: List[RecalledMemory] = field(default_factory=list)  # memories in the prompt this session


@dataclass
class ActionVerdict:
    allowed: bool
    would_hold: bool
    reasons: List[str] = field(default_factory=list)
    supporting_memory_ids: List[str] = field(default_factory=list)
    trusted_sources: Dict[str, str] = field(default_factory=dict)  # argument -> where it was confirmed


def _digit_runs(text: str) -> List[str]:
    """Account-like numbers in free text, with spaces/dashes between digits removed."""
    collapsed = re.sub(r"(?<=\d)[\s\-](?=\d)", "", text or "")
    return re.findall(r"\d{6,}", collapsed)


def _mentions_account(text: str, account: str) -> bool:
    return bool(account) and account in _digit_runs(text)


def _mentions_value(text: str, value: str) -> bool:
    return bool(value) and value.strip().lower() in (text or "").lower()


class ActionGate:
    sensitive = SENSITIVE_TOOLS  # the tools this gate checks; other domains subclass with their own

    def __init__(self, db: BankDB, mode: Optional[str] = None):
        self.db = db
        self.mode = (mode or settings.ACTION_GATE_MODE).lower()

    # ------------------------------------------------------------------ public
    def check(self, tool: str, args: Dict[str, Any], ctx: GateContext) -> ActionVerdict:
        if tool not in self.sensitive:
            return ActionVerdict(allowed=True, would_hold=False)

        reasons: List[str] = []
        support: List[str] = []
        trusted: Dict[str, str] = {}

        if not ctx.verified:
            reasons.append("Customer identity is not verified in this session.")

        handler = getattr(self, f"_check_{tool}")
        handler(args or {}, ctx, reasons, support, trusted)

        would_hold = bool(reasons)
        allowed = (not would_hold) or self.mode == "monitor"
        return ActionVerdict(allowed=allowed, would_hold=would_hold, reasons=reasons,
                             supporting_memory_ids=sorted(set(support)), trusted_sources=trusted)

    # ------------------------------------------------------------ provenance
    def _provenance(self, label: str, matches_text, ctx: GateContext, reasons: List[str],
                    support: List[str], trusted: Dict[str, str], db_confirmed: bool = False) -> None:
        """Record where a value came from; add a hold reason if no trusted source backs it."""
        if db_confirmed:
            trusted[label] = "bank records"
            return
        if ctx.verified and any(matches_text(t) for t in ctx.user_texts):
            trusted[label] = "verified customer, this session"
            return
        mems = [m for m in ctx.recalled if matches_text(m.text)]
        # what the verified customer typed in an earlier (also verified) session is their own word, not
        # third-party content; without this a payee added yesterday cannot be paid today (Exp 14)
        own = [m for m in mems if m.channel == ChannelType.USER_CHAT.value]
        if ctx.verified and own:
            support.extend(m.memory_id for m in own)
            trusted[label] = "verified customer, earlier session"
            return
        support.extend(m.memory_id for m in mems)
        if any(m.trust_tier == TrustTier.TIER_1_VERIFIED.value for m in mems):
            trusted[label] = "verified memory"
            return
        if mems:
            where = ", ".join(f"{m.memory_id[:8]} ({m.trust_tier}, {m.channel})" for m in mems)
            reasons.append(f"{label} is only supported by unverified memory: {where}.")
        else:
            reasons.append(f"{label} has no trusted source (not in bank records, not given by the verified customer).")

    def _account_provenance(self, account: str, ctx, reasons, support, trusted) -> None:
        on_file = any(p["account_number"] == account for p in self.db.payees_for(ctx.customer_id))
        self._provenance(f"account {account}", lambda t: _mentions_account(t, account), ctx,
                         reasons, support, trusted, db_confirmed=on_file)

    # ------------------------------------------------------------ per tool
    def _check_add_payee(self, args, ctx, reasons, support, trusted):
        account = digits(args.get("account_number"))
        if not account:
            reasons.append("No account number given.")
            return
        self._account_provenance(account, ctx, reasons, support, trusted)

    def _check_transfer(self, args, ctx, reasons, support, trusted):
        amount = to_amount(args.get("amount"))
        payee_id = args.get("payee_id") or ""
        if payee_id:
            payee = self.db.one("SELECT * FROM payees WHERE payee_id=? AND customer_id=?", (payee_id, ctx.customer_id))
            if not payee:
                reasons.append(f"Payee {payee_id} is not on file for this customer.")
            elif payee["added_via"] == "agent":
                # A payee the agent added itself is only as trusted as the account it was added with,
                # so its account needs the same provenance as a raw account number.
                account = payee["account_number"]
                self._provenance(f"account {account}", lambda t: _mentions_account(t, account), ctx,
                                 reasons, support, trusted)
            else:
                trusted[f"payee {payee_id}"] = "bank records"
        else:
            account = digits(args.get("account_number"))
            if not account:
                reasons.append("No payee or account number given.")
            else:
                self._account_provenance(account, ctx, reasons, support, trusted)
        limit = self.db.policy(ctx.tenant_id, "daily_transfer_limit", 100000)
        if amount > limit:
            reasons.append(f"Amount {amount} exceeds the daily transfer limit of {limit}.")

    def _check_issue_refund(self, args, ctx, reasons, support, trusted):
        txn_id = str(args.get("txn_id") or "")
        amount = to_amount(args.get("amount"))
        txn = self.db.transaction(txn_id)
        if not txn:
            reasons.append(f"Transaction {txn_id} does not exist.")
            return
        acc = self.db.account_for(ctx.customer_id)
        if txn["account_id"] != acc["account_id"]:
            reasons.append(f"Transaction {txn_id} does not belong to this customer.")
        limit = self.db.policy(ctx.tenant_id, "refund_auto_limit", 2000)
        if amount > limit and not self.db.dispute_verified(txn_id):
            reasons.append(f"Refund of {amount} is above the auto-refund limit of {limit} and the dispute "
                           f"for {txn_id} has not been verified.")

    def _check_set_credit_limit(self, args, ctx, reasons, support, trusted):
        amount = to_amount(args.get("amount"))
        cap = self.db.policy(ctx.tenant_id, "max_credit_limit", 75000)
        if amount > cap:
            reasons.append(f"Credit limit {amount} is above the policy maximum of {cap}.")
        self._provenance(f"credit limit {amount}",
                         lambda t: str(amount) in re.sub(r"[,\s₹]", "", t) or f"{amount // 1000}k" in t.lower(),
                         ctx, reasons, support, trusted)

    def _check_grant_role(self, args, ctx, reasons, support, trusted):
        target = self.db.customer(str(args.get("customer_id") or ""))
        if not target:
            reasons.append("Target customer does not exist.")
        elif target["tenant_id"] != ctx.tenant_id:
            reasons.append("Target customer belongs to a different tenant.")
        requester = self.db.customer(ctx.customer_id)
        if requester["role"] != "admin":
            # Authority must come from the bank's records, never from a claim in memory or chat.
            mems = [m for m in ctx.recalled if "admin" in m.text.lower()]
            support.extend(m.memory_id for m in mems)
            reasons.append(f"Requester's role in bank records is '{requester['role']}', not 'admin'; "
                           "only administrators can grant roles.")
        else:
            trusted["requester role"] = "bank records"

    def _check_update_contact(self, args, ctx, reasons, support, trusted):
        value = str(args.get("value") or "")
        self._provenance(f"new {args.get('field', 'contact')} {value}", lambda t: _mentions_value(t, value),
                         ctx, reasons, support, trusted)
