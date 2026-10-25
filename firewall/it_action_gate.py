"""Action gate for the ITDesk helpdesk agent (the deck's "forged 'user is admin' note" example).

Same contract as the bank's ActionGate: runs before every sensitive call, holds it (enforce) or
records what it would hold (monitor). Authority comes only from the directory: a role in the
employees table or an approval in the approvals table, never from a ticket, a KB article, a memory
or a claim in chat.

  add_to_group     privileged group: requester is it_admin in the directory AND an approval for that
                   employee and group is on record; other groups: only for yourself, or by an it_admin
  reset_password   yourself, or by an it_admin (the link always goes to the email on record)
  update_email     yourself only, and the new address must be typed by the verified employee
  disable_mfa      only by an it_admin
  unlock_account   yourself, or by an it_admin
  every call       target employee exists and is in the requester's tenant
"""

from typing import Any, Dict, List

from firewall.action_gate import ActionGate, GateContext, _mentions_value
from sandbox.it_tools import IT_SENSITIVE_TOOLS


class ITActionGate(ActionGate):
    sensitive = IT_SENSITIVE_TOOLS

    def _requester(self, ctx: GateContext) -> Dict[str, Any]:
        return self.db.employee(ctx.customer_id)

    def _target_ok(self, emp_id: str, ctx: GateContext, reasons: List[str]) -> bool:
        target = self.db.employee(str(emp_id or ""))
        if not target:
            reasons.append(f"Employee {emp_id} does not exist.")
            return False
        if target["tenant_id"] != ctx.tenant_id:
            reasons.append(f"Employee {emp_id} belongs to a different organisation.")
            return False
        return True

    def _admin_in_records(self, ctx: GateContext, support: List[str], what: str, reasons: List[str],
                          trusted: Dict[str, str]) -> bool:
        if self._requester(ctx)["role"] == "it_admin":
            trusted["requester role"] = "directory"
            return True
        # name any recalled memory claiming admin rights, so lineage shows what pushed for the call
        support.extend(m.memory_id for m in ctx.recalled if "admin" in m.text.lower())
        reasons.append(f"{what} needs an IT administrator; the requester's role in the directory is "
                       f"'{self._requester(ctx)['role']}'.")
        return False

    def _self_or_admin(self, emp_id, ctx, reasons, support, trusted, what: str) -> None:
        if not self._target_ok(emp_id, ctx, reasons):
            return
        if emp_id == ctx.customer_id:
            trusted["target"] = "the verified employee themself"
            return
        self._admin_in_records(ctx, support, f"{what} for another employee", reasons, trusted)

    # ------------------------------------------------------------ per tool
    def _check_add_to_group(self, args, ctx, reasons, support, trusted):
        emp_id, group = str(args.get("emp_id") or ""), str(args.get("group") or "")
        if not self._target_ok(emp_id, ctx, reasons):
            return
        g = self.db.group(group)
        if not g:
            reasons.append(f"Group {group} does not exist.")
            return
        if g["privileged"]:
            self._admin_in_records(ctx, support, f"Adding to privileged group {group}", reasons, trusted)
            if self.db.approved(emp_id, group):
                trusted[f"approval {emp_id}/{group}"] = "approvals on record"
            else:
                reasons.append(f"No approval on record for {emp_id} to join {group}.")
        elif emp_id != ctx.customer_id:
            self._admin_in_records(ctx, support, "Adding another employee to a group", reasons, trusted)

    def _check_reset_password(self, args, ctx, reasons, support, trusted):
        self._self_or_admin(str(args.get("emp_id") or ""), ctx, reasons, support, trusted, "A password reset")

    def _check_unlock_account(self, args, ctx, reasons, support, trusted):
        self._self_or_admin(str(args.get("emp_id") or ""), ctx, reasons, support, trusted, "Unlocking an account")

    def _check_disable_mfa(self, args, ctx, reasons, support, trusted):
        if self._target_ok(str(args.get("emp_id") or ""), ctx, reasons):
            self._admin_in_records(ctx, support, "Disabling MFA", reasons, trusted)

    def _check_update_email(self, args, ctx, reasons, support, trusted):
        emp_id, email = str(args.get("emp_id") or ""), str(args.get("email") or "")
        if not self._target_ok(emp_id, ctx, reasons):
            return
        if emp_id != ctx.customer_id:
            reasons.append("An employee's email can only be changed by that employee.")
        self._provenance(f"new email {email}", lambda t: _mentions_value(t, email), ctx, reasons, support, trusted)
