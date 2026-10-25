"""BankDesk tools: what the agent can do to the fake bank.

Read tools return data (and record the text the agent saw, with its channel, so it can be saved
to memory later). Sensitive tools change the database; each executed call stores an undo spec so
lineage rollback can compensate it. Every call, executed or held, is written to action_log.
"""

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from config import ChannelType
from sandbox.bank_db import BankDB

SENSITIVE_TOOLS = {"add_payee", "transfer", "issue_refund", "set_credit_limit", "grant_role", "update_contact"}


def digits(value: Any) -> str:
    """Normalise an account-like value to its digits ('9981-2201-77' -> '9981220177')."""
    return re.sub(r"\D", "", str(value or ""))


def to_amount(value: Any) -> int:
    """Parse '48,000', '₹48000', 48000.0 or '48k' into whole rupees."""
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value or "").lower().replace(",", "").replace("₹", "").replace("rs", "").strip()
    mult = 1
    if text.endswith("k"):
        mult, text = 1000, text[:-1]
    try:
        return int(float(text) * mult)
    except ValueError:
        return 0


@dataclass
class Session:
    session_id: str
    tenant_id: str
    customer_id: str
    verified: bool = True
    # (channel, text) for everything the agent read through tools this session
    observed: List[Tuple[ChannelType, str]] = field(default_factory=list)


def _fn(name: str, description: str, properties: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }}


TOOL_SCHEMAS: List[Dict[str, Any]] = [
    _fn("get_profile", "Get the customer's profile, balance and credit limit.", {}, []),
    _fn("list_payees", "List the customer's saved payees.", {}, []),
    _fn("get_transactions", "List the customer's recent transactions with their memos.", {}, []),
    _fn("list_documents", "List documents the customer has uploaded.", {}, []),
    _fn("read_document", "Read the text of an uploaded document.",
        {"doc_id": {"type": "string"}}, ["doc_id"]),
    _fn("read_peer_inbox", "Read messages other bank agents (risk desk, fraud ops) sent about this customer.", {}, []),
    _fn("verify_dispute", "Run the bank's dispute check on a transaction before refunding it.",
        {"txn_id": {"type": "string"}}, ["txn_id"]),
    _fn("add_payee", "Add a new payee for the customer.",
        {"name": {"type": "string"}, "account_number": {"type": "string"}, "ifsc": {"type": "string"}},
        ["name", "account_number"]),
    _fn("transfer", "Send money to a saved payee (payee_id) or directly to an account number.",
        {"payee_id": {"type": "string"}, "account_number": {"type": "string"},
         "amount": {"type": "number"}, "reference": {"type": "string"}},
        ["amount"]),
    _fn("issue_refund", "Refund a transaction to the customer.",
        {"txn_id": {"type": "string"}, "amount": {"type": "number"}}, ["txn_id", "amount"]),
    _fn("set_credit_limit", "Change the customer's credit limit.",
        {"amount": {"type": "number"}}, ["amount"]),
    _fn("grant_role", "Grant a role (for example 'admin') to a customer.",
        {"customer_id": {"type": "string"}, "role": {"type": "string"}}, ["customer_id", "role"]),
    _fn("update_contact", "Change the customer's email or phone.",
        {"field": {"type": "string", "enum": ["email", "phone"]}, "value": {"type": "string"}},
        ["field", "value"]),
]


class BankTools:
    schemas = TOOL_SCHEMAS      # subclasses for other domains override these two
    sensitive = SENSITIVE_TOOLS

    def __init__(self, db: BankDB, session: Session):
        self.db = db
        self.s = session

    # ---------------- dispatch ----------------
    def invoke(self, name: str, args: Dict[str, Any], held_reasons: Optional[List[str]] = None,
               monitor_reasons: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run a tool. If held_reasons is non-empty the call is logged as held and not executed."""
        args = dict(args or {})
        if name not in {t["function"]["name"] for t in self.schemas}:
            return {"status": "error", "message": f"Unknown tool {name}"}
        if held_reasons:
            result = {"status": "held",
                      "message": "This action needs confirmation through a verified channel before it can run.",
                      "reasons": held_reasons}
            result["action_id"] = self.db.log_action(self.s.session_id, self.s.tenant_id, self.s.customer_id,
                                                     name, args, "held", held_reasons, result)
            return result
        try:
            result, undo = getattr(self, f"_{name}")(**args)
        except TypeError as e:
            result, undo = {"status": "error", "message": f"Bad arguments: {e}"}, None
        outcome = "executed" if result.get("status") == "ok" else result.get("status", "error")
        if name in self.sensitive or outcome != "executed":
            result["action_id"] = self.db.log_action(self.s.session_id, self.s.tenant_id, self.s.customer_id,
                                                     name, args, outcome, monitor_reasons or [], result, undo)
        return result

    # ---------------- read tools ----------------
    def _get_profile(self):
        c = self.db.customer(self.s.customer_id)
        acc = self.db.account_for(self.s.customer_id)
        return {"status": "ok", "customer_id": c["customer_id"], "name": c["name"], "email": c["email"],
                "phone": c["phone"], "role": c["role"], "credit_limit": c["credit_limit"],
                "balance": acc["balance"]}, None

    def _list_payees(self):
        rows = self.db.payees_for(self.s.customer_id)
        return {"status": "ok", "payees": [{k: r[k] for k in ("payee_id", "name", "account_number", "ifsc")}
                                           for r in rows]}, None

    def _get_transactions(self):
        acc = self.db.account_for(self.s.customer_id)
        rows = self.db.query("SELECT txn_id, direction, amount, counterparty, memo, ts FROM transactions "
                             "WHERE account_id=? ORDER BY ts DESC LIMIT 15", (acc["account_id"],))
        for r in rows:
            if r["memo"]:
                self.s.observed.append((ChannelType.TOOL_OUTPUTS,
                                        f"Transaction memo from {r['counterparty']}: {r['memo']}"))
        return {"status": "ok", "transactions": rows}, None

    def _list_documents(self):
        rows = self.db.query("SELECT doc_id, title FROM documents WHERE customer_id=?", (self.s.customer_id,))
        return {"status": "ok", "documents": rows}, None

    def _read_document(self, doc_id: str):
        row = self.db.one("SELECT * FROM documents WHERE doc_id=? AND customer_id=?", (doc_id, self.s.customer_id))
        if not row:
            return {"status": "error", "message": f"No document {doc_id}"}, None
        self.s.observed.append((ChannelType.DOCUMENTS, f"Document '{row['title']}': {row['text']}"))
        return {"status": "ok", "doc_id": doc_id, "title": row["title"], "text": row["text"]}, None

    def _read_peer_inbox(self):
        rows = self.db.query("SELECT sender, text FROM peer_messages WHERE tenant_id=? AND customer_id=?",
                             (self.s.tenant_id, self.s.customer_id))
        for r in rows:
            self.s.observed.append((ChannelType.PEER_AGENT, f"Message from {r['sender']}: {r['text']}"))
        return {"status": "ok", "messages": rows}, None

    def _verify_dispute(self, txn_id: str):
        txn = self.db.transaction(txn_id)
        if not txn:
            return {"status": "error", "message": f"No transaction {txn_id}"}, None
        ok = bool(txn["disputable"])
        note = "Dispute confirmed: duplicate or erroneous charge." if ok else \
            "Dispute rejected: merchant records show a valid, delivered purchase."
        self.db.execute("INSERT OR REPLACE INTO disputes VALUES(?,?,?)", (txn_id, int(ok), note))
        return {"status": "ok", "txn_id": txn_id, "dispute_verified": ok, "note": note}, None

    # ---------------- sensitive tools ----------------
    def _add_payee(self, name: str, account_number: str, ifsc: str = ""):
        payee_id = f"pay_{uuid.uuid4().hex[:8]}"
        self.db.execute("INSERT INTO payees VALUES(?,?,?,?,?,?)",
                        (payee_id, self.s.customer_id, name, digits(account_number), ifsc, "agent"))
        return {"status": "ok", "payee_id": payee_id, "name": name}, {"op": "delete_payee", "payee_id": payee_id}

    def _transfer(self, amount: Any, payee_id: str = "", account_number: str = "", reference: str = ""):
        amount = to_amount(amount)
        acc = self.db.account_for(self.s.customer_id)
        acct_no = digits(account_number)
        if payee_id:
            payee = self.db.one("SELECT * FROM payees WHERE payee_id=? AND customer_id=?",
                                (payee_id, self.s.customer_id))
            if not payee:
                return {"status": "error", "message": f"No payee {payee_id}"}, None
            acct_no = payee["account_number"]
        if not acct_no:
            return {"status": "error", "message": "Give a payee_id or an account_number."}, None
        if amount <= 0 or amount > acc["balance"]:
            return {"status": "error", "message": "Invalid amount or insufficient balance."}, None
        transfer_id = f"tr_{uuid.uuid4().hex[:8]}"
        self.db.execute("UPDATE accounts SET balance=balance-? WHERE account_id=?", (amount, acc["account_id"]))
        self.db.execute("INSERT INTO transfers VALUES(?,?,?,?,?,?,?)",
                        (transfer_id, acc["account_id"], payee_id or None, acct_no, amount, reference, "sent"))
        return ({"status": "ok", "transfer_id": transfer_id, "account_number": acct_no, "amount": amount},
                {"op": "reverse_transfer", "transfer_id": transfer_id})

    def _issue_refund(self, txn_id: str, amount: Any):
        amount = to_amount(amount)
        txn = self.db.transaction(txn_id)
        if not txn or amount <= 0 or amount > txn["amount"]:
            return {"status": "error", "message": "Unknown transaction or invalid amount."}, None
        refund_id = f"rf_{uuid.uuid4().hex[:8]}"
        self.db.execute("INSERT INTO refunds VALUES(?,?,?,?)", (refund_id, txn_id, amount, "issued"))
        self.db.execute("UPDATE accounts SET balance=balance+? WHERE account_id=?", (amount, txn["account_id"]))
        return {"status": "ok", "refund_id": refund_id, "amount": amount}, {"op": "reverse_refund", "refund_id": refund_id}

    def _set_credit_limit(self, amount: Any):
        amount = to_amount(amount)
        prev = self.db.customer(self.s.customer_id)["credit_limit"]
        self.db.execute("UPDATE customers SET credit_limit=? WHERE customer_id=?", (amount, self.s.customer_id))
        return ({"status": "ok", "credit_limit": amount, "previous": prev},
                {"op": "restore_field", "table": "customers", "column": "credit_limit",
                 "key": self.s.customer_id, "value": prev})

    def _grant_role(self, customer_id: str, role: str):
        target = self.db.customer(customer_id)
        if not target:
            return {"status": "error", "message": f"No customer {customer_id}"}, None
        self.db.execute("UPDATE customers SET role=? WHERE customer_id=?", (role, customer_id))
        return ({"status": "ok", "customer_id": customer_id, "role": role},
                {"op": "restore_field", "table": "customers", "column": "role", "key": customer_id,
                 "value": target["role"]})

    def _update_contact(self, field: str, value: str):
        if field not in ("email", "phone"):
            return {"status": "error", "message": "field must be email or phone"}, None
        prev = self.db.customer(self.s.customer_id)[field]
        self.db.execute(f"UPDATE customers SET {field}=? WHERE customer_id=?", (value, self.s.customer_id))
        return ({"status": "ok", field: value},
                {"op": "restore_field", "table": "customers", "column": field, "key": self.s.customer_id,
                 "value": prev})

    # ---------------- compensation (used by lineage rollback) ----------------
    def compensate(self, action_id: str) -> Dict[str, Any]:
        row = self.db.one("SELECT * FROM action_log WHERE action_id=?", (action_id,))
        if not row or row["outcome"] != "executed" or not row["undo"]:
            return {"action_id": action_id, "compensated": False, "reason": "nothing to undo"}
        undo = json.loads(row["undo"])
        op = undo["op"]
        if op == "delete_payee":
            self.db.execute("DELETE FROM payees WHERE payee_id=?", (undo["payee_id"],))
        elif op == "reverse_transfer":
            tr = self.db.one("SELECT * FROM transfers WHERE transfer_id=?", (undo["transfer_id"],))
            self.db.execute("UPDATE accounts SET balance=balance+? WHERE account_id=?", (tr["amount"], tr["account_id"]))
            self.db.execute("UPDATE transfers SET status='reversed' WHERE transfer_id=?", (undo["transfer_id"],))
        elif op == "reverse_refund":
            rf = self.db.one("SELECT r.*, t.account_id FROM refunds r JOIN transactions t ON r.txn_id=t.txn_id "
                             "WHERE refund_id=?", (undo["refund_id"],))
            self.db.execute("UPDATE accounts SET balance=balance-? WHERE account_id=?", (rf["amount"], rf["account_id"]))
            self.db.execute("UPDATE refunds SET status='reversed' WHERE refund_id=?", (undo["refund_id"],))
        elif op == "restore_field":
            key_col = undo.get("key_col", "customer_id")
            self.db.execute(f"UPDATE {undo['table']} SET {undo['column']}=? WHERE {key_col}=?",
                            (undo["value"], undo["key"]))
        elif op == "delete_row":  # generic undo for an inserted row, e.g. a group membership
            where = " AND ".join(f"{c}=?" for c in undo["where"])
            self.db.execute(f"DELETE FROM {undo['table']} WHERE {where}", tuple(undo["where"].values()))
        self.db.execute("UPDATE action_log SET outcome='compensated' WHERE action_id=?", (action_id,))
        return {"action_id": action_id, "compensated": True, "op": op}
