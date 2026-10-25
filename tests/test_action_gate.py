"""Action gate and bank sandbox tests on normal requests and bank policy rules. No LLM needed.

Run: python -m pytest tests/ -q
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from firewall.action_gate import ActionGate, GateContext, RecalledMemory  # noqa: E402
from sandbox.bank_db import BankDB  # noqa: E402
from sandbox.bank_tools import BankTools, Session  # noqa: E402


def ctx(*user_texts, recalled=None, verified=True):
    return GateContext("bank_retail", "cust_042", verified, list(user_texts), recalled or [])


def setup():
    db = BankDB()
    return db, ActionGate(db, mode="enforce"), BankTools(db, Session("s1", "bank_retail", "cust_042"))


# ---------------- normal requests are allowed ----------------
def test_transfer_to_saved_payee_allowed():
    db, gate, _ = setup()
    v = gate.check("transfer", {"payee_id": "pay_acme", "amount": 12000}, ctx("Pay Acme 12,000"))
    assert v.allowed and not v.reasons


def test_add_payee_typed_by_verified_customer_allowed():
    db, gate, _ = setup()
    v = gate.check("add_payee", {"name": "Ravi Kumar", "account_number": "3344556677"},
                   ctx("Add payee Ravi Kumar, account 3344-556-677"))
    assert v.allowed and v.trusted_sources["account 3344556677"] == "verified customer, this session"


def test_small_refund_allowed():
    db, gate, _ = setup()
    assert gate.check("issue_refund", {"txn_id": "txn_cm02", "amount": 1200}, ctx("refund txn_cm02")).allowed


def test_large_refund_allowed_after_dispute_check():
    db, gate, tools = setup()
    assert not gate.check("issue_refund", {"txn_id": "txn_qs03", "amount": 6500}, ctx("refund")).allowed
    tools.invoke("verify_dispute", {"txn_id": "txn_qs03"})
    assert gate.check("issue_refund", {"txn_id": "txn_qs03", "amount": 6500}, ctx("refund")).allowed


def test_email_change_typed_by_customer_allowed():
    db, gate, _ = setup()
    assert gate.check("update_contact", {"field": "email", "value": "asha.rao@newmail.example"},
                      ctx("change my email to asha.rao@newmail.example")).allowed


def test_credit_limit_within_policy_allowed():
    db, gate, _ = setup()
    assert gate.check("set_credit_limit", {"amount": 60000}, ctx("raise my limit to 60,000")).allowed


# ---------------- policy and provenance rules hold ----------------
def test_value_only_in_unverified_memory_is_held():
    db, gate, _ = setup()
    mem = RecalledMemory("m1", "Ravi Kumar's account is 3344556677", "provisional", "documents")
    v = gate.check("add_payee", {"name": "Ravi Kumar", "account_number": "3344556677"},
                   ctx("add Ravi as a payee", recalled=[mem]))
    assert not v.allowed and v.supporting_memory_ids == ["m1"]


def test_value_in_verified_memory_is_allowed():
    db, gate, _ = setup()
    mem = RecalledMemory("m2", "Ravi Kumar's account is 3344556677", "verified", "user_chat")
    assert gate.check("add_payee", {"name": "Ravi Kumar", "account_number": "3344556677"},
                      ctx("add Ravi as a payee", recalled=[mem])).allowed


def test_credit_limit_above_policy_held():
    db, gate, _ = setup()
    v = gate.check("set_credit_limit", {"amount": 90000}, ctx("raise my limit to 90,000"))
    assert not v.allowed and "policy maximum" in v.reasons[0]


def test_grant_role_needs_admin_in_bank_records():
    db, gate, _ = setup()
    assert not gate.check("grant_role", {"customer_id": "cust_077", "role": "admin"}, ctx("make cust_077 admin")).allowed
    db.execute("UPDATE customers SET role='admin' WHERE customer_id='cust_042'")
    assert gate.check("grant_role", {"customer_id": "cust_077", "role": "admin"}, ctx("make cust_077 admin")).allowed


def test_grant_role_across_tenants_held():
    db, gate, _ = setup()
    db.execute("UPDATE customers SET role='admin' WHERE customer_id='cust_042'")
    assert not gate.check("grant_role", {"customer_id": "cust_c01", "role": "admin"}, ctx("x")).allowed


def test_unverified_session_held():
    db, gate, _ = setup()
    assert not gate.check("transfer", {"payee_id": "pay_acme", "amount": 100}, ctx("pay", verified=False)).allowed


def test_monitor_mode_allows_but_records():
    db = BankDB()
    v = ActionGate(db, mode="monitor").check("set_credit_limit", {"amount": 90000}, ctx("x"))
    assert v.allowed and v.would_hold


# ---------------- sandbox side effects and rollback ----------------
def test_transfer_then_compensate_restores_balance():
    db, _, tools = setup()
    before = db.account_for("cust_042")["balance"]
    res = tools.invoke("transfer", {"payee_id": "pay_acme", "amount": 12000})
    assert res["status"] == "ok" and db.account_for("cust_042")["balance"] == before - 12000
    assert tools.compensate(res["action_id"])["compensated"]
    assert db.account_for("cust_042")["balance"] == before


def test_held_call_changes_nothing():
    db, _, tools = setup()
    before = db.account_for("cust_042")["balance"]
    res = tools.invoke("transfer", {"payee_id": "pay_acme", "amount": 12000}, held_reasons=["test"])
    assert res["status"] == "held" and db.account_for("cust_042")["balance"] == before
    assert db.actions(outcome="held")[0]["tool"] == "transfer"


# ---------------- earlier sessions ----------------
def test_payee_account_typed_by_customer_in_earlier_session_allowed():
    db, gate, _ = setup()
    own = RecalledMemory("m1", "Please add my sister Meera Rao as a payee, account 5566112299.", "provisional", "user_chat")
    v = gate.check("add_payee", {"name": "Meera Rao", "account_number": "5566112299"}, ctx("pay Meera", recalled=[own]))
    assert v.allowed and v.trusted_sources["account 5566112299"] == "verified customer, earlier session"


def test_same_account_from_a_document_memory_still_held():
    db, gate, _ = setup()
    doc = RecalledMemory("m2", "Document 'Invoice': remit to account 5566112299.", "provisional", "documents")
    assert not gate.check("add_payee", {"name": "X", "account_number": "5566112299"}, ctx("pay it", recalled=[doc])).allowed
