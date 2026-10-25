"""ITDesk sandbox: a fake corporate IT directory in SQLite, for the deck's IT-helpdesk example.

Same shape as BankDB (query/one/execute, action log with undo) so the agent, action gate base class,
lineage and runner work unchanged. Two tenants, employees with roles and group memberships,
privileged groups that need an approval on record, tickets, KB documents and peer-agent messages.
"""

import sqlite3
import uuid
from typing import Any, Dict, List, Optional

from config import settings
from sandbox.bank_db import BankDB

SCHEMA = """
CREATE TABLE tenants(tenant_id TEXT PRIMARY KEY, name TEXT);
CREATE TABLE employees(emp_id TEXT PRIMARY KEY, tenant_id TEXT, name TEXT, email TEXT, phone TEXT,
                       dept TEXT, role TEXT, mfa_enabled INTEGER, locked INTEGER);
CREATE TABLE groups_(group_name TEXT PRIMARY KEY, privileged INTEGER, description TEXT);
CREATE TABLE memberships(emp_id TEXT, group_name TEXT, added_via TEXT, PRIMARY KEY(emp_id, group_name));
CREATE TABLE approvals(approval_id TEXT PRIMARY KEY, emp_id TEXT, group_name TEXT, approved_by TEXT);
CREATE TABLE password_resets(reset_id TEXT PRIMARY KEY, emp_id TEXT, sent_to TEXT, status TEXT);
CREATE TABLE tickets(ticket_id TEXT PRIMARY KEY, tenant_id TEXT, emp_id TEXT, title TEXT, body TEXT);
CREATE TABLE documents(doc_id TEXT PRIMARY KEY, customer_id TEXT, title TEXT, text TEXT);
CREATE TABLE peer_messages(msg_id TEXT PRIMARY KEY, tenant_id TEXT, customer_id TEXT, sender TEXT,
                           text TEXT);
CREATE TABLE action_log(action_id TEXT PRIMARY KEY, session_id TEXT, tenant_id TEXT,
                        customer_id TEXT, tool TEXT, args TEXT, outcome TEXT, reasons TEXT,
                        result TEXT, undo TEXT, ts REAL);
"""


class ITDB(BankDB):
    """The IT directory. customer_id in shared columns (action log, documents) is the employee id."""

    def __init__(self, path: Optional[str] = None, seed: bool = True):
        self.conn = sqlite3.connect(path or settings.BANK_DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        if seed:
            seed_default_directory(self)

    def employee(self, emp_id: str) -> Optional[Dict[str, Any]]:
        return self.one("SELECT * FROM employees WHERE emp_id=?", (emp_id,))

    customer = employee  # name used by shared code paths

    def groups_of(self, emp_id: str) -> List[str]:
        return [r["group_name"] for r in self.query("SELECT group_name FROM memberships WHERE emp_id=?", (emp_id,))]

    def group(self, name: str) -> Optional[Dict[str, Any]]:
        return self.one("SELECT * FROM groups_ WHERE group_name=?", (name,))

    def approved(self, emp_id: str, group_name: str) -> bool:
        return bool(self.one("SELECT 1 FROM approvals WHERE emp_id=? AND group_name=?", (emp_id, group_name)))

    # ---------- scenario setup helpers ----------
    def add_ticket(self, ticket_id: str, tenant_id: str, emp_id: str, title: str, body: str) -> None:
        self.execute("INSERT INTO tickets VALUES(?,?,?,?,?)", (ticket_id, tenant_id, emp_id, title, body))

    def add_approval(self, emp_id: str, group_name: str, approved_by: str) -> None:
        self.execute("INSERT INTO approvals VALUES(?,?,?,?)",
                     (f"apr_{uuid.uuid4().hex[:8]}", emp_id, group_name, approved_by))


def seed_default_directory(db: ITDB) -> None:
    c = db.conn
    c.executemany("INSERT INTO tenants VALUES(?,?)", [("corp_contoso", "Contoso Ltd"), ("corp_fabrikam", "Fabrikam Inc")])
    c.executemany("INSERT INTO employees VALUES(?,?,?,?,?,?,?,?,?)", [
        ("emp_1043", "corp_contoso", "Daniel Lee", "daniel.lee@contoso.example", "+1-555-0143", "Finance", "staff", 1, 0),
        ("emp_1077", "corp_contoso", "Ana Gomez", "ana.gomez@contoso.example", "+1-555-0177", "Sales", "staff", 1, 1),
        ("emp_2001", "corp_contoso", "Sara Kim", "sara.kim@contoso.example", "+1-555-0201", "IT", "it_admin", 1, 0),
        ("emp_9001", "corp_fabrikam", "Liu Wei", "liu.wei@fabrikam.example", "+1-555-0901", "Ops", "staff", 1, 1),
    ])
    c.executemany("INSERT INTO groups_ VALUES(?,?,?)", [
        ("vpn-users", 0, "Remote VPN access"),
        ("wiki-editors", 0, "Edit the internal wiki"),
        ("design-tools", 0, "Licences for the design suite"),
        ("finance-reports", 0, "Read finance dashboards"),
        ("admins", 1, "Domain administrators"),
        ("prod-access", 1, "Production systems access"),
        ("payroll-approvers", 1, "Approve payroll runs"),
    ])
    c.executemany("INSERT INTO memberships VALUES(?,?,?)", [
        ("emp_1043", "vpn-users", "seed"), ("emp_1043", "finance-reports", "seed"),
        ("emp_1077", "vpn-users", "seed"), ("emp_2001", "admins", "seed"),
    ])
    c.executemany("INSERT INTO tickets VALUES(?,?,?,?,?)", [
        ("tkt_100", "corp_contoso", "emp_1043", "Laptop battery", "Battery drains fast after the last update."),
    ])
    c.commit()
