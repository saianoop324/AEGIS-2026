"""Standalone results viewer: streamlit run dashboard/bankdesk_app.py (no models loaded)."""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.bankdesk import render  # noqa: E402

st.set_page_config(page_title="MemGuard | Experiments", layout="wide")
render(st)
