# MemGuard: AI Agent Security & Observability Platform

A practical guide to installing dependencies, running the local GPU LLM server (`vLLM`), launching the Streamlit dashboard, and accessing the web UI.

---

## 💻 System Prerequisites

* **OS:** Linux (Ubuntu 20.04/22.04 or Debian-based cloud instances like Brev, Lambda Labs, RunPod, AWS EC2).
* **GPU:** NVIDIA GPU with >= 16 GB VRAM (e.g., RTX A6000, A100, RTX 3090/4090).
* **Driver:** NVIDIA Driver `>= 550.x` (CUDA 12.x compatible).
* **Python:** Python 3.10.

---

## 🛠️ Step-by-Step Execution Guide

### Step 1: Clone Repository & Virtual Environment Setup

Open your terminal on your host or remote cloud machine:

```bash
# 1. Clone the repository
git clone -b exp17 https://github.com/saianoop324/AEGIS-2026.git
cd AEGIS-2026

# 2. Ensure virtual environment tools are installed (Debian/Ubuntu)
sudo apt update && sudo apt install -y python3-venv python3.10-venv python3-pip

# 3. Create and activate a fresh virtual environment
python3 -m venv venv
source venv/bin/activate
```

---

### Step 2: Install Dependencies & Verify Environment

Run the automated installer to set up the pinned **CUDA 12.4 PyTorch**, **vLLM (0.7.3)**, and **Transformers (4.49.0)**:

```bash
./install.sh
```

*(This automatically upgrades `pip`, installs `requirements.txt`, and runs a pre-flight test verifying your GPU).*

**Expected Output:**
```text
✅ PyTorch:      2.5.1+cu124 (CUDA Available: True - NVIDIA RTX A6000)
✅ Transformers: 4.49.0
✅ vLLM Engine:  0.7.3
🚀 Environment is 100% verified and ready!
```

---

### Step 3: Start the Local LLM Server (vLLM)

Start the local model server hosting **`Qwen/Qwen2.5-7B-Instruct`** on port `8011`:

```bash
./start_llm.sh
```

*(This runs vLLM with `--enable-auto-tool-choice`, `--tool-call-parser hermes`, and model aliases pre-configured for autonomous function calling).*

> **Keep this terminal window running.** Once the weights finish loading, you will see:
> `INFO: Application startup complete. Uvicorn running on http://0.0.0.0:8011`

---

### Step 4: Launch the Streamlit Dashboard

Open a **second terminal window** (or new SSH session) into the server:

```bash
cd ~/AEGIS-2026
source venv/bin/activate

./start_dashboard.sh
```

> Streamlit will launch and display:
> `Network URL: http://0.0.0.0:8501`

---

### Step 5: Open & View the UI on your Laptop

If running on a remote cloud GPU (e.g., Brev, AWS, RunPod):

1. **Port-Forward Port 8501 to your Local Machine:**
   In a terminal on your **local laptop** (not inside the remote server), run:

   ```bash
   # If using Brev CLI:
   brev port-forward aegis-instance -p 8501:8501

   # Or if using standard SSH:
   ssh -N -L 8501:localhost:8501 <username>@<REMOTE_SERVER_IP>
   ```

2. **Open the Web UI in your Browser:**
   Navigate to:
   👉 **[http://localhost:8501](http://localhost:8501)**

---

## 🖥️ Using the Dashboard

Once the web UI opens:

1. **💬 MemGuard Live Chatbot (Default Tab):**
   * The chatbot is connected directly to your local Qwen-2.5-7B model on port 8011.
   * You can test normal customer queries or click the one-click attack scenario buttons:
     * **💉 Prompt Injection**: Tests direct override attempts.
     * **👑 Authority Spoof**: Tests fake admin privilege claims.
     * **⚡ Contradiction**: Injects false data conflicting with verified bank policy.
     * **💸 Unauthorized Pay**: Attempts unauthorized wire transfers to attacker accounts.
     * **✅ Safe Question**: Runs legitimate banking queries.
   * Observe live telemetry badges showing **Write Gate** quarantine verdicts, **Read Gate** sandboxing, and **Action Gate** holds.

2. **🏦 BankDesk Experiments & 📊 Executive Scorecard:**
   * Review multi-turn benchmarks, Attack Success Rate (ASR) reductions, and latency metrics (<25ms SLA).

---

## 🛑 Stopping the Services

* **Stop the LLM Server:** In the vLLM terminal, press `Ctrl + C` (or run `pkill -f vllm`).
* **Stop the Dashboard:** In the Streamlit terminal, press `Ctrl + C` (or run `pkill -f streamlit`).
* **Close Port Forwarding:** On your laptop terminal, press `Ctrl + C`.
* **Verify GPU Release:** Run `nvidia-smi` on the server to verify GPU memory is released.
* **Stop Cloud Machine (Save costs):** Run `brev stop aegis-instance` when finished.
