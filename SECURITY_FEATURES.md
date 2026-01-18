# 🔐 Security Features - Quick Reference

## Implemented Features

**Server-Side (strategy.py):**
- ✅ **Byzantine-Robust Aggregation** - Detects and rejects malicious updates
- ✅ **Trust Score Prediction** - ML-based early warning system

**Client-Side (client.py):**
- ✅ **Differential Privacy** - Adds calibrated noise to gradients for privacy
- ✅ **Advanced Attacks** - Simulates backdoor, poisoning, Sybil, Byzantine attacks

---

## 🚀 Quick Setup

### 1. Clone Repository

```bash
git clone <repository-url>
cd FederatedLearning_6G
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
# Run a quick test
python main.py --data synthetic_network_dataset.csv \
    --num-clients 12 --rounds 5 --cpu
```

---

## 1. Byzantine-Robust Aggregation

**Purpose:** Detect and reject malicious client updates on the server

### Quick Start

```bash
# Recommended: Multi-Krum (best balance)
python main.py --data synthetic_network_dataset.csv \
    --aggregation multi-krum --num-clients 12 --rounds 20

# Maximum robustness: Median (tolerates 50% malicious)
python main.py --data synthetic_network_dataset.csv \
    --aggregation median --num-clients 12 --rounds 20
```

### All Methods

| Method | Command | Best For | Robustness |
|--------|---------|----------|------------|
| **FedAvg** | `--aggregation fedavg` | Baseline (no defense) | None |
| **Krum** | `--aggregation krum` | <20% malicious | Medium |
| **Multi-Krum** | `--aggregation multi-krum` | General use ⭐ | High |
| **Trimmed Mean** | `--aggregation trimmed-mean` | Many clients (20+) | High |
| **Median** | `--aggregation median` | >30% malicious | Maximum |

### What to Expect

```
[BYZANTINE DETECTION] Potential Byzantine clients: [2, 7, 9]
[ROBUSTNESS] Update dispersion score: 0.234
```

---

## 2. Differential Privacy

**Purpose:** Add noise to gradients to protect client data privacy

### Quick Start

```bash
# Balanced privacy/accuracy (recommended)
python main.py --data synthetic_network_dataset.csv \
    --use-dp --dp-epsilon 1.0 --dp-delta 1e-5

# Strong privacy
python main.py --data synthetic_network_dataset.csv \
    --use-dp --dp-epsilon 0.1 --dp-delta 1e-6
```

### Privacy Budgets

| Epsilon (ε) | Privacy Level | Use Case |
|-------------|---------------|----------|
| **0.1** | Very strong | Healthcare, finance |
| **1.0** | Strong (recommended) | General sensitive data |
| **5.0** | Moderate | Less sensitive data |
| **10.0** | Weak | Testing only |

**Delta (δ):** Usually `1e-5` (good default)

---

## 3. Trust Score Prediction

**Purpose:** Predict future client behavior, enable early warnings

### Quick Start

```bash
python main.py --data synthetic_network_dataset.csv \
    --use-trust-prediction --num-clients 12 --rounds 20
```

### Output

```
[EARLY WARNING] 3 clients with degrading trust:
  Client 5: declining_trust (severity: medium)
  Client 2: low_trust (severity: high)
```

**Note:** Requires 5+ rounds to start making predictions

---

## 4. Attack Simulations

**Purpose:** Test defenses against realistic attacks

### Available Attacks

```bash
# Backdoor attack (trigger patterns)
python main.py --data synthetic_network_dataset.csv \
    --attack-type backdoor --malicious-frac 0.2

# Model poisoning (degrade performance)
python main.py --data synthetic_network_dataset.csv \
    --attack-type poisoning --malicious-frac 0.15

# Sybil attack (multiple fake clients)
python main.py --data synthetic_network_dataset.csv \
    --attack-type sybil --malicious-frac 0.3

# Byzantine attack (random malicious)
python main.py --data synthetic_network_dataset.csv \
    --attack-type byzantine --malicious-frac 0.25
```

---

## Complete Examples

### Example 1: Full Security Stack

```bash
python main.py --data synthetic_network_dataset.csv \
    --aggregation multi-krum \
    --use-dp --dp-epsilon 1.0 \
    --use-trust-prediction \
    --num-clients 12 --rounds 20 --cpu
```

**Enables:** Byzantine defense + Differential Privacy + Trust prediction

### Example 2: Test Under Attack

```bash
python main.py --data synthetic_network_dataset.csv \
    --aggregation multi-krum \
    --attack-type backdoor --malicious-frac 0.25 \
    --num-clients 12 --rounds 20
```

**Tests:** Byzantine defense against 25% backdoor attackers

### Example 3: Extreme Attack Scenario

```bash
python main.py --data synthetic_network_dataset.csv \
    --aggregation median \
    --attack-type sybil --malicious-frac 0.4 \
    --use-trust-prediction \
    --num-clients 20 --rounds 30
```

**Tests:** Maximum robustness against 40% Sybil attack

---

## Visualization

### Launch Dashboard

```bash
# View single run (default port 8050)
python interactive_dashboard.py artifacts_flower

# View with custom port
python interactive_dashboard.py artifacts_flower 8051

# Compare multiple runs
python interactive_dashboard.py baseline_artifacts 8050 &
python interactive_dashboard.py multikrum_artifacts 8051 &
python interactive_dashboard.py fullstack_artifacts 8052
```

Open browser: `http://localhost:8050` (or respective port)

---

## Architecture

**Client Side** (client.py):
- Gradient clipping (prevents gradient explosion)
- Differential Privacy noise (protects data)
- Attack simulation (label flipping for malicious clients)

**Server Side** (strategy.py):
- Byzantine detection (2-layer: anomaly signals + distance-based)
- Robust aggregation (Multi-Krum, Median, etc.)
- Trust score tracking and prediction

---

## Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| Training too slow | Reduce `--dp-epsilon` or disable DP |
| Low accuracy with defense | Normal trade-off, try Multi-Krum |
| No Byzantine detected | Increase `--malicious-frac` to 0.3+ |
| Trust prediction not showing | Need 5+ rounds minimum |
| Dashboard empty | Check CSV files in artifacts directory |

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                  CLIENT SIDE                        │
│  (Runs on each device - client.py)                  │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Client 1                  Client 2 (Malicious)     │
│  ┌──────────┐             ┌──────────┐              │
│  │ Train    │             │ Train    │              │
│  │ + DP     │             │ + Attack │              │
│  │ + Clip   │             │ + DP     │              │
│  └────┬─────┘             └────┬─────┘              │
│       │                        │                    │
│       │ Send update            │ Send malicious     │
│       │ [0.1, 0.2, 0.3]        │ [999, -999, 999]   │
│       │                        │                    │
└───────┼────────────────────────┼─────────────────── ┘
        │                        │
        │                        │
        ▼                        ▼
┌─────────────────────────────────────────────────────┐
│                  SERVER SIDE                        │
│  (Runs on central server - strategy.py)             │
├─────────────────────────────────────────────────────┤
│                                                     │
│  CustomStrategy.aggregate_fit()                     │
│  ┌───────────────────────────────────────────┐      │
│  │ Line 38: Initialize ByzantineDefense      │      │
│  │ self.byzantine_defense = ByzantineDefense()│     │
│  └───────────────────────────────────────────┘      │
│                    ▼                                │
│  ┌───────────────────────────────────────────┐     │
│  │ Line 184: Apply Multi-Krum                │     │
│  │ flat_aggregated = self.byzantine_defense  │     │
│  │     .multi_krum(flattened_params, m=5)    │     │
│  │                                           │     │
│  │ → Compares all client updates             │     │
│  │ → Selects top 5 most similar              │     │
│  │ → Rejects Client 2 (outlier!)             │     │
│  └───────────────────────────────────────────┘     │
│                    ▼                               │
│  ┌───────────────────────────────────────────┐     │
│  │ Line 213: Detect Byzantine Clients        │     │
│  │ byzantine_detected =                      │     │
│  │     detect_byzantine_clients(...)         │     │
│  │                                           │     │
│  │ → Identifies [4, 7, 3] as suspicious     │      │
│  │ → Reduces their trust scores              │     │
│  └───────────────────────────────────────────┘     │
│                    ▼                               │
│  ┌───────────────────────────────────────────┐     │
│  │ Return aggregated global model             │    │
│  │ (protected from malicious updates!)        │    │
│  └───────────────────────────────────────────┘     │
│                                                    │
└────────────────────────────────────────────────────┘
```