# 🧪 RL Algorithms from Scratch
**PyTorch implementations of core Reinforcement Learning algorithms, built to understand the mathematical progression from Policy Gradients to Actor-Critic.**

This repository is a "Lab Notebook" documenting the trade-offs between **Bias**, **Variance**, and **Sample Efficiency** in Deep RL.

## 📊 The Evolution of Algorithms
I implemented three distinct variations to observe how changing the update target affects learning stability.

| Implementation | Type | Update Frequency | Key Characteristic | Pros/Cons |
| :--- | :--- | :--- | :--- | :--- |
| **1. [REINFORCE](./1_reinforce)** | Monte Carlo | End of Episode | Uses full episode return $G_t$. | ✅ Unbiased<br>❌ High Variance (Noisy) |
| **2. [Truncated PG](./2_policy_gradient_steps)** | N-Step | Every 10 Steps | **Experimental:** Updates on partial returns without a Critic. | ✅ Faster Updates<br>❌ **Biased/Myopic** (Ignores long-term future) |
| **3. [A2C](./3_a2c)** | Actor-Critic | Every 10 Steps | Uses $N$-step return + $V(s)$ bootstrap. | ✅ Low Variance & Low Bias<br>❌ Complex Implementation |

---

## 🛠️ Implementation Details

### 1. REINFORCE (Vanilla Policy Gradient)
*The baseline implementation.*
- **Method:** Uses full Monte Carlo returns. The agent must finish an entire episode before learning anything.
- **Optimization:** Implemented **Batching** (updating every 4-8 episodes) to average out the noise in the gradients.
- **Result:** Solves `CartPole-v1`, but training is jittery due to high variance in episode lengths.

### 2. Truncated Policy Gradient 
*An intermediate experiment between REINFORCE and A2C.*
- **Method:** Updates the policy every `N=10` steps using the immediate sum of rewards.
- **The "Bias" Lesson:** Unlike REINFORCE, this method cuts off the return calculation after 10 steps. Without a Value Function to estimate the rest of the future, the agent becomes **short-sighted**, prioritizing immediate survival over long-term balance.
- **Key Feature:** Implemented custom **KL Divergence** tracking to monitor how aggressively the policy updates.

### 3. A2C (N-step Advantage Actor-Critic)
*The robust solution.*
- **Method:** Solves the "Short-sighted" problem of Method #2 by introducing a **Critic Network**.
- **Bootstrapping:** $G_t = r_t + \gamma r_{t+1} + \dots + \gamma^n V(s_{t+n})$.
- **Shared Architecture:** Uses a single backbone network with split heads (Actor vs. Critic) for computational efficiency.
- **Status:** Currently migrating this agent to **Continuous Control** (Inverted Pendulum) using Gaussian Distributions.

- **Increasing N generally reduces bias but increases variance.**
---

## 📉 Performance & Metrics
All experiments are tracked using **Weights & Biases (WandB)**.

*Key metrics tracked:*
* **Entropy:** To monitor policy collapse (premature convergence).
* **Approx KL:** To ensure updates aren't destroying the policy.
* **Grad Norm:** To detect exploding gradients in the early phase.

## 🧰 Tech Stack
- **Framework:** PyTorch (MPS/Metal Acceleration enabled for Mac).
- **Environment:** Gymnasium (CartPole-v1) & MuJoCo (InvertedPendulum-v4).
- **Logging:** WandB.

---

### 🔜 Next Steps
- [ ] Complete **Sim-to-Real** transfer tests on Inverted Pendulum (Domain Randomization).
- [ ] Implement **PPO** (Proximal Policy Optimization) to fix the trust-region issue in A2C.

## 🔬 Experimental Observations

**Finding:**
Contrary to standard theory, **Batched REINFORCE outperformed A2C** on `CartPole-v1`.

**Analysis:**
1.  **Complexity Overhead:** A2C suffers from "non-stationary targets"—the Actor learns from a Critic that is itself still learning. On a simple task like CartPole, this added complexity slows down initial convergence compared to the unbiased Monte Carlo returns of REINFORCE.
2.  **The "Bias" of Truncated PG:** The Truncated Policy Gradient (10-step updates without a Critic) struggled because it is **myopic**. It optimizes for immediate survival (next 10 steps) but lacks a value signal for long-term stability.
3.  **Gradient Clipping:** Aggressive clipping (Norm=2.0) **degraded performance**. In CartPole, recovering from a near-fail state requires large gradient updates. Clipping prevented the policy from making these urgent corrections quickly.