# ♟️ Machine Learning Chess Bot in Python

### 🔹 Overview
This project implements a **Machine Learning–based Chess Bot** developed in **Python** that learns and improves its move selection through data-driven evaluation.  
Instead of relying solely on fixed heuristics, the bot analyzes thousands of move–outcome pairs and applies trained models to predict the most promising moves in real time.

It combines the logic of **rule-based chess mechanics** with **supervised learning**, forming an intelligent hybrid system capable of adaptive strategy and improved decision-making over repeated games.

---

### 🧠 Core Features
- 🧩 **Machine Learning Evaluation Model** – Predicts optimal moves based on board-state data.  
- ♞ **Hybrid ML + Rule-Based Logic** – Integrates classical search with learned move prediction.  
- 📊 **Data Pipeline** – Stores past game data for continuous training and pattern discovery.  
- 🤖 **Self-Play Training Mode** – Generates labeled training data autonomously.  
- 🧮 **Custom Scoring Mechanism** – Uses weighted positional analysis learned from training data.  
- 🔁 **Turn-Based Play** – Human vs Bot and Bot vs Bot supported.  

---

### 🧰 Tech Stack
| Category | Tools / Frameworks |
|-----------|--------------------|
| **Programming Language** | Python |
| **Libraries** | `scikit-learn`, `numpy`, `pandas`, `python-chess`, `matplotlib` |
| **Concepts Used** | Supervised Learning, Feature Engineering, Move Evaluation, Reinforcement Concepts |
| **Version Control** | Git & GitHub |

---

### 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ml-chess-bot.git
   cd ml-chess-bot
