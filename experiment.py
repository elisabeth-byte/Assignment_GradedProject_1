import numpy as np
import matplotlib.pyplot as plt

from parameters import EPISODES
from q_learning import QLearningAgent

# Seeds used to evaluate robustness across different random runs
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Number of evaluation episodes per policy
N_EVAL_EPISODES = 100

# Evaluate a policy multiple times and return the average reward
def average_evaluation(agent, method, n=N_EVAL_EPISODES):

    rewards = []

    for _ in range(n):
        rewards.append(method())

    return np.mean(rewards)

# Store evaluation results
q_results = []
rule_results = []
random_results = []

# Train and evaluate one agent for each seed
for seed in SEEDS:

    # Create a new agent with a reproducible random seed
    agent = QLearningAgent(seed=seed)

    # Train the agent using Q-learning
    agent.train(EPISODES)

    # Evaluate the learned greedy policy
    q_reward = average_evaluation(
        agent,
        agent.evaluate
    )

    # Evaluate the rule-based baseline policy
    rule_reward = average_evaluation(
        agent,
        agent.evaluate_rule_based
    )

    # Evaluate the random baseline policy
    random_reward = average_evaluation(
        agent,
        agent.evaluate_random
    )

    # Store results for statistical analysis
    q_results.append(q_reward)
    rule_results.append(rule_reward)
    random_results.append(random_reward)

    # Print results for the current seed
    print(
        f"Seed {seed}: "
        f"Q-learning = {q_reward:.1f}, "
        f"Rule-based = {rule_reward:.1f}, "
        f"Random = {random_reward:.1f}"
    )


# ==============================================================
# SUMMARY STATISTICS
# ==============================================================
print("\nRESULTS")
print("--------------------------------")

print(
    f"Q-learning: "
    f"{np.mean(q_results):.2f} ± {np.std(q_results):.2f}"
)

print(
    f"Rule-based: "
    f"{np.mean(rule_results):.2f} ± {np.std(rule_results):.2f}"
)

print(
    f"Random: "
    f"{np.mean(random_results):.2f} ± {np.std(random_results):.2f}"
)


# ==============================================================
# POLICY COMPARISON PLOT
# ==============================================================

methods = ["Q-learning", "Rule-based", "Random"]

# Average reward for each policy
means = [
    np.mean(q_results),
    np.mean(rule_results),
    np.mean(random_results),
]

# Standard deviation used as error bars
stds = [
    np.std(q_results),
    np.std(rule_results),
    np.std(random_results),
]

plt.figure(figsize=(8, 5))

colors = [
    "tab:blue",
    "tab:orange",
    "tab:green"
]

plt.bar(
    methods,
    means,
    yerr=stds,
    capsize=5,
    color=colors,
)

plt.ylabel("Average reward")
plt.title("Policy Comparison")
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

#plt.show()
plt.savefig("policy_comparison.png")
plt.close()