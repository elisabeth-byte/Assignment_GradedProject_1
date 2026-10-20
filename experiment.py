import numpy as np

from parameters import EPISODES
from q_learning import QLearningAgent
from world import GOAL, shortest_path_len


# Seeds used for each experiment
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

first_goals = []
goals = []
pits = []
step_limits = []
greedy_lengths = []


for seed in SEEDS:
    agent = QLearningAgent(seed=seed)
    agent.train(EPISODES)

    # Find the first episode that reached the goal
    first_goal = None

    for episode_number, (_, outcome) in enumerate(agent.history, start=1):
        if outcome == "goal":
            first_goal = episode_number
            break

    # Count outcomes
    goal_count = sum(
        outcome == "goal"
        for _, outcome in agent.history
    )

    pit_count = sum(
        outcome == "pit"
        for _, outcome in agent.history
    )

    step_limit_count = sum(
        outcome == "step limit"
        for _, outcome in agent.history
    )

    # Final greedy path
    path = agent.greedy_path()

    if path[-1] == GOAL:
        greedy_length = len(path) - 1
    else:
        greedy_length = None

    # Save results
    if first_goal is not None:
        first_goals.append(first_goal)

    goals.append(goal_count)
    pits.append(pit_count)
    step_limits.append(step_limit_count)

    if greedy_length is not None:
        greedy_lengths.append(greedy_length)

    # Show each seed
    print(
        f"Seed {seed}: "
        f"first goal = {first_goal}, "
        f"goals = {goal_count}, "
        f"pits = {pit_count}, "
        f"step limits = {step_limit_count}, "
        f"greedy path = {greedy_length}"
    )


# ==============================================================
# SUMMARY
# ==============================================================

print("\nRESULTS")
print("----------------------------")

print(
    f"First goal: "
    f"{np.mean(first_goals):.1f} +/- {np.std(first_goals):.1f} episodes"
)

print(
    f"Goals: "
    f"{np.mean(goals):.1f} +/- {np.std(goals):.1f}"
)

print(
    f"Pits: "
    f"{np.mean(pits):.1f} +/- {np.std(pits):.1f}"
)

print(
    f"Step limits: "
    f"{np.mean(step_limits):.1f} +/- {np.std(step_limits):.1f}"
)

print(
    f"Greedy path: "
    f"{np.mean(greedy_lengths):.1f} +/- {np.std(greedy_lengths):.1f} steps"
)

print(
    f"Shortest possible path: {shortest_path_len()} steps"
)