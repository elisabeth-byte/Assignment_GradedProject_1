import matplotlib.pyplot as plt
import numpy as np

from parameters import EPISODES
from q_learning import QLearningAgent


def main():

    agent = QLearningAgent()

    # Train the agent using Q-learning
    print("Training started...")
    agent.train(EPISODES)

    print("Training finished.")

    score = agent.evaluate()

    # Prints greedy evaluation reward and final epsilon
    print(f"Greedy evaluation reward: {score:.2f}")
    print(f"Final epsilon: {agent.epsilon:.4f}")

    # Plot the reward the agent obtained in each episode
    plt.figure(figsize=(10, 5))
    plt.plot(agent.history)

    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Q-learning training performance")

    plt.grid(True)

    plt.savefig("learning_curve.png")
    plt.close()
    #plt.show()
    print("Learning curve saved as learning_curve.png")

    # Smooth the learning curve using a 100-episode moving average
    window = 100
    moving_avg = np.convolve(
        agent.history,
        np.ones(window) / window,
        mode="valid"
    )

    plt.figure(figsize=(10, 5))
    plt.plot(moving_avg)

    plt.xlabel("Episode")
    plt.ylabel("Average reward")
    plt.title("Learning curve (100-episode moving average)")

    plt.grid(True)
    plt.savefig("learning_curve_moving_average.png")
    plt.close()
    #plt.show()

    print("\nPOLICY ANALYSIS")
    print("--------------------------------")

    # Example states used to inspect the learned policy
    test_states = [
        (0, 2, 0),    # early, low battery, low price
        (0, 7, 2),    # early, high battery, high price
        (12, 3, 0),   # midday, low battery, low price
        (12, 7, 2),   # midday, battery near target, high price
        (22, 7, 0),   # late, near target, low price
        (22, 9, 2),   # late, high battery, high price
        (22, 3, 0),  # late, low battery, low price
        (21, 3, 0),  # late, low battery, low price
        (20, 3, 0),  # late, low battery, low price
        (19, 3, 0),  # late, low battery, low price
    ]

    for state in test_states:

        # Select the greedy action learned for this state
        action = int(agent.q[state].argmax())

        # Convert action index to a readable label
        action_name = (
            "Charge"
            if action == 1
            else "Do not charge"
        )

        print(
            f"State {state}: "
            f"{action_name} "
            f"Q-values = {agent.q[state]}"
        )

if __name__ == "__main__":
    main()

