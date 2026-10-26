import matplotlib.pyplot as plt

from parameters import EPISODES
from q_learning import QLearningAgent


def main():

    agent = QLearningAgent()

    print("Training started...")
    agent.train(EPISODES)

    print("Training finished.")
    print(agent.history[-10:])

    score = agent.evaluate()

    print(f"Greedy evaluation reward: {score:.2f}")

    print(f"Final epsilon: {agent.epsilon:.4f}")

    # LÆRING-KURVE
    plt.figure(figsize=(10, 5))
    plt.plot(agent.history)

    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Q-learning training performance")

    plt.grid(True)

    #plt.savefig("learning_curve.png")
    #plt.close()
    plt.show()

    #print("Learning curve saved as learning_curve.png")

    # MOVING AVERAGE LEARNING CURVE
    import numpy as np

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
    plt.show()

    q_reward = agent.evaluate()
    random_reward = agent.evaluate_random()
    rule_reward = agent.evaluate_rule_based()

    print()
    print(f"Q-learning reward: {q_reward:.2f}")
    print(f"Random policy reward: {random_reward:.2f}")
    print(f"Rule-based reward: {rule_reward:.2f}")
'''
    print()
    print("State (22,3,2):", agent.q[22, 3, 2])
    print("State (10,9,2):", agent.q[10, 9, 2])
    print("State (10,3,0):", agent.q[10, 3, 0])
    print("State (20,3,0):", agent.q[20, 3, 0])
    print("State (20,3,2):", agent.q[20, 3, 2])

    print("State (22,7,0):", agent.q[22, 7, 0])
    print("State (22,7,2):", agent.q[22, 7, 2])
    print("State (0, 2, 0):", agent.q[0, 2, 0])
    print("State (0, 3, 0):", agent.q[0, 3, 0])
    print("State (0, 4, 0):", agent.q[0, 4, 0])
'''

if __name__ == "__main__":
    main()

