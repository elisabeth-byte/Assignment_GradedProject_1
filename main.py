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

if __name__ == "__main__":
    main()

