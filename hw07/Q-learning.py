import numpy as np
from collections import defaultdict

class QLearning:

    def __init__(self):
        # The Q table with entries Q(s, a)
        self.Q = defaultdict(lambda: np.zeros(2))  # 2 actions: 0 for move, 1 for stay
        self.gamma = 0.8  # Discount factor

    def train(self, steps):
        state = "A"  # Initial state
        actions = ["move", "stay"]
        
        for _ in range(steps):
            # Choose the best action according to the current action-value table
            best_action = np.argmax(self.Q[state])
            
            # If there is a tie, prefer move
            # print('Q[state]', self.Q[state])
            # print('Q[state][best_action]', self.Q[state][best_action])
            # print(self.Q[state] == self.Q[state][best_action])
            if np.sum(self.Q[state] == self.Q[state][best_action]) == 1:
                best_action = 0  # Choose move
            
            # Take the chosen action
            action = actions[best_action]
            
            # Perform the action and observe the next state and reward
            next_state = "B" if action == "move" else "A"
            reward = 0 if action == "move" else 1
            
            # Update the Q value using the Q-learning update rule
            td_target = reward + self.gamma * np.max(self.Q[next_state])
            td_delta = td_target - self.Q[state][best_action]
            self.Q[state][best_action] += 0.1 * td_delta  # Assuming alpha = 0.1 for learning rate
            
            # Move to the next state
            state = next_state
            print(f"State: {state}, Action: {action}, Reward: {reward}")

        # Print the action-value table at the end
        print("Action-Value Table:")
        for s in self.Q.keys():
            print(f"State {s}: Move Q-value={self.Q[s][0]}, Stay Q-value={self.Q[s][1]}")


# Run Q-learning for 200 steps
ql = QLearning()
ql.train(200)
