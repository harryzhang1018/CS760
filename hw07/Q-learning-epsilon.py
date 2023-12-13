import numpy as np
from collections import defaultdict

class QLearning:

    def __init__(self):
        # The Q table with entries Q(s, a)
        self.Q = defaultdict(lambda: np.zeros(2))  # 2 actions: 0 for move, 1 for stay
        self.gamma = 0.8  # Discount factor
        self.epsilon = 0.5  # Epsilon for epsilon-greedy policy

    def epsilon_greedy_action(self, state):
        epsilon = self.epsilon  # Assuming self.options is defined and contains epsilon
        action_probs = np.zeros(2)
        
        for i in range(2):
            if i == np.argmax(self.Q[state]):
                action_probs[i] = 1 - epsilon + epsilon / 2
            else:
                action_probs[i] = epsilon / 2

        action = np.random.choice(np.arange(2), p=action_probs)
        return action

    def train(self, steps):
        state = "A"  # Initial state
        actions = ["move", "stay"]
        
        for _ in range(steps):
            # Choose an action using epsilon-greedy policy
            action = self.epsilon_greedy_action(state)
            
            # Take the chosen action
            chosen_action = actions[action]
            
            # Perform the action and observe the next state and reward
            next_state = "B" if chosen_action == "move" else "A"
            reward = 0 if chosen_action == "move" else 1
            
            # Update the Q value using the Q-learning update rule
            td_target = reward + self.gamma * np.max(self.Q[next_state])
            td_delta = td_target - self.Q[state][action]
            self.Q[state][action] += 0.1 * td_delta  # Assuming alpha = 0.1 for learning rate
            
            # Move to the next state
            state = next_state

        # Print the action-value table at the end
        print("Action-Value Table:")
        for s in self.Q.keys():
            print(f"State {s}: Move Q-value={self.Q[s][0]}, Stay Q-value={self.Q[s][1]}")


# Run Q-learning for 200 steps
ql = QLearning()
ql.train(200)