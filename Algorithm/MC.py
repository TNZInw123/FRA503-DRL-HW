from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base import BaseAlgorithm, ControlType

class MC(BaseAlgorithm):
    def __init__(
            self,
            num_of_action: int,
            action_range: list,
            discretize_state_weight: list,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float,
    ) -> None:
        """
        Initialize the Monte Carlo algorithm.

        Args:
            num_of_action (int): Number of possible actions.
            action_range (list): Scaling factor for actions.
            discretize_state_weight (list): Scaling factor for discretizing states.
            learning_rate (float): Learning rate for Q-value updates.
            initial_epsilon (float): Initial value for epsilon in epsilon-greedy policy.
            epsilon_decay (float): Rate at which epsilon decays.
            final_epsilon (float): Minimum value for epsilon.
            discount_factor (float): Discount factor for future rewards.
        """
        super().__init__(
            control_type=ControlType.MONTE_CARLO,
            num_of_action=num_of_action,
            action_range=action_range,
            discretize_state_weight=discretize_state_weight,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )
        
    def update(
        self,
        obs: dict,
        action: int,
        reward: float,
        done: bool
    ):
        """
        Update Q-values using Monte Carlo.

        This method applies the Monte Carlo update rule to improve policy decisions by updating the Q-table.
        This method collects transitions during the episode. Once the episode ends, 
        it calculates the return G_t backwards and updates the Q-table.
        """
        # 1. Discretize the current state
        obs_dis = self.discretize_state(obs)
        
        # 2. Store the current transition in the agent's history
        self.obs_hist.append(obs_dis)
        self.action_hist.append(action)
        self.reward_hist.append(reward)
        
        # 3. Only perform the Q-value update if the episode is finished
        if done:
            G = 0
            # Iterate backwards through the episode to calculate returns
            for t in reversed(range(len(self.reward_hist))):
                state_t = self.obs_hist[t]
                action_t = self.action_hist[t]
                reward_t = self.reward_hist[t]
                
                # Calculate the cumulative discounted reward (Return G)
                # G_t = R_{t+1} + gamma * G_{t+1}
                G = reward_t + self.discount_factor * G
                
                # Increment the visit count for this state-action pair
                self.n_values[state_t][action_t] += 1
                
                # Calculate the error
                error = G - self.q_values[state_t][action_t]
                
                # Update Q-value using the learning rate (alpha)
                self.q_values[state_t][action_t] += self.lr * error
                
            # Optional: Log the final error of the episode for your learning speed graphs
            self.training_error.append(error)
                
            # 4. Clear the history lists to start fresh for the next episode
            self.obs_hist.clear()
            self.action_hist.clear()
            self.reward_hist.clear()
            