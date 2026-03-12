from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base import BaseAlgorithm, ControlType

class SARSA(BaseAlgorithm):
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
        Initialize the SARSA algorithm.

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
            control_type=ControlType.SARSA,
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
        next_obs: dict,
        next_action: int,
        done: bool
    ):
        """
        Update Q-values using SARSA .

        This method applies the SARSA update rule to improve policy decisions by updating the Q-table.
        """
        # 1. Discretize the current state and the next state
        state = self.discretize_state(obs)
        next_state = self.discretize_state(next_obs)
        
        # 2. Get the current Q-value for the state-action pair
        current_q = self.q_values[state][action]
        
        # 3. Determine the next Q-value
        if done:
            # If the episode is over, there is no next state or future reward
            next_q = 0.0
        else:
            # SARSA uses the Q-value of the ACTUAL next action taken by the epsilon-greedy policy
            next_q = self.q_values[next_state][next_action]
            
        # 4. Calculate the Temporal Difference (TD) Target
        td_target = reward + (self.discount_factor * next_q)
        
        # 5. Calculate the TD Error
        td_error = td_target - current_q
        
        # 6. Update the Q-value using the learning rate
        self.q_values[state][action] += self.lr * td_error
        
        # 7. (Optional) Log the error for your learning speed graphs
        self.training_error.append(td_error)
        