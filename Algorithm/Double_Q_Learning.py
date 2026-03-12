from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base import BaseAlgorithm, ControlType

class Double_Q_Learning(BaseAlgorithm):
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
        Initialize the Double Q-Learning algorithm.

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
            control_type=ControlType.DOUBLE_Q_LEARNING,
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
        #========= put your code here =========#
        obs: dict,
        action: int,
        reward: float,
        next_obs: dict,
        done: bool
    ):
        """
        Update Q-values using Double Q-Learning.

        This method applies the Double Q-Learning update rule to improve policy decisions by updating the Q-table.
        """
        # 1. Discretize the current state and the next state
        state = self.discretize_state(obs)
        next_state = self.discretize_state(next_obs)
        
        # 2. Randomly decide which Q-table (QA or QB) to update
        if np.random.rand() < 0.5:
            # Update QA
            if done:
                td_target = reward
            else:
                # SELECT the best action using QA
                best_next_action_a = np.argmax(self.qa_values[next_state])
                # EVALUATE that action's value using QB
                td_target = reward + self.discount_factor * self.qb_values[next_state][best_next_action_a]
            
            # Calculate error and update QA
            td_error = td_target - self.qa_values[state][action]
            self.qa_values[state][action] += self.lr * td_error
            
        else:
            # Update QB
            if done:
                td_target = reward
            else:
                # SELECT the best action using QB
                best_next_action_b = np.argmax(self.qb_values[next_state])
                # EVALUATE that action's value using QA
                td_target = reward + self.discount_factor * self.qa_values[next_state][best_next_action_b]
            
            # Calculate error and update QB
            td_error = td_target - self.qb_values[state][action]
            self.qb_values[state][action] += self.lr * td_error
            
        # 3. Synchronize the main Q-table for the base class's epsilon-greedy policy
        # We average QA and QB to provide a stable value estimate for action selection
        self.q_values[state][action] = (self.qa_values[state][action] + self.qb_values[state][action]) / 2.0
        
        # 4. (Optional) Log the error for your learning speed graphs
        self.training_error.append(td_error)
        #======================================#
        