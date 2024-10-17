import numpy as np
import random
import copy

# by default, actions are ordered
# [left, up, down, right]
def get_cliff_mdp(k_total_states=38):
    """
    This method generates two 3-dimensional arrays (4 x 38 x 38) representing 
    action-based transition probabilities and corresponding rewards for a grid-based MDP.
    The actions follow this order: [left, up, down, right].
    """

    transitions = np.zeros((4, k_total_states, k_total_states))  # Placeholder for probability matrices
    payoffs = np.ones((4, k_total_states, k_total_states)) * -1  # Placeholder for reward matrices, initialized to -1

    for source_idx in range(k_total_states):
        pos = source_idx + 1

        # Left action handling
        if pos == 2:
            dest = 2
            reward = 0
        elif pos <= 5:
            dest = pos
            reward = -1
        else:
            dest = pos - 3
            reward = -1

        transitions[0, source_idx, dest - 1] = 1
        payoffs[0, source_idx, dest - 1] = reward

        # Up action logic
        if pos == 2:
            dest = 2
            reward = 0
        elif pos == 1:
            dest = 5
            reward = -1
        elif pos % 3 == 0:
            dest = pos
            reward = -1
        else:
            dest = pos - 1
            reward = -1

        transitions[1, source_idx, dest - 1] = 1
        payoffs[1, source_idx, dest - 1] = reward

        # Down action logic
        if pos == 2:
            dest = 2
            reward = 0
        elif pos == 1:
            dest = pos
            reward = -1
        elif pos == 5:
            dest = 1
            reward = -1
        elif pos == k_total_states:
            dest = 2
            reward = -1
        elif (pos + 1) % 3 == 0:
            dest = 1
            reward = -100
        else:
            dest = pos + 1
            reward = -1

        transitions[2, source_idx, dest - 1] = 1
        payoffs[2, source_idx, dest - 1] = reward

        # Right action handling
        if pos == 2:
            dest = 2
            reward = 0
        elif pos == 1:
            dest = 1
            reward = -100
        elif pos >= k_total_states - 2:
            dest = pos
            reward = -1
        else:
            dest = pos + 3
            reward = -1

        transitions[3, source_idx, dest - 1] = 1
        payoffs[3, source_idx, dest - 1] = reward

    return [transitions, payoffs]

# by default, actions are ordered
# [left, up, down, right]
def get_optimal_policy(k_total_states=38):
    """
    This function returns a 2D array (38 x 4) where each row represents the optimal probabilistic policy for each state.
    The action order is [left, up, down, right] by default.
    """
    num_of_actions = 4

    move_left = 0
    move_up = 1
    move_down = 2
    move_right = 3

    policy_matrix = np.zeros((k_total_states, num_of_actions))

    for state_index in range(k_total_states):
        current_state = state_index + 1

        if current_state == 2:  # goal state
            chosen_action = move_left  # arbitrary action
            action_probability = 1
        elif current_state == 1:  # only go up
            chosen_action = move_up
            action_probability = 1
        elif current_state >= k_total_states - 2:  # last few states -> down only
            chosen_action = move_down
            action_probability = 1
        elif (current_state + 1) % 3 == 0:  # bottom row -> move right only
            chosen_action = move_right
            action_probability = 1
        else:  # right as the default option
            chosen_action = move_right
            action_probability = 1

        policy_matrix[state_index, chosen_action] = action_probability

    # Ensure rows are normalized
    policy_matrix = normalize1_row(policy_matrix)

    return policy_matrix


def get_pi_e_greedy(policy_matrix, epsilon):
    """
    This function returns a modified policy matrix (38 x 4) where the greedy policy is adjusted with an epsilon value.
    Actions are still in the order [left, up, down, right].
    The highest probability action is set to 1 - epsilon, while the epsilon is distributed equally among the remaining actions.
    """

    epsilon_greedy_policy = np.copy(policy_matrix)

    for idx, action_probs in enumerate(policy_matrix):
        optimal_action = np.argmax(action_probs)

        if np.sum(action_probs == 1) != 1:
            print("Warning! Expected only one optimal action per state.")
            break

        other_actions = np.where(action_probs == 0)[0]

        epsilon_greedy_policy[idx, optimal_action] -= epsilon
        epsilon_greedy_policy[idx, other_actions] = epsilon / len(other_actions)

    # Normalize each row to ensure proper probability distribution
    epsilon_greedy_policy = normalize1_row(epsilon_greedy_policy)

    return epsilon_greedy_policy

def get_P_R(mdp_data, policy_matrix):
    """
    This function returns:
    - A transition matrix (38 x 38) that represents the probabilities of moving between states under a given policy;
    - A reward vector (38,) that captures the expected reward for each state.
    """

    transition_probs = mdp_data[0]
    reward_matrix = mdp_data[1]

    total_states = transition_probs.shape[1]  # 38 states
    total_actions = transition_probs.shape[0]  # 4 possible actions

    prob_matrix = np.zeros((total_states, total_states))
    reward_vector = np.zeros((total_states, total_states))

    for src_state_idx, row in enumerate(prob_matrix):
        action_probabilities = np.zeros((total_actions, total_states))
        state_rewards = np.zeros((total_actions, total_states))

        # Fill action probability and reward arrays for each action
        for action_idx in range(total_actions):
            action_probabilities[action_idx, :] = transition_probs[action_idx, src_state_idx]
            state_rewards[action_idx, :] = reward_matrix[action_idx, src_state_idx]

        policy_for_state = np.reshape(policy_matrix[src_state_idx], (1, -1))

        prob_matrix[src_state_idx, :] = policy_for_state @ action_probabilities
        reward_vector[src_state_idx, :] = policy_for_state @ (action_probabilities * state_rewards)

    expected_rewards = reward_vector @ np.ones(total_states)

    return prob_matrix, expected_rewards

def get_v(P, R, gamma):
    k_states = P.shape[0] 

    I = np.identity(k_states)

    v = np.linalg.inv(I - gamma*P)@R

    return v

# Generate an episode (trajectory) based on policy and starting conditions.
# The episode starts from the initial state and stops at the terminal state.
def gen_episode(mdp_data, policy, start_state, end_state, discount_factor=0.9):

    trajectory = [start_state]  # Store the sequence of states
    total_reward = 0  # Accumulated discounted reward

    transition_probs = mdp_data[0]
    rewards = mdp_data[1]

    state = start_state
    step_count = 0

    while state != end_state:
        step_count += 1

        current_policy = policy[state, :]
        rand_for_action = np.random.rand()

        chosen_action = get_value(current_policy, rand_for_action)

        rand_for_next_state = np.random.rand()
        current_probabilities = transition_probs[chosen_action, state]

        next_state = get_value(current_probabilities, rand_for_next_state)

        reward_for_transition = rewards[chosen_action, state, next_state]

        trajectory.append(next_state)

        total_reward += reward_for_transition * (discount_factor ** (step_count - 1))

        state = next_state

    return total_reward, step_count


def get_value(policy_distribution, random_val):
    cumulative_prob = np.cumsum(policy_distribution)
    selected_action = np.argmax(cumulative_prob >= random_val)
    return selected_action



def normalize1_row(matrix):
    normalized_matrix = matrix.copy()  # Copy to avoid modifying the input directly
    for idx, row in enumerate(matrix):
        row_sum = np.sum(row)
        if row_sum > 0:
            normalized_matrix[idx] = row / row_sum
    return normalized_matrix


def get_avg_std(data_sequence):
    data_np = np.array(data_sequence)
    avg_value = round(np.mean(data_np), 2)
    std_dev = round(np.std(data_np), 2)
    return avg_value, std_dev

if __name__ == "__main__":

    random.seed(1)  

    total_states = 38
    discount_factor = 0.9  
    start_state = 0

    mdp_cliff = get_cliff_mdp(total_states)  
    optimal_policy = get_optimal_policy(total_states)  

    # Modified code starts here
    epsilons = [0.0, 0.1, 0.2]
    episodes = 1000
    print("Total episodes: {}".format(episodes))

    for idx, epsilon in enumerate(epsilons):
        print("## Running for Epsilon: {} ##".format(epsilon))

        epsilon_greedy_policy = get_pi_e_greedy(optimal_policy, epsilon)
        [transition_matrix, expected_rewards] = get_P_R(mdp_cliff, epsilon_greedy_policy)
        
        state_values = get_v(transition_matrix, expected_rewards, discount_factor)

        print("State values:\n{}".format(state_values.round(2)))
        print("Initial state value (label {}): {}".format(start_state + 1, state_values[start_state].round(2)))

        episode_transitions = []
        total_discounted_reward = 0

        for episode in range(episodes):
            reward, transitions = gen_episode(mdp_cliff, epsilon_greedy_policy, start_state, 1, discount_factor)

            total_discounted_reward += reward
            episode_transitions.append(transitions)

        avg_discounted_reward = total_discounted_reward / episodes
        avg_transitions, std_transitions = get_avg_std(episode_transitions)

        print("Average discounted reward: {}".format(round(avg_discounted_reward, 2)))
        print("Average transitions: {} ± {}".format(avg_transitions, std_transitions))
