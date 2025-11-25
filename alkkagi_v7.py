import kymnasium as kym
import gymnasium as gym
import numpy as np
from typing import Dict, Any
import collections
import random
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp

# Method only for Manual Play
# kymnasium.alkkagi.ManualPlayWrapper("kymnasium/AlKkaGi-3x3-v0", debug=True).play()

# v7 Patch Notes
# 1. Dead stone processing modified
# 2. Sorting removed
# 3. Distance-based reward removed
# ---------- Helper Functions ----------
def observation_to_input(observation, turn) :
    WIDTH = 600
    HEIGHT = 600
    
    if(turn == 0) :
        player = "black"
        opponent = "white"
    else :
        player = "white"
        opponent = "black"
    
    player_stones = np.array(observation[player], dtype=np.float32)
    for i in range(len(player_stones)) :
        if(player_stones[i][2] == 0) :
            player_stones[i][0] = -1.0
            player_stones[i][1] = -1.0
    player_stones = player_stones.flatten()  # (9, )
    
    # Sort opponent stones by y-coordinate
    opponent_stones = np.array(observation[opponent], dtype=np.float32)
    for i in range(len(opponent_stones)) :
        if(opponent_stones[i][2] == 0) :
            opponent_stones[i][0] = -1.0
            opponent_stones[i][1] = -1.0
    # opponent_stones = sorted(opponent_stones, key=lambda x: (-x[2], x[1])) # alive first, then by y-coordinate
    opponent_stones = np.array(opponent_stones, dtype=np.float32).flatten()  # (9, )
    
    obstacles = np.array(observation["obstacles"], dtype=np.float32).flatten()  # (12, )
    
    # Normalize
    player_stones[0::3] /= WIDTH
    player_stones[1::3] /= HEIGHT
    opponent_stones[0::3] /= WIDTH
    opponent_stones[1::3] /= HEIGHT
    obstacles[0::4] /= WIDTH
    obstacles[1::4] /= HEIGHT
    obstacles[2::4] /= WIDTH
    obstacles[3::4] /= HEIGHT
    
    # Dead stone masking
    valid_mask = player_stones[2::3]  # (3, ) e.g., [1., 0., 1.] -> alive, dead, alive
    
    return np.concatenate([player_stones, opponent_stones, obstacles]), valid_mask  # (30, ), (3, )

def build_actor_critic_model(input_shape) :
    inputs = keras.layers.Input(shape=input_shape, name="state_input")
    mask_input = keras.layers.Input(shape=(3,), name="mask_input")
    
    common = keras.layers.Dense(256, activation="relu")(inputs)
    common = keras.layers.Dense(256, activation="relu")(common)
    
    # --- actor ---
    # 1. Stone Selection - masking applied
    # selection = keras.layers.Dense(3, activation="softmax", name="selection_out")(common)
    selection_logits = keras.layers.Dense(3, name="selection_logits")(common)  # logits means "log-odds"
    # By adding a large negative number before softmax, we can effectively mask out invalid choices
    adder = keras.layers.Lambda(lambda x : (1.0 - x) * -1e9, name ="mask_adder")(mask_input)
    masked_selection_logits = keras.layers.Add(name="masked_selection_logits")([selection_logits, adder])
    # selection_probabilities = keras.layers.Activation("softmax", name="selection_probabilities")(masked_selection_logits)
    # -> NaN issue
    
    # 2. Power
    power_mean = keras.layers.Dense(3, activation="sigmoid", name="power_mean")(common)  # (0.0 ~ 1.0)
    # power_mean = keras.layers.Lambda(lambda x : x * 2500.0 + 1.0)(power_mean)  # (1.0 ~ 2501.0) -> clipped later
    power_std_dev = keras.layers.Dense(3, activation="softplus", name="power_std_dev")(common)  # must be positive
    power_std_dev = keras.layers.Lambda(lambda x : x + 0.001)(power_std_dev)  # avoid zero stddev
    
    # 3. Angle
    angle_mean = keras.layers.Dense(3, activation="tanh", name="angle_mean")(common)  # (-1.0 ~ 1.0)
    # angle_mean = keras.layers.Lambda(lambda x : x * 180.0)(angle_mean)  # (-180.0 ~ 180.0)
    angle_std_dev = keras.layers.Dense(3, activation="softplus", name="angle_std_dev")(common)  # must be positive
    angle_std_dev = keras.layers.Lambda(lambda x : x + 0.001)(angle_std_dev)  # avoid zero stddev
    # ----------
    
    # --- critic ---
    value = keras.layers.Dense(1, name="value_out")(common)
    model = keras.Model(
        inputs = [inputs, mask_input],
        outputs = [masked_selection_logits, power_mean, power_std_dev, angle_mean, angle_std_dev, value]
    )
    # ----------
    return model

def stone_count(observation, color) :
    return sum(1 for stone in observation[color] if stone[2] == 1)

def who_is_the_winner(observation) :
    black_survived = stone_count(observation, "black")
    white_survived = stone_count(observation, "white")
    if(black_survived > white_survived) : return "black"
    elif(white_survived > black_survived) : return "white"
    return "draw"

def calculate_returns(rewards, gamma=0.99) :
    returns = []
    G_t = 0
    for reward in reversed(rewards) :
        G_t = reward + gamma * G_t
        returns.insert(0, G_t)
    return returns

def train_by_records(agent, states, masks, actions, returns, optimizer) :
    action_indices = tf.cast(actions[:, 0], tf.int32)
    action_powers = tf.cast(actions[:, 1], tf.float32)
    action_angles = tf.cast(actions[:, 2], tf.float32)
    
    states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
    mask_tensor = tf.convert_to_tensor(masks, dtype=tf.float32)
    returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)
    
    with tf.GradientTape() as tape :
        # Model Inference
        selection_logits, power_means, power_std_devs, angle_means, angle_std_devs, values = agent.model([states_tensor, mask_tensor])
        values = tf.squeeze(values)
        
        # Calculate Advantage (G_t - V(s))
        advantages = returns_tensor - values
        # Advantage Normalization
        advantage_mean = tf.math.reduce_mean(advantages)
        advantage_std = tf.math.reduce_std(advantages) + 1e-8
        normalized_advantages = (advantages - advantage_mean) / advantage_std
        
        # Actor Loss
        # 1. Stone Selection (Categorical)
        # distribution_selection = tfp.distributions.Categorical(probs=selection_probs) -> NaN issue
        distribution_selection = tfp.distributions.Categorical(logits=selection_logits)
        log_probability_selection = distribution_selection.log_prob(action_indices)
        
        # 2. Power (Normal)
        selected_power_means = tf.gather(power_means, action_indices, batch_dims=1)
        selected_power_std_devs = tf.gather(power_std_devs, action_indices, batch_dims=1)
        distribution_power = tfp.distributions.Normal(loc=selected_power_means, scale=selected_power_std_devs)
        log_probability_power = distribution_power.log_prob(action_powers)
        
        # 3. Angle (Normal)
        selected_angle_means = tf.gather(angle_means, action_indices, batch_dims=1)
        selected_angle_std_devs = tf.gather(angle_std_devs, action_indices, batch_dims=1)
        distribution_angle = tfp.distributions.Normal(loc=selected_angle_means, scale=selected_angle_std_devs)
        log_probability_angle = distribution_angle.log_prob(action_angles)
        
        # Total Log Probability
        total_log_probability = log_probability_selection + log_probability_power + log_probability_angle
        
        # Policy Loss = - (log_prob * advantage)
        actor_loss = -tf.reduce_mean(total_log_probability * normalized_advantages)
        
        # 4. Critic Loss (Mean Squared Error)
        critic_loss = tf.reduce_mean(tf.square(advantages))
        
        # 5. Entropy Bonus (to encourage exploration)
        entropy = tf.reduce_mean(distribution_selection.entropy() + distribution_power.entropy() + distribution_angle.entropy())
        
        # Total Loss
        total_loss = actor_loss + (0.5 * critic_loss) - (0.01 * entropy)
        
    # Gradient Update
    gradients = tape.gradient(total_loss, agent.model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, agent.model.trainable_variables))
    
    return total_loss

def get_aiming_reward(observation, turn, selection, angle) :
    if(turn == 0) :
        players = observation["black"]
        opponents = observation["white"]
    else :
        players = observation["white"]
        opponents = observation["black"]
        
    player_stone = players[selection]
    if(player_stone[2] == 0) : return -1.0
    
    alive_opponents = [stone for stone in opponents if stone[2] == 1]
    if(not alive_opponents) : return 0.0  # this code should not be reached normally
    
    max_cosine_similarity = -1.0  # if aiming directly, cosine similarity is 1.0
    for opponent in alive_opponents :
        dx = opponent[0] - player_stone[0]
        dy = opponent[1] - player_stone[1]
        target_angle_degree = np.degrees(np.arctan2(dy, dx))
        cosine_similarity = np.cos(np.radians(target_angle_degree - angle))
        max_cosine_similarity = max(max_cosine_similarity, cosine_similarity)
        
    return max_cosine_similarity * 0.1  # range: [-0.1 ~ 0.1]
# -------- End of Helper Functions --------

# ---------- Agent Class ----------
class Agent(kym.Agent) :
    def __init__(self, model = None, path: str = None) :
        if(path) :
            self.model = keras.models.load_model(path, safe_mode=False)
        elif(model) :
            self.model = model
        else :
            self.model = build_actor_critic_model((30, ))  # input shape (30, )

    def save(self, path: str) :
        self.model.save(path)

    @classmethod
    def load(cls, path: str) -> "kym.Agent" :
        return cls(path=path)

    def act(self, observation: Any, info: Dict, deterministic: bool = False) -> Any :
        turn = observation["turn"]
        state, valid_mask = observation_to_input(observation, turn)
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        mask_tensor = tf.convert_to_tensor([valid_mask], dtype=tf.float32)
        
        # Model Inference
        selection_logits, power_mean, power_std_dev, angle_mean, angle_std_dev, value = self.model([state_tensor, mask_tensor])
        
        # Stone Selection
        if(deterministic) : selected_index = np.argmax(selection_logits.numpy()[0])
        else :
            # selected_probabilities = selection_probabilities.numpy()[0] -> NaN issue
            selected_probabilities = tf.nn.softmax(selection_logits).numpy()[0]
            selected_index = np.random.choice(3, p=selected_probabilities)
        
        # Power Sampling (Gaussian)
        selected_power_mean = power_mean.numpy()[0][selected_index]
        selected_power_std_dev = power_std_dev.numpy()[0][selected_index]
        if deterministic : raw_power = selected_power_mean
        else : raw_power = np.random.normal(selected_power_mean, selected_power_std_dev)
        raw_power = np.clip(raw_power, 0.0, 1.0)
        # Angle Sampling (Gaussian)
        selected_angle_mean = angle_mean.numpy()[0][selected_index]
        selected_angle_std_dev = angle_std_dev.numpy()[0][selected_index]
        if deterministic : raw_angle = selected_angle_mean
        else : raw_angle = np.random.normal(selected_angle_mean, selected_angle_std_dev)
        raw_angle = np.clip(raw_angle, -1.0, 1.0)
        
        # RescaleW
        power = raw_power * 2500.0
        power = max(1.0, power)
        angle = raw_angle * 180.0
        
        return {
            "turn" : turn,
            "index" : int(selected_index),
            "power" : float(power),
            "angle" : float(angle)
        }
    
class BlackAgent(Agent) :
    pass

class WhiteAgent(Agent) :
    pass
# -------- End of Agent Class --------

# ---------- Training & Testing ----------
def train() :
    # Environment
    env = gym.make(
        id = "kymnasium/AlKkaGi-3x3-v0",
        render_mode = "rgb_array",
        bgm = False,
        obs_type = "custom"
    )
    
    black_agent = BlackAgent()
    white_agent = WhiteAgent()
    
    black_optimizer = keras.optimizers.Adam(learning_rate = 0.0001, clipnorm=1.0)
    white_optimizer = keras.optimizers.Adam(learning_rate = 0.0001, clipnorm=1.0)
    episodes = 100000
    max_distance = np.sqrt(600**2 + 600**2)
    
    # Training Loop
    for episode in range(episodes) :
        observation, info = env.reset()
        done = False
        
        loss_black = 0.0
        loss_white = 0.0
        
        black_records = {"states" : [], "masks" : [], "actions" : [], "rewards" : []}
        white_records = {"states" : [], "masks" : [], "actions" : [], "rewards" : []}
        old_black_count = stone_count(observation, "black")
        old_white_count = stone_count(observation, "white")
        new_black_count = old_black_count
        new_white_count = old_white_count
        
        step_count = 0
        time_over = False 
        while not done :
            step_count += 1
            if(step_count >= 300) :
                time_over = True
                break
                
            turn = observation["turn"]
            aiming_reward = 0.0
            
            if(turn == 0) :
                action = black_agent.act(observation, info)
                input_state, input_mask = observation_to_input(observation, turn)
                black_records["states"].append(input_state)
                black_records["masks"].append(input_mask)
                norm_power = (action["power"] - 1.0) / 2500.0  # Normalize back to (0.0 ~ 1.0)
                norm_power = max(0.0, norm_power)
                norm_angle = action["angle"] / 180.0  # Normalize back to (-1.0 ~ 1.0)
                raw_action = [action["index"], norm_power, norm_angle]
                black_records["actions"].append(raw_action)
                black_records["rewards"].append(0)  # Placeholder for reward
                
            else :
                action = white_agent.act(observation, info)
                input_state, input_mask = observation_to_input(observation, turn)
                white_records["states"].append(input_state)
                white_records["masks"].append(input_mask)
                norm_power = (action["power"] - 1.0) / 2500.0  # Normalize back to (0.0 ~ 1.0)
                norm_power = max(0.0, norm_power)
                norm_angle = action["angle"] / 180.0  # Normalize back to (-1.0 ~ 1.0)
                raw_action = [action["index"], norm_power, norm_angle]
                white_records["actions"].append(raw_action)
                white_records["rewards"].append(0)  # Placeholder for reward (must be +=, not =)
            
            # Reward by Aiming
            aiming_reward = get_aiming_reward(observation, turn, action["index"], action["angle"])
            if(turn == 0) : black_records["rewards"][-1] += aiming_reward
            else : white_records["rewards"][-1] += aiming_reward
            
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            observation = next_observation
            
            # Reward by Stone Capture
            old_black_count = new_black_count
            old_white_count = new_white_count
            new_black_count = stone_count(observation, "black")
            new_white_count = stone_count(observation, "white")
            capture_reward_black = (old_black_count - new_black_count)
            capture_reward_white = (old_white_count - new_white_count)
            if(turn == 0) :
                black_records["rewards"][-1] -= capture_reward_black * 5.0
                black_records["rewards"][-1] += capture_reward_white * 5.0
            else :
                white_records["rewards"][-1] -= capture_reward_white * 5.0
                white_records["rewards"][-1] += capture_reward_black * 5.0
        
        if(time_over) : winner = "draw"
        else : winner = who_is_the_winner(observation)
        if(winner == "black") :
            black_reward = 10.0
            white_reward = -10.0
        elif(winner == "white") :
            black_reward = -10.0
            white_reward = 10.0
        else :
            black_reward = -1.0
            white_reward = -1.0
            
        if(len(black_records["rewards"])) : black_records["rewards"][-1] += black_reward
        if(len(white_records["rewards"])) : white_records["rewards"][-1] += white_reward

        if(len(black_records["states"]) > 0) :
            black_returns = calculate_returns(black_records["rewards"], gamma=0.99)
            loss_black = train_by_records(black_agent, np.array(black_records["states"]), np.array(black_records["masks"]), np.array(black_records["actions"]), black_returns, black_optimizer)
        if(len(white_records["states"]) > 0) :
            white_returns = calculate_returns(white_records["rewards"], gamma=0.99)
            loss_white = train_by_records(white_agent, np.array(white_records["states"]), np.array(white_records["masks"]), np.array(white_records["actions"]), white_returns, white_optimizer)
            
        loss_black_val = loss_black.numpy() if isinstance(loss_black, tf.Tensor) else loss_black
        loss_white_val = loss_white.numpy() if isinstance(loss_white, tf.Tensor) else loss_white
        
        print(f"Episode {episode + 1}/{episodes} completed | Winner: {winner}. | Black Loss: {loss_black_val:10.4f} | White Loss: {loss_white_val:10.4f} | Steps: {step_count:10d}", end="\r")
        
        if((episode + 1) % 10000 == 0) :  # temporary save
            black_agent.save(f"./moka_black_v7_{episode + 1}.keras")
            white_agent.save(f"./moka_white_v7_{episode + 1}.keras")
    
    black_agent.save("./moka_black_v7.keras")
    white_agent.save("./moka_white_v7.keras")
    env.close()

def test() :
    env = gym.make(
        id = "kymnasium/AlKkaGi-3x3-v0",
        render_mode = "human",
        bgm = True,
        obs_type = "custom"
    )
    black_agent = BlackAgent.load("./moka_black_v7_10000.keras")
    white_agent = WhiteAgent.load("./moka_white_v7_10000.keras")
    for _ in range(10) :    
        observation, info = env.reset()
        done = False
        while not done :
            turn = observation["turn"]
            if(turn == 0) : action = black_agent.act(observation, info, deterministic=True)
            else : action = white_agent.act(observation, info, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
        print(f"{_ + 1}/10 games played.")
    
    env.close()
# ---------- End of Training & Testing ----------
    
if __name__ == "__main__" :
    # kym.alkkagi.ManualPlayWrapper("kymnasium/AlKkaGi-3x3-v0", debug=True).play()
    # train()
    test()