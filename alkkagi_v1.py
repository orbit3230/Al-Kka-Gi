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
    
    player_stones = np.array(observation[player], dtype=np.float32).flatten()  # (9, )
    opponent_stones = np.array(observation[opponent], dtype=np.float32).flatten()  # (9, )
    
    # Normalize
    player_stones[0::3] /= WIDTH
    player_stones[1::3] /= HEIGHT
    opponent_stones[0::3] /= WIDTH
    opponent_stones[1::3] /= HEIGHT
    
    return np.concatenate([player_stones, opponent_stones])  # (18, )

def build_actor_critic_model(input_shape) :
    inputs = keras.layers.Input(shape=input_shape)
    
    common = keras.layers.Dense(128, activation="relu")(inputs)
    common = keras.layers.Dense(128, activation="relu")(common)
    
    # --- actor ---
    # 1. Stone Selection
    selection = keras.layers.Dense(3, activation="softmax", name="selection_out")(common)
    
    # 2. Power
    power_mean = keras.layers.Dense(1, activation="sigmoid")(common)  # (0.0 ~ 1.0)
    # power_mean = keras.layers.Lambda(lambda x : x * 2500.0 + 1.0)(power_mean)  # (1.0 ~ 2501.0) -> clipped later
    power_std_dev = keras.layers.Dense(1, activation="softplus")(common)  # must be positive
    power_std_dev = keras.layers.Lambda(lambda x : x + 0.05)(power_std_dev)  # avoid zero stddev
    
    # 3. Angle
    angle_mean = keras.layers.Dense(1, activation="tanh")(common)  # (-1.0 ~ 1.0)
    # angle_mean = keras.layers.Lambda(lambda x : x * 180.0)(angle_mean)  # (-180.0 ~ 180.0)
    angle_std_dev = keras.layers.Dense(1, activation="softplus")(common)  # must be positive
    angle_std_dev = keras.layers.Lambda(lambda x : x + 0.05)(angle_std_dev)  # avoid zero stddev
    # ----------
    
    # --- critic ---
    value = keras.layers.Dense(1, name="value_out")(common)
    model = keras.Model(
        inputs = inputs,
        outputs = [selection, power_mean, power_std_dev, angle_mean, angle_std_dev, value]
    )
    # ----------
    return model

def who_is_the_winner(observation) :
    black_survived = sum(1 for stone in observation["black"] if stone[2] == 1)
    white_survived = sum(1 for stone in observation["white"] if stone[2] == 1)
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

def train_by_records(agent, states, actions, returns, optimizer) :
    action_indices = tf.cast(actions[:, 0], tf.int32)
    action_powers = tf.cast(actions[:, 1], tf.float32)
    action_angles = tf.cast(actions[:, 2], tf.float32)
    
    states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
    returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)
    
    with tf.GradientTape() as tape :
        # Model Inference
        selection_probs, power_means, power_std_devs, angle_means, angle_std_devs, values = agent.model(states_tensor)
        values = tf.squeeze(values)
        
        # Calculate Advantage (G_t - V(s))
        advantages = returns_tensor - values
        
        # Actor Loss
        # 1. Stone Selection (Categorical)
        distribution_selection = tfp.distributions.Categorical(probs=selection_probs)
        log_probability_selection = distribution_selection.log_prob(action_indices)
        
        # 2. Power (Normal)
        distribution_power = tfp.distributions.Normal(loc=power_means, scale=power_std_devs)
        log_probability_power = distribution_power.log_prob(action_powers)
        
        # 3. Angle (Normal)
        distribution_angle = tfp.distributions.Normal(loc=angle_means, scale=angle_std_devs)
        log_probability_angle = distribution_angle.log_prob(action_angles)
        
        # Total Log Probability
        total_log_probability = log_probability_selection + tf.squeeze(log_probability_power) + tf.squeeze(log_probability_angle)
        
        # Policy Loss = - (log_prob * advantage)
        actor_loss = -tf.reduce_mean(total_log_probability * advantages)
        
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
# -------- End of Helper Functions --------

# ---------- Agent Class ----------
class Agent(kym.Agent) :
    def __init__(self, model = None, path: str = None) :
        if(path) :
            self.model = keras.models.load_model(path, safe_mode=False)
        elif(model) :
            self.model = model
        else :
            self.model = build_actor_critic_model((18,))

    def save(self, path: str) :
        self.model.save(path)

    @classmethod
    def load(cls, path: str) -> "kym.Agent" :
        return cls(path=path)

    def act(self, observation: Any, info: Dict) -> Any :
        turn = observation["turn"]
        state = observation_to_input(observation, turn)
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        
        # Model Inference
        selection_probabilities, power_mean, power_std_dev, angle_mean, angle_std_dev, value = self.model(state_tensor)
        
        # Stone Selection
        selected_index = np.random.choice(3, p=selection_probabilities.numpy()[0])
        # Power Sampling (Gaussian)
        raw_power = np.random.normal(power_mean.numpy()[0][0], power_std_dev.numpy()[0][0])
        raw_power = np.clip(raw_power, 0.0, 1.0)
        # Angle Sampling (Gaussian)
        raw_angle = np.random.normal(angle_mean.numpy()[0][0], angle_std_dev.numpy()[0][0])
        raw_angle = np.clip(raw_angle, -1.0, 1.0)
        
        # Rescale
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
    # TODO
    pass

class WhiteAgent(Agent) :
    # TODO
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
    episodes = 10000
    
    # Training Loop
    for episode in range(episodes) :
        observation, info = env.reset()
        done = False
        
        loss_black = 0.0
        loss_white = 0.0
        
        black_records = {"states" : [], "actions" : [], "rewards" : []}
        white_records = {"states" : [], "actions" : [], "rewards" : []}
        
        step_count = 0
        
        while not done :
            step_count += 1
            turn = observation["turn"]
            if(turn == 0) :
                action = black_agent.act(observation, info)
                input_state = observation_to_input(observation, turn)
                black_records["states"].append(input_state)
                norm_power = (action["power"] - 1.0) / 2500.0  # Normalize back to (0.0 ~ 1.0)
                norm_power = max(0.0, norm_power)
                norm_angle = action["angle"] / 180.0  # Normalize back to (-1.0 ~ 1.0)
                raw_action = [action["index"], norm_power, norm_angle]
                black_records["actions"].append(raw_action)
                black_records["rewards"].append(0)  # Placeholder for reward
                
            else :
                action = white_agent.act(observation, info)
                input_state = observation_to_input(observation, turn)
                white_records["states"].append(input_state)
                norm_power = (action["power"] - 1.0) / 2500.0  # Normalize back to (0.0 ~ 1.0)
                norm_power = max(0.0, norm_power)
                norm_angle = action["angle"] / 180.0  # Normalize back to (-1.0 ~ 1.0)
                raw_action = [action["index"], norm_power, norm_angle]
                white_records["actions"].append(raw_action)
                white_records["rewards"].append(0)  # Placeholder for reward
            
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            observation = next_observation
            
        winner = who_is_the_winner(observation)
        if(winner == "black") :
            black_reward = 1.0
            white_reward = -1.0
        elif(winner == "white") :
            black_reward = -1.0
            white_reward = 1.0
        else :
            black_reward = -0.1
            white_reward = -0.1
            
        if(len(black_records["rewards"])) : black_records["rewards"][-1] = black_reward
        if(len(white_records["rewards"])) : white_records["rewards"][-1] = white_reward

        if(len(black_records["states"]) > 0) :
            black_returns = calculate_returns(black_records["rewards"], gamma=0.99)
            loss_black = train_by_records(black_agent, np.array(black_records["states"]), np.array(black_records["actions"]), black_returns, black_optimizer)
        if(len(white_records["states"]) > 0) :
            white_returns = calculate_returns(white_records["rewards"], gamma=0.99)
            loss_white = train_by_records(white_agent, np.array(white_records["states"]), np.array(white_records["actions"]), white_returns, white_optimizer)
            
        loss_black_val = loss_black.numpy() if isinstance(loss_black, tf.Tensor) else loss_black
        loss_white_val = loss_white.numpy() if isinstance(loss_white, tf.Tensor) else loss_white
        
        print(f"Episode {episode + 1}/{episodes} completed | Winner: {winner}. | Black Loss: {loss_black_val:10.4f} | White Loss: {loss_white_val:10.4f} | Steps: {step_count:10d}", end="\r")
    
    black_agent.save("./moka_black_v1.keras")
    white_agent.save("./moka_white_v1.keras")
    env.close()

def test() :
    env = gym.make(
        id = "kymnasium/AlKkaGi-3x3-v0",
        render_mode = "human",
        bgm = True,
        obs_type = "custom"
    )
    black_agent = BlackAgent.load("./moka_black_v1.keras")
    white_agent = WhiteAgent.load("./moka_white_v1.keras")
    for _ in range(10) :    
        observation, info = env.reset()
        done = False
        while not done :
            turn = observation["turn"]
            if(turn == 0) : action = black_agent.act(observation, info)
            else : action = white_agent.act(observation, info)
            observation, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
        print(f"{_ + 1}/10 games played.", end="\r")
    
    env.close()
# ---------- End of Training & Testing ----------
    
if __name__ == "__main__" :
    # kym.alkkagi.ManualPlayWrapper("kymnasium/AlKkaGi-3x3-v0", debug=True).play()
    # train()
    test()