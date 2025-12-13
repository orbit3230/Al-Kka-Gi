import kymnasium as kym
import gymnasium as gym
import numpy as np
from typing import Dict, Any
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import csv
import os

# Method only for Manual Play
# kymnasium.alkkagi.ManualPlayWrapper("kymnasium/AlKkaGi-3x3-v0", debug=True).play()

# v11 Patch Notes
# 1. Collision detection -> Reward giving
# 2. Bigger Entropy bonus (encourage exploration)
# 3. Seperate Actor and Critic networks
# 4. Huber loss for critic
# 5. Cross-Training
# 6. Brought the sorting of opponent stones back (by y-coordinate)
# 7. Dead stone processing after normalization
# 8. Reward tuning
# ---
# v12 Patch Notes
# 1. Reward Minimization Version.
# ---
# v13 Patch Notes
# 1. Added minimum power limit (500.0) to avoid zero power shots (to encourage active play).
# ---
# v14 Patch Notes
# 1. Coordinate Reflection
# 2. Norm power redesigned when storing action records
# 3. Shared Model between Black and White Agents
# 4. Added Agent Load & Train
# ---
# v15 Patch Notes
# 1. Restored Collision Detection Reward
# 2. Restored Stone Capture Reward
# 3. Restored Self Dying Penalty
# 4. Removed Coordinate Reflection (v14)
# 5. Seperated Black and White Agent Load (v14)
# ---
# v16 Patch Notes
# 1. Step Penalty Adjustment
# 2. Collision Reward Adjustment
# 3. Win/Lose/Draw Penalty Adjustment
# 4. Added Distance Reward
# 5. Layer 256 -> 512
# ---
# v17 Patch Notes
# 1. (CANCELED) Do NOT Use Deterministic Policy When Training
# 2. Black & White Agent simultaneous learning during first 10000 episodes
# 3. Added Survival Bonus
# 4. Removed Step Penalty
# 5. Removed Distance Reward
# 6. Restored Aiming Reward
# ---
# v18 Patch Notes
# 1. Restored Step Penalty 
# 2. Removed all Reward Mechanics except win/lose & step penalty
# 3. Lower Learning Rate
# ---
# v19 Patch Notes
# 1. (CANCELLED) No Cross-Training. Always train both agents. (Test)
# 2. Added Small Reward for Collision & Kill
# ---
# v20 Patch Notes
# 1. Added Standard Deviation Upper Bound
# 2. Added Return Clipping
# 3. Added Advantage Clipping
# 4. Added Advantage Normalization
# 5. Added Entropy Clipping
# ---
# v21 Patch Notes -> Canceled
# ---
# v22 Patch Notes
# 1. Using "range limited traslation" instead of "cliplayer"
# 2. Enabled self-dying penalty
# 3. Ease the Return/Advantage clipping
# 4. Simultaneous self-play (TEST)
# ---
# v23 Patch Notes
# 1. stddev minimum increased
# 2. Restored Aiming Reward
# ---
# v24 Patch Notes
# 1. Lower Aiming Reward
# 2. Big update in observation_to_input() function
# ---------- Helper Functions ----------
def point_in_obstacle(x, y, rx1, ry1, rx2, ry2) :
    return((min(rx1, rx2) <= x <= max(rx1, rx2)) and (min(ry1, ry2) <= y <= max(ry1, ry2)))

def ccw(ax, ay, bx, by, cx, cy) :
    return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)

def segments_intersect(ax, ay, bx, by, cx, cy, dx, dy) :
    return (ccw(ax, ay, cx, cy, dx, dy) != ccw(bx, by, cx, cy, dx, dy)) and \
              (ccw(ax, ay, bx, by, cx, cy) != ccw(ax, ay, bx, by, dx, dy)) 
              
def segment_blocked_by_obstacle(x1, y1, x2, y2, obstacles) :
    if(obstacles.size == 0) : return False
    for(ox1, oy1, ox2, oy2) in obstacles :
        xmin = min(ox1, ox2)
        xmax = max(ox1, ox2)
        ymin = min(oy1, oy2)
        ymax = max(oy1, oy2)
        if(point_in_obstacle(x1, y1, ox1, oy1, ox2, oy2) or point_in_obstacle(x2, y2, ox1, oy1, ox2, oy2)) :
            return True
        edges = [
            (xmin, ymin, xmax, ymin),
            (xmax, ymin, xmax, ymax),
            (xmax, ymax, xmin, ymax),
            (xmin, ymax, xmin, ymin)
        ]
        for(ex1, ey1, ex2, ey2) in edges :
            if(segments_intersect(x1, y1, x2, y2, ex1, ey1, ex2, ey2)) :
                return True
    return False

def observation_to_input(observation, turn) :
    WIDTH = 600
    HEIGHT = 600
    DIAGONAL = (WIDTH ** 2 + HEIGHT ** 2) ** 0.5
    
    if(turn == 0) :
        player = "black"
        opponent = "white"
    else :
        player = "white"
        opponent = "black"
        
    player_stones = np.array(observation[player], dtype=np.float32)
    opponent_stones = np.array(observation[opponent], dtype=np.float32)
    obstacles = np.array(observation["obstacles"], dtype=np.float32)
    # Reshape obstacle information into [x1, y1, x2, y2]
    if(obstacles.size == 0) : obstacles = obstacles.reshape(0, 4)
    else : obstacles = obstacles.reshape(-1, 4)
    
    features = []
    
    for i in range(3) :
        x, y, alive = player_stones[i]
        for j in range(3) :
            ox, oy, oalive = opponent_stones[j]
            if(alive == 0 or oalive == 0) :
                angle_norm = 0.0
                distance_norm = 1.0
                blocked = 0.0
                features.extend([angle_norm, distance_norm, blocked])
                continue
            dx = float(ox - x)
            dy = float(oy - y)
            angle = np.arctan2(dy, dx)  # -pi ~ +pi
            angle_norm = angle / np.pi  # -1.0 ~ +1.0
            distance = float(np.sqrt(dx * dx + dy * dy))
            distance_norm = float(distance / DIAGONAL)  # 0.0 ~ ~1.0
            blocked = 1.0 if segment_blocked_by_obstacle(x, y, ox, oy, obstacles) else 0.0
            features.extend([angle_norm, distance_norm, blocked])
    
    # 3 x 3 x 3 = 27 features from stone interactions
    features = np.array(features, dtype=np.float32)  # (27, )
    turn_indicator = np.array([float(turn)], dtype=np.float32)  # (1, )
    valid_mask = player_stones[:, 2].astype(np.float32)  # (3, ) e.g., [1., 0., 1.] -> alive, dead, alive
    return np.concatenate([features, turn_indicator]), valid_mask  # (28, ), (3, )
    
# def observation_to_input(observation, turn) :
#     WIDTH = 600
#     HEIGHT = 600
    
#     if(turn == 0) :
#         player = "black"
#         opponent = "white"
#     else :
#         player = "white"
#         opponent = "black"
    
#     player_stones = np.array(observation[player], dtype=np.float32)
#     player_stones = player_stones.flatten()  # (9, )
    
#     # Sort opponent stones by y-coordinate
#     opponent_stones = np.array(observation[opponent], dtype=np.float32)
#     opponent_stones = sorted(opponent_stones, key=lambda x: (-x[2], x[1])) # alive first, then by y-coordinate
#     opponent_stones = np.array(opponent_stones, dtype=np.float32).flatten()  # (9, )
    
#     obstacles = np.array(observation["obstacles"], dtype=np.float32).flatten()  # (12, )
    
#     # Normalize
#     player_stones[0::3] /= WIDTH
#     player_stones[1::3] /= HEIGHT
#     opponent_stones[0::3] /= WIDTH
#     opponent_stones[1::3] /= HEIGHT
#     obstacles[0::4] /= WIDTH
#     obstacles[1::4] /= HEIGHT
#     obstacles[2::4] /= WIDTH
#     obstacles[3::4] /= HEIGHT
    
#     # if(turn == 1) :
#     #     for i in range(len(player_stones) // 3) :
#     #         if(player_stones[i*3+2] == 1) :  # alive stone
#     #             player_stones[i*3+0] = 1.0 - player_stones[i*3+0]  # x
#     #     for i in range(len(opponent_stones) // 3) :
#     #         if(opponent_stones[i*3+2] == 1) :  # alive stone
#     #             opponent_stones[i*3+0] = 1.0 - opponent_stones[i*3+0]  # x
    
#     # Dead stone processing after normalization
#     for i in range(len(player_stones) // 3) :
#         if(player_stones[i*3+2] == 0) :  # dead stone
#             player_stones[i*3+0] = -1.0  # x
#             player_stones[i*3+1] = -1.0  # y
#     for i in range(len(opponent_stones) // 3) :
#         if(opponent_stones[i*3+2] == 0) :  # dead stone
#             opponent_stones[i*3+0] = -1.0  # x
#             opponent_stones[i*3+1] = -1.0  # y
    
#     # Dead stone masking
#     valid_mask = player_stones[2::3]  # (3, ) e.g., [1., 0., 1.] -> alive, dead, alive
    
#     turn_indicator = np.array([float(turn)], dtype=np.float32)  # (1, )
    
#     return np.concatenate([player_stones, opponent_stones, obstacles, turn_indicator]), valid_mask  # (31, ), (3, )

class ClipLayer(keras.layers.Layer) :
    def __init__(self, min_value, max_value, **kwargs) :
        super(ClipLayer, self).__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value
    def call(self, inputs) :
        return tf.clip_by_value(inputs+0.001, self.min_value, self.max_value)
    def get_config(self) :
        config = super(ClipLayer, self).get_config()
        config.update({
            "min_value" : self.min_value,
            "max_value" : self.max_value
        })
        return config

def build_actor_critic_model(input_shape) :
    inputs = keras.layers.Input(shape=input_shape, name="state_input")
    mask_input = keras.layers.Input(shape=(3,), name="mask_input")
    
    actor_common = keras.layers.Dense(512, activation="relu", name="actor_dense_1")(inputs)
    actor_common = keras.layers.Dense(512, activation="relu", name="actor_dense_2")(actor_common)
    
    # --- actor ---
    # 1. Stone Selection - masking applied
    # selection = keras.layers.Dense(3, activation="softmax", name="selection_out")(common)
    selection_logits = keras.layers.Dense(3, name="selection_logits")(actor_common)  # logits means "log-odds"
    # By adding a large negative number before softmax, we can effectively mask out invalid choices
    adder = keras.layers.Lambda(lambda x : (1.0 - x) * -1e9, output_shape=(3,), name="mask_adder")(mask_input)
    masked_selection_logits = keras.layers.Add(name="masked_selection_logits")([selection_logits, adder])
    # selection_probabilities = keras.layers.Activation("softmax", name="selection_probabilities")(masked_selection_logits)
    # -> NaN issue
    
    # 2. Power
    power_mean = keras.layers.Dense(3, activation="sigmoid", name="power_mean")(actor_common)  # (0.0 ~ 1.0)
    # power_mean = keras.layers.Lambda(lambda x : x * 2500.0 + 1.0)(power_mean)  # (1.0 ~ 2501.0) -> clipped later
    # power_std_dev_raw = keras.layers.Dense(3, activation="softplus", name="power_std_dev")(actor_common)  # must be positive
    # power_std_dev = keras.layers.Lambda(lambda x : x + 0.001)(power_std_dev)  # avoid zero stddev
    # power_std_dev = keras.layers.Lambda(lambda x : tf.clip_by_value(x+0.001, 0.001, 0.2), output_shape=(3,))(power_std_dev_raw)  # upper bound
    # power_std_dev = ClipLayer(0.001, 0.2, name="power_std_dev_clip")(power_std_dev_raw)
    power_std_dev_base = keras.layers.Dense(3, activation="sigmoid", name="power_std_dev_base")(actor_common)
    power_std_dev = 0.02 + 0.08 * power_std_dev_base
    
    # 3. Angle
    angle_mean = keras.layers.Dense(3, activation="tanh", name="angle_mean")(actor_common)  # (-1.0 ~ 1.0)
    # angle_mean = keras.layers.Lambda(lambda x : x * 180.0)(angle_mean)  # (-180.0 ~ 180.0)
    # angle_std_dev_raw = keras.layers.Dense(3, activation="softplus", name="angle_std_dev")(actor_common)  # must be positive
    # angle_std_dev = keras.layers.Lambda(lambda x : x + 0.001)(angle_std_dev)  # avoid zero stddev
    # angle_std_dev = keras.layers.Lambda(lambda x : tf.clip_by_value(x+0.001, 0.001, 0.05), output_shape=(3,))(angle_std_dev_raw)  # upper bound
    # angle_std_dev = ClipLayer(0.001, 0.05, name="angle_std_dev_clip")(angle_std_dev_raw)
    angle_std_dev_base = keras.layers.Dense(3, activation="sigmoid", name="angle_std_dev_base")(actor_common)
    angle_std_dev = 0.02 + 0.02 * angle_std_dev_base
    
    actor_model = keras.Model(
        inputs = [inputs, mask_input],
        outputs = [masked_selection_logits, power_mean, power_std_dev, angle_mean, angle_std_dev]
    )
    # ----------
    
    # --- critic ---
    critic_common = keras.layers.Dense(512, activation="relu", name="critic_dense_1")(inputs)
    critic_common = keras.layers.Dense(512, activation="relu", name="critic_dense_2")(critic_common)
    
    value = keras.layers.Dense(1, name="value_out")(critic_common)
    critic_model = keras.Model(
        inputs = inputs,
        outputs = value
    )
    # ----------
    return actor_model, critic_model

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

def train_by_records(agent, states, masks, actions, returns, actor_optimizer, critic_optimizer) :
    action_indices = tf.cast(actions[:, 0], tf.int32)
    action_powers = tf.cast(actions[:, 1], tf.float32)
    action_angles = tf.cast(actions[:, 2], tf.float32)
    
    states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
    mask_tensor = tf.convert_to_tensor(masks, dtype=tf.float32)
    returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)
    returns_tensor = tf.reshape(returns_tensor, (-1, ))  # (batch_size, )
    # Clipping returns
    returns_tensor = tf.clip_by_value(returns_tensor, -10.0, 10.0)
    
    # 1. Critic Update
    with tf.GradientTape() as tape_critic :
        values_raw = agent.critic(states_tensor)
        values = tf.squeeze(values_raw, axis=1)  # (batch_size, )
        huber_loss = tf.keras.losses.Huber()
        critic_loss = huber_loss(returns_tensor, values)
    critic_gradients = tape_critic.gradient(critic_loss, agent.critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(critic_gradients, agent.critic.trainable_variables))
    
    # advantages = returns_tensor - values
    # advantage_mean = tf.math.reduce_mean(advantages)
    # advantage_std = tf.math.reduce_std(advantages) + 1e-8
    # normalized_advantages = (advantages - advantage_mean) / advantage_std
    values_detached = tf.stop_gradient(values)
    
    # 2. Actor Update
    with tf.GradientTape() as tape :
        # Model Inference
        selection_logits, power_means, power_std_devs, angle_means, angle_std_devs = agent.actor([states_tensor, mask_tensor])
        # values = tf.squeeze(values)
        
        # Calculate Advantage (G_t - V(s))
        # advantages = returns_tensor - tf.stop_gradient(values)
        advantages = returns_tensor - values_detached
        # Advantage Normalization
        advantage_mean = tf.math.reduce_mean(advantages)
        advantage_std = tf.math.reduce_std(advantages) + 1e-8
        normalized_advantages = (advantages - advantage_mean) / advantage_std
        # Clipping Advantages
        # normalized_advantages = tf.clip_by_value(normalized_advantages, -2.0, 2.0)
        
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
        
        # 4. Entropy Bonus (to encourage exploration)
        entropy = tf.reduce_mean(distribution_selection.entropy() + distribution_power.entropy() + distribution_angle.entropy())
        # Clipping Entropy
        entropy = tf.clip_by_value(entropy, 0.0, 5.0)
        
        # Total Loss
        total_actor_loss = actor_loss - (0.01 * entropy)
        
    # Gradient Update
    gradients = tape.gradient(total_actor_loss, agent.actor.trainable_variables)
    actor_optimizer.apply_gradients(zip(gradients, agent.actor.trainable_variables))
    
    return total_actor_loss, critic_loss, entropy

def get_aiming_reward(observation, turn, selection, angle) :
    if(turn == 0) :
        players = observation["black"]
        opponents = observation["white"]
    else :
        players = observation["white"]
        opponents = observation["black"]
        
    player_stone = players[selection]
    if(player_stone[2] == 0) : return 0.0
    
    alive_opponents = [stone for stone in opponents if stone[2] == 1]
    if(not alive_opponents) : return 0.0  # this code should not be reached normally
    
    max_cosine_similarity = -1.0  # if aiming directly, cosine similarity is 1.0
    for opponent in alive_opponents :
        dx = opponent[0] - player_stone[0]
        dy = opponent[1] - player_stone[1]
        target_angle_degree = np.degrees(np.arctan2(dy, dx))
        cosine_similarity = np.cos(np.radians(target_angle_degree - angle))
        max_cosine_similarity = max(max_cosine_similarity, cosine_similarity)
        
    # -0.003 ~ +0.003
    return 0.003 * max_cosine_similarity

# return : {is_there_black_collision, is_there_white_collision}
def collision_detected(observation_before, observation_after) :
    black_x_before = [stone[0] for stone in observation_before["black"] if stone[2] == 1]
    black_y_before = [stone[1] for stone in observation_before["black"] if stone[2] == 1]
    white_x_before = [stone[0] for stone in observation_before["white"] if stone[2] == 1]
    white_y_before = [stone[1] for stone in observation_before["white"] if stone[2] == 1]
    black_x_after = [stone[0] for stone in observation_after["black"] if stone[2] == 1]
    black_y_after = [stone[1] for stone in observation_after["black"] if stone[2] == 1]
    white_x_after = [stone[0] for stone in observation_after["white"] if stone[2] == 1]
    white_y_after = [stone[1] for stone in observation_after["white"] if stone[2] == 1]
    
    is_there_black_collision = False
    is_there_white_collision = False
    for x_b_before, y_b_before, x_b_after, y_b_after in zip(black_x_before, black_y_before, black_x_after, black_y_after) :
        if(abs(x_b_before - x_b_after) > 1.0 or abs(y_b_before - y_b_after) > 1.0) :
            is_there_black_collision = True
            break
    for x_w_before, y_w_before, x_w_after, y_w_after in zip(white_x_before, white_y_before, white_x_after, white_y_after) :
        if(abs(x_w_before - x_w_after) > 1.0 or abs(y_w_before - y_w_after) > 1.0) :
            is_there_white_collision = True
            break
    return is_there_black_collision, is_there_white_collision

# def nearest_distance(observation, turn, selection) :
#     player = "black" if turn == 0 else "white"
#     opponent = "white" if turn == 0 else "black"
    
#     player_stone = observation[player][selection]
#     if(player_stone[2] == 0) : return 0.0  # dead stone
#     opponent_stones = [stone for stone in observation[opponent] if stone[2] == 1]
#     if(not opponent_stones) : return 0.0  # no alive opponent stones
#     distance = [np.hypot(player_stone[0] - stone[0], player_stone[1] - stone[1]) for stone in opponent_stones]
#     return float(min(distance))
# -------- End of Helper Functions --------

# ---------- Agent Class ----------
class Agent(kym.Agent) :
    def __init__(self, model = None, path: str = None) :
        if(path) :
            custom_objects = {"ClipLayer": ClipLayer}
            self.actor = keras.models.load_model(path + "_actor.keras", custom_objects=custom_objects, safe_mode=False)
            self.critic = keras.models.load_model(path + "_critic.keras", custom_objects=custom_objects, safe_mode=False)
        elif(model) :
            self.actor, self.critic = model
        else :
            self.actor, self.critic = build_actor_critic_model((28, ))  # input shape (28, )

    def save(self, path: str) :
        self.actor.save(path + "_actor.keras")
        self.critic.save(path + "_critic.keras")

    @classmethod
    def load(cls, path: str) -> "kym.Agent" :
        custom_objects = {"ClipLayer": ClipLayer}
        actor = keras.models.load_model(path + "_actor.keras", custom_objects=custom_objects,safe_mode=False)
        critic = keras.models.load_model(path + "_critic.keras", safe_mode=False)
        model = (actor, critic)
        return cls(model=model)

    def act(self, observation: Any, info: Dict, deterministic: bool = True) -> Any :
        turn = observation["turn"]
        state, valid_mask = observation_to_input(observation, turn)
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        mask_tensor = tf.convert_to_tensor([valid_mask], dtype=tf.float32)
        
        # Model Inference
        selection_logits, power_mean, power_std_dev, angle_mean, angle_std_dev = self.actor([state_tensor, mask_tensor])
        
        # debug
        # print("turn : ", 'black' if turn == 0 else 'white')
        # print("selection_logits : ", selection_logits.numpy()[0])
        # print("power_mean : ", power_mean.numpy()[0])
        # print("power_std_dev : ", power_std_dev.numpy()[0])
        # print("angle_mean : ", angle_mean.numpy()[0])
        # print("angle_std_dev : ", angle_std_dev.numpy()[0])
        # print("---------------------------")
        
        # Stone Selection
        if(deterministic) : selected_index = np.argmax(selection_logits.numpy()[0])
        else :
            # selected_probabilities = selection_probabilities.numpy()[0] -> NaN issue
            # selected_probabilities = tf.nn.softmax(selection_logits).numpy()[0]
            # selected_index = np.random.choice(3, p=selected_probabilities)
            probabilities = tf.nn.softmax(selection_logits).numpy()[0]
            if((not np.all(np.isfinite(probabilities)) or (probabilities.sum() <= 0.0))) :
                mask = valid_mask
                if(np.any(mask > 0.5)) :
                    alive_indices = np.where(mask > 0.5)[0]
                    logits = selection_logits.numpy()[0]
                    selected_index = int(alive_indices[np.argmax(logits[alive_indices])])
                else :
                    selected_index = int(np.argmax(selection_logits.numpy()[0]))
            else :
                selected_index = np.random.choice(3, p=probabilities)
        
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
        
        # Rescale
        # v13 added
        min_power = 500.0
        max_power = 2500.0
        power = raw_power * (max_power - min_power) + min_power
        power = max(min_power, power)
        angle = raw_angle * 180.0
        
        # if(turn == 1) :
        #     angle = 180.0 - angle
        #     if(angle > 180.0) : angle -= 360.0
        #     if(angle < -180.0) : angle += 360.0
        
        return {
            "turn" : turn,
            "index" : int(selected_index),
            "power" : float(power),
            "angle" : float(angle)
        }
    
class BlackAgent(Agent) :
    def act(self, observation: Any, info: Dict, deterministic: bool = True) :
        if(observation["turn"] != 0) : return None
        return super().act(observation, info, deterministic)

class WhiteAgent(Agent) :
    def act(self, observation: Any, info: Dict, deterministic: bool = True) :
        if(observation["turn"] != 1) : return None
        return super().act(observation, info, deterministic)
# -------- End of Agent Class --------

# ---------- Training & Testing ----------
def train(resume_path_black = None, resume_path_white = None, start_episode = 0) :
    # Environment
    env = gym.make(
        id = "kymnasium/AlKkaGi-3x3-v0",
        render_mode = "rgb_array",
        bgm = False,
        obs_type = "custom"
    )
    if(resume_path_black and resume_path_white) :
        print(f"Resuming training from {resume_path_black} and {resume_path_white} ...")
        custom_objects = {"ClipLayer": ClipLayer}
        black_actor_model = keras.models.load_model(resume_path_black + "_actor.keras", custom_objects=custom_objects, safe_mode=False)
        black_critic_model = keras.models.load_model(resume_path_black + "_critic.keras", custom_objects=custom_objects, safe_mode=False)
        white_actor_model = keras.models.load_model(resume_path_white + "_actor.keras", custom_objects=custom_objects, safe_mode=False)
        white_critic_model = keras.models.load_model(resume_path_white + "_critic.keras", custom_objects=custom_objects, safe_mode=False)
    else :
        print("Starting training from scratch ...")
        black_actor_model, black_critic_model = build_actor_critic_model((28, ))
        white_actor_model, white_critic_model = build_actor_critic_model((28, ))
    # shared_model = (actor_model, critic_model)
    # black_agent = BlackAgent(model=shared_model)
    # white_agent = WhiteAgent(model=shared_model)
    black_agent = BlackAgent(model=(black_actor_model, black_critic_model))
    white_agent = WhiteAgent(model=(white_actor_model, white_critic_model))
    
    black_actor_optimizer = keras.optimizers.Adam(learning_rate = 0.00003, clipnorm=1.0)
    black_critic_optimizer = keras.optimizers.Adam(learning_rate = 0.0001, clipnorm=1.0)
    white_actor_optimizer = keras.optimizers.Adam(learning_rate = 0.00003, clipnorm=1.0)
    white_critic_optimizer = keras.optimizers.Adam(learning_rate = 0.0001, clipnorm=1.0)
    # actor_optimizer = keras.optimizers.Adam(learning_rate = 0.0001, clipnorm=1.0)
    # critic_optimizer = keras.optimizers.Adam(learning_rate = 0.001, clipnorm=1.0)
    
    # Cross-Training
    cross_frequency = 500
    simultaneous_learning_episodes = 10000
    episodes = 500000
    
    # Open CSV file for logging
    log_filename = "training_log_v24.csv"
    file_mode = 'a' if (resume_path_black and resume_path_white) and os.path.exists(log_filename) else 'w'
    with open(log_filename, mode=file_mode, newline='') as log_file:
        log_writer = csv.writer(log_file)
        if(file_mode == 'w') :
            log_writer.writerow([
                "Episode", "Steps", "Winner", "Active_Side",
                "Black_Reward", "White_Reward",
                "Actor_Loss_Black", "Critic_Loss_Black",
                "Actor_Loss_White", "Critic_Loss_White",
                "Entropy_Black", "Entropy_White",
                "Black_Self_Dying_Count", "White_Self_Dying_Count",
                "Black_Kill_Count", "White_Kill_Count",
                "Black_Hit_Count", "White_Hit_Count",
                "White_Correct_Aim_Count", "Black_Correct_Aim_Count"
            ])
    
    # Training Loop
    for episode in range(start_episode, start_episode + episodes) :
        observation, info = env.reset()
        done = False
        
        loss_black = 0.0
        loss_white = 0.0
        critic_loss_black = 0.0
        critic_loss_white = 0.0
        entropy_black = 0.0
        entropy_white = 0.0
        
        black_records = {"states" : [], "masks" : [], "actions" : [], "rewards" : []}
        white_records = {"states" : [], "masks" : [], "actions" : [], "rewards" : []}
        old_black_count = stone_count(observation, "black")
        old_white_count = stone_count(observation, "white")
        new_black_count = old_black_count
        new_white_count = old_white_count
        
        step_count = 0
        black_self_dying_count = 0
        white_self_dying_count = 0
        black_kill_count = 0
        white_kill_count = 0
        black_hit_count = 0
        white_hit_count = 0
        black_correct_aim_count = 0
        white_correct_aim_count = 0
        GAMMA = 0.99
        # Made Draw's current value same regardless of episode length
        step_penalty = 0.005
        # time_over = False 
        
        min_power = 500.0
        max_power = 2500.0
        
        # 0 : Black Agent Training Phase (White : Frozen)
        # 1 : White Agent Training Phase (Black : Frozen)
        
        # (CANCELED) v19 change: always train both agents (Test)
        # training_phase = (episode // cross_frequency) % 2
        # black_training_phase = (training_phase == 0)
        # white_training_phase = (training_phase == 1)
        black_training_phase = True
        white_training_phase = True
        # if(episode < simultaneous_learning_episodes) :
        #     black_training_phase = True
        #     white_training_phase = True
        
        while not done :
            step_count += 1
            if(step_count >= 100000) :  # I hope this won't happen...
                break
                
            turn = observation["turn"]
            # aiming_reward = 0.0
            
            if(turn == 0) :
                # when black doesn't train, use deterministic policy
                action = black_agent.act(observation, info, deterministic = not black_training_phase)
                input_state, input_mask = observation_to_input(observation, turn)
                black_records["states"].append(input_state)
                black_records["masks"].append(input_mask)
                # norm_power = (action["power"] - 1.0) / 2500.0  # Normalize back to (0.0 ~ 1.0)
                # norm_power = max(0.0, norm_power)
                # norm_angle = action["angle"] / 180.0  # Normalize back to (-1.0 ~ 1.0)
                norm_power = (action["power"] - min_power) / (max_power - min_power)
                norm_power = float(np.clip(norm_power, 0.0, 1.0))
                norm_angle = float(np.clip(action["angle"] / 180.0, -1.0, 1.0))  # Normalize back to (-1.0 ~ 1.0)
                raw_action = [action["index"], norm_power, norm_angle]
                black_records["actions"].append(raw_action)
                black_records["rewards"].append(0)  # Placeholder for reward
                
            else :
                # when white doesn't train, use deterministic policy
                action = white_agent.act(observation, info, deterministic = not white_training_phase)
                input_state, input_mask = observation_to_input(observation, turn)
                white_records["states"].append(input_state)
                white_records["masks"].append(input_mask)
                # norm_power = (action["power"] - 1.0) / 2500.0  # Normalize back to (0.0 ~ 1.0)
                # norm_power = max(0.0, norm_power)
                # norm_angle = action["angle"] / 180.0  # Normalize back to (-1.0 ~ 1.0)
                norm_power = (action["power"] - min_power) / (max_power - min_power)
                norm_power = float(np.clip(norm_power, 0.0, 1.0))
                norm_angle = float(np.clip(action["angle"] / 180.0, -1.0, 1.0))  # Normalize back to (-1.0 ~ 1.0)
                raw_action = [action["index"], norm_power, norm_angle]
                white_records["actions"].append(raw_action)
                white_records["rewards"].append(0)  # Placeholder for reward (must be +=, not =)
                
            # Step Penalty
            if(turn == 0) : black_records["rewards"][-1] -= step_penalty
            else : white_records["rewards"][-1] -= step_penalty
            
            # Reward by Aiming
            aiming_reward = get_aiming_reward(observation, turn, action["index"], action["angle"])
            if(turn == 0) :
                if(aiming_reward > 0.0025) : black_correct_aim_count += 1
                black_records["rewards"][-1] += aiming_reward
            else :
                if(aiming_reward > 0.0025) : white_correct_aim_count += 1
                white_records["rewards"][-1] += aiming_reward
            
            # distance_before = nearest_distance(observation, turn, action["index"])
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            is_there_black_collision, is_there_white_collision = collision_detected(observation, next_observation)
            observation = next_observation
            # distance_after = nearest_distance(observation, turn, action["index"])
            
            # engage_distance = 100.0
            # approach_scale = 0.05  # +- 0.05
            # knockback_scale = 2.0  # +- 2.0
            # Reward by Distance (Only when out of the engage distance)
            # if((distance_before > engage_distance) and (distance_after > engage_distance)) :
            #     delta_distance = distance_before - distance_after
            #     distance_reward = approach_scale * np.tanh(delta_distance / 50.0)
            #     if(turn == 0) : black_records["rewards"][-1] += distance_reward
            #     else : white_records["rewards"][-1] += distance_reward
                
            # Reward by Collision (Knockback)            
            if(turn == 0 and is_there_white_collision) :
                black_hit_count += 1
                black_records["rewards"][-1] += 0.1  # collision constant reward
            #     if(distance_before < distance_after) :  # opponent stone knocked back
            #         delta_distance = distance_after - distance_before
            #         knockback_reward = knockback_scale * np.tanh(delta_distance / 50.0)
            #         black_records["rewards"][-1] += knockback_reward * 5.0
            elif(turn == 1 and is_there_black_collision) :
                white_hit_count += 1
                white_records["rewards"][-1] += 0.1  # collision constant reward
            #     if(distance_before < distance_after) :  # opponent stone knocked back
            #         delta_distance = distance_after - distance_before
            #         knockback_reward = knockback_scale * np.tanh(delta_distance / 50.0)
            #         white_records["rewards"][-1] += knockback_reward * 5.0
                
            # Reward by Stone Capture
            old_black_count = new_black_count
            old_white_count = new_white_count
            new_black_count = stone_count(observation, "black")
            new_white_count = stone_count(observation, "white")
            capture_reward_black = (old_black_count - new_black_count)
            capture_reward_white = (old_white_count - new_white_count)
            if(turn == 0) :
                black_records["rewards"][-1] -= capture_reward_black * 0.5  # self dying penalty
                black_self_dying_count += capture_reward_black
                black_records["rewards"][-1] += capture_reward_white * 0.5  # opponent capture reward
                black_kill_count += capture_reward_white
            else :
                white_records["rewards"][-1] -= capture_reward_white * 0.5  # self dying penalty
                white_self_dying_count += capture_reward_white
                white_records["rewards"][-1] += capture_reward_black * 0.5  # opponent capture reward
                white_kill_count += capture_reward_black
                
            # Reward by Survival
            # survival_bonus = 0.1
            # if(turn == 0) :
            #     alive_stones = stone_count(observation, "black")
            #     black_records["rewards"][-1] += alive_stones * survival_bonus
            # else :
            #     alive_stones = stone_count(observation, "white")
            #     white_records["rewards"][-1] += alive_stones * survival_bonus
        
        # if(time_over) : winner = "draw"
        else : winner = who_is_the_winner(observation)
        black_survival_count = stone_count(observation, "black")
        white_survival_count = stone_count(observation, "white")
        if(winner == "black") :
            black_reward = 0.5 * black_survival_count
            white_reward = -0.5 * black_survival_count
        elif(winner == "white") :
            black_reward = -0.5 * white_survival_count
            white_reward = 0.5 * white_survival_count
        else :  # draw
            black_reward = -0.5
            white_reward = -0.5
            
        # if(len(black_records["rewards"])) : black_records["rewards"][-1] += black_reward
        # if(len(white_records["rewards"])) : white_records["rewards"][-1] += white_reward
        # only use when calculating returns
        if(len(black_records["rewards"])) : black_records["rewards"].append(black_reward)
        if(len(white_records["rewards"])) : white_records["rewards"].append(white_reward)

        if(black_training_phase) :
            if(len(black_records["states"]) > 0) :
                black_returns = calculate_returns(black_records["rewards"], gamma=GAMMA)
                if(len(black_returns) == len(black_records["states"])+1) : # remove the last reward used for bootstrapping
                    black_returns = black_returns[:-1]
                loss_black, critic_loss_black, entropy_black = train_by_records(black_agent, np.array(black_records["states"]), np.array(black_records["masks"]), np.array(black_records["actions"]), black_returns, black_actor_optimizer, black_critic_optimizer)
        if(white_training_phase) :
            if(len(white_records["states"]) > 0) :
                white_returns = calculate_returns(white_records["rewards"], gamma=GAMMA)
                if(len(white_returns) == len(white_records["states"])+1) : # remove the last reward used for bootstrapping
                    white_returns = white_returns[:-1]
                loss_white, critic_loss_white, entropy_white = train_by_records(white_agent, np.array(white_records["states"]), np.array(white_records["masks"]), np.array(white_records["actions"]), white_returns, white_actor_optimizer, white_critic_optimizer)
            
        loss_black_val = loss_black.numpy() if isinstance(loss_black, tf.Tensor) else loss_black
        loss_white_val = loss_white.numpy() if isinstance(loss_white, tf.Tensor) else loss_white
        
        # Log to CSV
        total_rewards_black = sum(black_records["rewards"])
        total_rewards_white = sum(white_records["rewards"])
        with open(log_filename, mode='a', newline='') as log_file:
            log_writer = csv.writer(log_file)
            log_writer.writerow([
                episode + 1, step_count, winner, "Black" if black_training_phase else "White" if white_training_phase else "None",
                total_rewards_black, total_rewards_white,
                loss_black_val, critic_loss_black.numpy() if isinstance(critic_loss_black, tf.Tensor) else critic_loss_black,
                loss_white_val, critic_loss_white.numpy() if isinstance(critic_loss_white, tf.Tensor) else critic_loss_white,
                entropy_black.numpy() if isinstance(entropy_black, tf.Tensor) else entropy_black,
                entropy_white.numpy() if isinstance(entropy_white, tf.Tensor) else entropy_white,
                black_self_dying_count, white_self_dying_count,
                black_kill_count, white_kill_count,
                black_hit_count, white_hit_count,
                black_correct_aim_count, white_correct_aim_count
            ])
        
        phase = "Black Training" if black_training_phase else "White Training" if white_training_phase else "No Training"
        print(f"Episode {episode + 1}/{start_episode + episodes} completed | Winner: {winner:5s}. | Black Loss: {loss_black_val:10.4f} | White Loss: {loss_white_val:10.4f} | Steps: {step_count:10d} | Phase: {phase:15s}", end="\r")
        
        if((episode + 1) % 10000 == 0) :  # temporary save
            black_agent.save(f"./moka_black_v24_{episode + 1}")
            white_agent.save(f"./moka_white_v24_{episode + 1}")
    
    black_agent.save("./moka_black_v24")
    white_agent.save("./moka_white_v24")
    env.close()

def test() :
    env = gym.make(
        id = "kymnasium/AlKkaGi-3x3-v0",
        render_mode = "human",
        bgm = True,
        obs_type = "custom"
    )
    black_agent = BlackAgent.load("./moka_black_v24_90000")
    white_agent = WhiteAgent.load("./moka_white_v24_90000")
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
    black_agent = BlackAgent.load("./moka_black_v24_90000")
    white_agent = WhiteAgent.load("./moka_white_v24_90000")
    kym.alkkagi.ManualPlayWrapper("kymnasium/AlKkaGi-3x3-v0", debug=True, agent=white_agent, agent_turn=1).play()
    # train()
    # test()