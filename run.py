'''
文件名： run.py
作者： JuneR
创建日期： 2025-09-16
描述: 
    1.训练
    2.绘图
    3.测试

'''
import gymnasium as gym
from td import SarsaAgent,QLearningAgent
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
'''
训练
'''
def train_agent(agent, env, n_episodes):
    for episode in tqdm(range(n_episodes)):

        obs,info=env.reset()
        done=False

        while not done:

            action=agent.get_action(obs)

            next_obs,reward,terminated,truncated,info=env.step(action)
            agent.update(obs,action,reward,terminated,next_obs)

            done=terminated or truncated
            obs=next_obs

        agent.decay_epsilon()

'''
绘图
'''
def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window
def plot_stats(agent, env):
    # Smooth over a 500-episode window
    rolling_length = 500
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    # Episode rewards (win/loss performance)
    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(
        env.return_queue,
        rolling_length,
        "valid"
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")

    # Episode lengths (how many actions per hand)
    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(
        env.length_queue,
        rolling_length,
        "valid"
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")

    # Training error (how much we're still learning)
    axs[2].set_title("Training Error")
    training_error_moving_average = get_moving_avgs(
        agent.training_error,
        rolling_length,
        "same"
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    axs[2].set_ylabel("Temporal Difference Error")
    axs[2].set_xlabel("Step")

    plt.tight_layout()
    plt.show()
'''
测试
'''
def test_agent(agent, env, num_episodes=1000):
    """Test agent performance without learning or exploration."""
    total_rewards = []

    # 纯利用
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  
    
    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)

    # 恢复探索率
    agent.epsilon = old_epsilon

    win_rate = np.mean(np.array(total_rewards) > 0)
    average_reward = np.mean(total_rewards)

    print(f"Test Results over {num_episodes} episodes:")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Average Reward: {average_reward:.3f}")
    print(f"Standard Deviation: {np.std(total_rewards):.3f}")

if __name__=='__main__':
    env=gym.make('Blackjack-v1')
    observation,info=env.reset(seed=42)
    print(f"starting observation:{observation},info:{info}")
    alpha=0.01
    n_episodes=100000
    start_epsilon=1.0
    epsilon_decay=start_epsilon/ (n_episodes / 2)
    final_epsilon=0.1

    env=gym.make("Blackjack-v1",sab=False)
    # 二十一点环境
    env=gym.wrappers.RecordEpisodeStatistics(env,buffer_length=n_episodes)
    # 此封装器将跟踪累积奖励和剧集长度
    agent=SarsaAgent(env=env,
                        learning_rate=alpha,
                        initial_epsilon=start_epsilon,
                        epsilon_decay=epsilon_decay,
                        final_epsilon=final_epsilon,
                        )
    
    train_agent(agent, env, n_episodes)
    plot_stats(agent, env)
    test_agent(agent, env)