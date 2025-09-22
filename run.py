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
from pg import MCPGAgent
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import time
'''
训练
'''
def train_agent(agent, env, n_episodes):
    print("Training on env:", env.spec.id)
    for episode in tqdm(range(n_episodes)):

        obs,info=env.reset()
        done=False

        while not done:

            action=agent.get_action(obs)

            next_obs,reward,terminated,truncated,info=env.step(action)
            agent.store_reward(reward)
            # agent.update(obs,action,reward,done,next_obs)

            done=terminated or truncated
            obs=next_obs

        # agent.decay_epsilon()
        agent.update()
        print(f"Episode {episode}, Loss={agent.training_error[-1]:.3f}")
'''
绘图
'''
def plot_stats(agent, env, rolling_length=50):
    """
    绘制 REINFORCE 策略梯度训练过程
    - Episode Reward
    - Training Loss
    """

    # ====== 获取奖励 ======
    try:
        rewards = np.array(env.return_queue)  # 如果用了 RecordEpisodeStatistics
    except AttributeError:
        rewards = np.array(getattr(agent, "episode_rewards", []))

    # ====== 获取 loss ======
    losses = np.array(agent.training_error)

    # ====== 移动平均 ======
    def moving_average(arr, window):
        if len(arr) < window:
            return arr
        return np.convolve(arr, np.ones(window)/window, mode='valid')

    reward_ma = moving_average(rewards, rolling_length)
    loss_ma = moving_average(losses, rolling_length)

    # ====== 绘图 ======
    fig, axs = plt.subplots(ncols=2, figsize=(12, 5))

    # 1️.Reward 曲线
    axs[0].set_title("Episode Rewards")
    axs[0].plot(range(len(rewards)), rewards, alpha=0.3, label="Raw")
    axs[0].plot(range(len(reward_ma)), reward_ma, label=f'{rolling_length}-episode MA', linewidth=2)
    axs[0].set_ylabel("Reward")
    axs[0].set_xlabel("Episode")
    axs[0].legend()

    # 2️.Loss 曲线
    axs[1].set_title("Training Loss (per episode)")
    axs[1].plot(range(len(losses)), losses, alpha=0.3, label="Raw")
    if len(loss_ma) > 0:
        axs[1].plot(range(len(loss_ma)), loss_ma, label=f'{rolling_length}-episode MA', linewidth=2)
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Episode")
    axs[1].legend()

    plt.tight_layout()
    plt.show()
'''
测试
'''
def test_agent_single(agent, env, episodes=1, render=True):
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        path = [obs]   # 保存走过的状态
        total_reward = 0

        while not done:
            action = agent.get_action(obs, train=False)   # 测试时不采样
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            path.append(next_obs)

            if render:
                env.render()   # 这里会打开一个窗口（必须在创建环境时设置 render_mode="human"）

            obs = next_obs
            done = terminated or truncated

        print(f"Episode {ep+1}:")
        print("路径长度:", len(path)-1)
        print("总奖励:", total_reward)
        print("-"*50)

def test_agent(agent, env, num_episodes=100,max_step=20000):
    print("Testing on env:", env.spec.id)
    
    total_rewards = []

    # 纯利用
    # old_epsilon = agent.epsilon
    # agent.epsilon = 0.0  
    
    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        step_count=0

        # while not done and step_count<max_step:  
        while not done :  
            action = agent.get_action(obs,train=False)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            step_count += 1
        # agent.update()
        total_rewards.append(episode_reward)

    # # 恢复探索率
    # agent.epsilon = old_epsilon

    # 不同环境成功率的计算不同
    # win_rate = np.mean(np.array(total_rewards) > 0) 
    
    # win_rate = np.mean(np.array(total_rewards) > -100)
    average_reward = np.mean(total_rewards)

    print(f"Test Results over {num_episodes} episodes:")
    # print(f"Win Rate: {win_rate:}")
    print(f"Average Reward: {average_reward:.3f}")
    print(f"Standard Deviation: {np.std(total_rewards):.3f}")


if __name__=='__main__':
    # alpha=0.05
    # n_episodes=10000
    # start_epsilon=1.0
    # epsilon_decay=start_epsilon/ (n_episodes / 2)
    # final_epsilon=0.1
    # # 此封装器将跟踪累积奖励和剧集长度
    # env=gym.make("CliffWalking-v1")
    # env=gym.wrappers.RecordEpisodeStatistics(env,buffer_length=n_episodes)
    # observation,info=env.reset(seed=42)
    # print(f"starting observation:{observation},info:{info}")
    # agent=QLearningAgent(env=env,
    #                     learning_rate=alpha,
    #                     initial_epsilon=start_epsilon,
    #                     epsilon_decay=epsilon_decay,
    #                     final_epsilon=final_epsilon,
    #                     )
    # time1=time.time()
    # train_agent(agent, env, n_episodes)
    # time2=time.time()
    # plot_stats(agent, env)
    # time3=time.time()
    # test_agent(agent, env)
    # time4=time.time()
    # print(f"训练时间:{time2-time1},绘图时间:{time3-time2},测试时间:{time4-time3}")
    # test_agent_single(agent, env, episodes=1, render=False)

    n_eposides=5000

    env=gym.make("CartPole-v1")
    env=gym.wrappers.RecordEpisodeStatistics(env,buffer_length=n_eposides)
    agent=MCPGAgent(env,learning_rate=5e-4,discount_factor=0.99)

    time1=time.time()
    train_agent(agent, env, n_eposides)
    time2=time.time()
    plot_stats(agent, env)
    time3=time.time()
    test_agent(agent, env)
    time4=time.time()
    print(f"训练时间:{time2-time1},绘图时间:{time3-time2},测试时间:{time4-time3}")
    env=gym.make("CartPole-v1", render_mode="human")
    test_agent_single(agent, env, episodes=1, render=True)
