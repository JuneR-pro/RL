import collections
import numpy as np
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gymnasium as gym
from tqdm import tqdm
class ReplayBuffer:
    def __init__(self,capacity):
        self.buffer=collections.deque(maxlen=capacity)

    def add(self,state,action,reward,next_state,done):
        self.buffer.append((state,action,reward,next_state,done))

    def sample(self,batchsize):
        transitions=random.sample(self.buffer,batchsize)
        state,action,reward,next_state,done=zip(*transitions)
        s,a,r,ns,d=np.array(state),action,reward,np.array(next_state),done
        transition_dict={
        'states':s,
        'actions':a,
        'rewards':r,
        'next_states':ns,
        'dones':d

        }
        return transition_dict
    
    def size(self):
        return len(self.buffer)
    
class Qnet(torch.nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(Qnet,self).__init__()
        self.fc1=torch.nn.Linear(state_dim,hidden_dim)
        self.fc2=torch.nn.Linear(hidden_dim,action_dim)

    def forward(self,x):
        x= F.relu(self.fc1(x))
        return self.fc2(x)

class DQNAgent:
    def __init__(self,state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim

        self.q_net = Qnet(state_dim, hidden_dim,
                          self.action_dim).to(device)  
        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        
        self.gamma = gamma  
        self.epsilon = epsilon  
        self.target_update = target_update  
        self.count = 0  
        self.device = device
    
    def get_action(self,state):
        if np.random.random()<self.epsilon:
            action=np.random.randint(self.action_dim)
        else:
            state=torch.tensor([state],dtype=torch.float).to(self.device)
            action=self.q_net(state).argmax().item()
        return action
    def update(self,transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        q_values = self.q_net(states).gather(1, actions)

        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
            -1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones) 
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))

        self.optimizer.zero_grad()
        dqn_loss.backward() 
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

lr = 2e-3
num_episodes = 500
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

env_name = 'CartPole-v0'
env = gym.make(env_name)
random.seed(42)
np.random.seed(42)
env.reset(seed=42)
torch.manual_seed(42)
replay_buffer = ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQNAgent(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device)

return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            state,_ = env.reset()
            done = False
            while not done:
                action = agent.get_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward
                # 当buffer数据的数量超过一定值后,才进行Q网络训练
                if replay_buffer.size() > minimal_size:
                    transition_dict = replay_buffer.sample(batch_size)
                    agent.update(transition_dict)
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)

# 绘制结果
episodes_list = list(range(len(return_list)))
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))

plt.subplot(1, 2, 2)
mv_return = moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {} (Moving Average)'.format(env_name))

plt.tight_layout()
plt.show()

print(f"平均回报: {np.mean(return_list):.2f}")
print(f"最后10个episode平均回报: {np.mean(return_list[-10:]):.2f}")

