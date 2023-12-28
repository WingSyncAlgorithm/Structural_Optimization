# train_optimization.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# from structure_test import create_structure
# from battle_test import win

class Actor(nn.Module):
    def __init__(self):
      """
      初始化
      """
      super(Actor, self).__init__()
      self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1)
      self.conv2 = nn.Conv3d(64, 1, kernel_size=3, stride=1, padding=1)
      self.fc1 = nn.Linear(64 * 64 * 64, 256)
      self.fc2 = nn.Linear(256, 1)

    def forward(self, boundary, num_points=10):
      """
      前向傳播

      Args:
        boundary(tensor,四維) : 邊界
      Returns;
        points(tensor,四維，形狀和boundary一樣) : 點
        probability(list,一維,0~100間的正實數) : 機率
      """
      x = self.conv1(boundary)
      x = torch.relu(x)
      x = self.conv2(x)
      x = torch.relu(x)
      x = x.view(x.size(0), -1)
      x = self.fc1(x)
      x = torch.relu(x)
      probability = torch.sigmoid(self.fc2(x))
      probability *= 100
      points = self.generate_points(boundary, num_points)
      return points, probability

    def generate_points(self, boundary, num_points):
      """
      生成點
      
      策略:
        先生成一個和邊界形狀大小一樣的點
        對每個batch進行迭代
        隨機生成idx
        idx在邊界或已經有點就跳過
        直到生滿num_points

      Args:
        boundary(tensor,四維) : 邊界
        num_points(int) : 要生成的點數
      Returns;
        points(tensor,四維，形狀和boundary一樣) : 點
        probability(list,一維,0~100間的正實數) : 機率
      """
      points = torch.zeros_like(boundary)
      num_points_generated = 0

      for i in range(boundary.size(0)):
          while num_points_generated < num_points:
              random_idx1 = torch.randint(0, boundary.size(1), (1,))
              random_idx2 = torch.randint(0, boundary.size(2), (1,))
              random_idx3 = torch.randint(0, boundary.size(3), (1,))

              if boundary[i, random_idx1, random_idx2, random_idx3] == 1 or points[i, random_idx1, random_idx2, random_idx3] == 1:
                  continue
              else:
                  points[i, random_idx1, random_idx2, random_idx3] = 1
                  num_points_generated += 1

      return points


def train_RL(actor_network, optimizer, num_episodes=1000):
  """
  訓練函數
  Args:
    actor_network(class):這邊用來生成點和機率
    optimizer:優化器
    num_episodes(int): 訓練次數
  Returns;
  """
  for episode in tqdm(range(num_episodes), desc="Training Episodes"):

    points, probability = actor_network(boundary)
    structure = create_structure(boundary, points, probability)

    '''
    我不知道要和誰比，所以就先假設前面那個比，我認為strcture應該是一個list?
    score1, score2 = win(structure, pre_structure)

    '''
    score1, score2 = win(structure[episode+1], structure[episode])


    loss = -score1
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


actor_network = Actor()
optimizer = optim.Adam(actor_network.parameters(), lr=1e-3)
train_RL(actor_network, optimizer)
