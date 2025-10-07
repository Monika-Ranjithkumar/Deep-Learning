import torch
import torch.nn as nn
import numpy as np

# 1. Generate synthetic user-movie interaction data
# Let's simulate 20 users and 30 movies
num_users = 20
num_movies = 30

# Simulate binary watch history (1 = watched/liked, 0 = not watched)
# Most users have seen about 20-60% of the movies
np.random.seed(42)
user_movie_matrix = np.random.binomial(1, p=0.4, size=(num_users, num_movies)).astype(np.float32)

# Convert to torch tensor
train_data = torch.tensor(user_movie_matrix)

# 2. Define RBM model
class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.v_bias = nn.Parameter(torch.zeros(n_visible))

    def sample_h(self, v):
        h_prob = torch.sigmoid(torch.matmul(v, self.W.t()) + self.h_bias)
        return h_prob, torch.bernoulli(h_prob)

    def sample_v(self, h):
        v_prob = torch.sigmoid(torch.matmul(h, self.W) + self.v_bias)
        return v_prob, torch.bernoulli(v_prob)

    def contrastive_divergence(self, v0, k=1, lr=0.01):
        vk = v0.clone()
        for _ in range(k):
            _, hk = self.sample_h(vk)
            _, vk = self.sample_v(hk)
            vk[v0 == 0] = v0[v0 == 0]  # keep 0s (unwatched)

        h0_prob, _ = self.sample_h(v0)
        hk_prob, _ = self.sample_h(vk)

        self.W.data += lr * (torch.matmul(h0_prob.t(), v0) - torch.matmul(hk_prob.t(), vk)) / v0.size(0)
        self.v_bias.data += lr * torch.mean(v0 - vk, dim=0)
        self.h_bias.data += lr * torch.mean(h0_prob - hk_prob, dim=0)

# 3. Train the RBM
n_visible = num_movies
n_hidden = 16

rbm = RBM(n_visible, n_hidden)
epochs = 20

for epoch in range(epochs):
    loss = 0
    for user in range(train_data.size(0)):
        v = train_data[user:user+1]
        rbm.contrastive_divergence(v, k=1)
        v_recon = rbm.sample_v(rbm.sample_h(v)[1])[0]
        loss += torch.mean(torch.abs(v - v_recon)).item()
    print(f"Epoch {epoch+1}, Loss: {loss / num_users:.4f}")

# 4. Recommend unseen movies for a test user
def recommend_movies(user_id, top_n=5):
    user_input = train_data[user_id:user_id+1]
    reconstructed, _ = rbm.sample_v(rbm.sample_h(user_input)[1])
    unseen_movies = (user_input == 0).squeeze()
    recommendations = reconstructed.squeeze()[unseen_movies]
    movie_indices = torch.topk(recommendations, top_n).indices
    print(f"\nRecommended movies for User {user_id}:")
    for idx in movie_indices:
        print(f"Movie {idx.item()} (Predicted score: {reconstructed[0][idx].item():.3f})")

# 5. Run recommendation for user 5
recommend_movies(user_id=5)
