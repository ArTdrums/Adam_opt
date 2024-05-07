import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
f = lambda x: 0.02 * x + 2
x = np.linspace(-2, 2.5, 10)

plt.plot(x, f(x), label='true')
plt.xlim([-2, 2.5])
plt.legend()
plt.show()

n = 50
np.random.seed(9)
x_train = np.random.uniform(-2, 2.5, n)
y_train = f(x_train) + 0.01 * np.random.randn(n)

weight_1 = 0.5
weight_0 = -9


def gr_mserror(X, w0, w1, y):
    y_pred = w1 * X + w0
    return np.array([2 / len(X) * np.sum((y - y_pred)) * (-1),
                     2 / len(X) * np.sum((y - y_pred) * (-X))])

k = 100
lr = 0.1
amount = 10





weights_0, weights_1 = [weight_0], [weight_1]

alphas = [0, 0] # накопленные градиенты
velocity = [0, 0] # наша история

eps = 10e-7

b1 = 0.6 # импусльс
b2 = 0.9 # накопленные градиенты
for i in range(k):
    idx = np.random.choice(len(x_train), size=amount, replace=False)
    grad = gr_mserror(x_train[idx], weights_0[-1], weights_1[-1], y_train[idx])

    velocity[0] = b1 * velocity[0] + grad[0]
    velocity[1] = b1 * velocity[1] + grad[1]

    alphas[0] = b2 * alphas[0] + (1 - b2) * grad[0] ** 2
    alphas[1] = b2 * alphas[1] + (1 - b2) * grad[1] ** 2

    new_w_0 = weights_0[-1] - (lr * velocity[0]) / np.sqrt(alphas[0] + eps)
    new_w_1 = weights_1[-1] - (lr * velocity[1]) / np.sqrt(alphas[1] + eps)

    weights_0.append(new_w_0)
    weights_1.append(new_w_1)

adam_sgd_weights_0 = weights_0.copy()
adam_sgd_weights_1 = weights_1.copy()
pred = weights_0[-1] + weights_1[-1] * x_train

plt.plot(x, f(x), label='true')
plt.scatter(x_train, y_train, label='train')
plt.scatter(x_train, pred, label='pred')
plt.xlim([-2, 2.5])
plt.legend()
plt.show()





def mse(w0, w1):
    y_pred = w1 * x_train + w0
    return np.mean((y_train - y_pred) ** 2)


coefs_a = np.linspace(-2, 2, num=100)
coefs_b = np.linspace(-10, 10, num=100)
w1, w0 = np.meshgrid(coefs_a, coefs_b)


fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')

zs = np.array([mse(i, j) for i, j in zip(np.ravel(w0), np.ravel(w1))])
Z = zs.reshape(w1.shape)

ax.plot_surface(w0, w1, Z, alpha=.5)
ax.scatter(weight_0, weight_1, mse(weight_0, weight_1), c='r', s=5)

mses = []
for i in range(len(adam_sgd_weights_0)):
    mses.append(mse(adam_sgd_weights_0[i], adam_sgd_weights_1[i]))
ax.plot(adam_sgd_weights_0, adam_sgd_weights_1, mses, marker='*')

ax.set_xlabel(r'$w_0$')
ax.set_ylabel(r'$w_1$')
ax.set_zlabel('MSE')

plt.show()



'''from keras.optimizers import Adam
Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)'''