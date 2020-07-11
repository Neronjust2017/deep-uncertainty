pdrop = 0.1
tau = 0.1
lengthscale = 0.01

N = 364

print(lengthscale ** 2 * (1 - pdrop) / (2. * N * tau))