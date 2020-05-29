# povlov dog model test
# 2019/12/07
# by Caiye
import numpy as np
from matplotlib import pyplot as plt

n_food = 0
n_ring = 0
n_sali = 0
w_food = 0.99
w_ring = 0.1
lr = 0.1
fr = 0.05

Nf = np.array([])
Nr = np.array([])
Ns = np.array([])
Wr = np.array([])

# Test1
for i in range(10):
    n_food, n_ring = 0, 1
    n_sali = n_food * w_food + n_ring * w_ring
    d_w = lr*w_ring*n_food*n_ring - fr*w_ring*n_ring*(1-n_food) - fr*w_ring*w_food*(1-n_ring)
    # w_ring = w_ring + d_w

    if w_ring > w_food:
        w_ring = w_food
    elif w_ring <= 0:
        w_ring = 0

    if n_sali > 0.99:
        n_sali = 1

    Nf = np.append(Nf, n_food)
    Nr = np.append(Nr, n_ring)
    Ns = np.append(Ns, n_sali)
    Wr = np.append(Wr, w_ring)

# Test1
for i in range(10):
    n_food, n_ring = 1, 0
    n_sali = n_food * w_food + n_ring * w_ring
    d_w = lr*w_ring*n_food*n_ring - fr*w_ring*n_ring*(1-n_food) - fr*w_ring*w_food*(1-n_ring)
    # w_ring = w_ring + d_w

    if w_ring > w_food:
        w_ring = w_food
    elif w_ring <= 0:
        w_ring = 0

    if n_sali > 0.99:
        n_sali = 1

    Nf = np.append(Nf, n_food)
    Nr = np.append(Nr, n_ring)
    Ns = np.append(Ns, n_sali)
    Wr = np.append(Wr, w_ring)

# Learning1
for i in range(30):
    n_food, n_ring = 1, 1
    n_sali = n_food * w_food + n_ring * w_ring
    d_w = lr*w_ring*n_food*n_ring - fr*w_ring*n_ring*(1-n_food) - fr*w_ring*w_food*(1-n_ring)
    w_ring = w_ring + d_w

    if w_ring > w_food:
        w_ring = w_food
    elif w_ring <= 0:
        w_ring = 0

    if n_sali > 0.99:
        n_sali = 1

    Nf = np.append(Nf, n_food)
    Nr = np.append(Nr, n_ring)
    Ns = np.append(Ns, n_sali)
    Wr = np.append(Wr, w_ring)

# Test2
for i in range(10):
    n_food, n_ring = 0, 1
    n_sali = n_food * w_food + n_ring * w_ring
    d_w = lr*w_ring*n_food*n_ring - fr*w_ring*n_ring*(1-n_food) - fr*w_ring*w_food*(1-n_ring)
    # w_ring = w_ring + d_w

    if w_ring > w_food:
        w_ring = w_food
    elif w_ring <= 0:
        w_ring = 0

    if n_sali > 0.99:
        n_sali = 1

    Nf = np.append(Nf, n_food)
    Nr = np.append(Nr, n_ring)
    Ns = np.append(Ns, n_sali)
    Wr = np.append(Wr, w_ring)

# Forgetting1
for i in range(40):
    n_food, n_ring = 1, 0
    n_sali = n_food * w_food + n_ring * w_ring
    d_w = lr*w_ring*n_food*n_ring - fr*w_ring*n_ring*(1-n_food) - fr*w_ring*w_food*(1-n_ring)
    w_ring = w_ring + d_w

    if w_ring > w_food:
        w_ring = w_food
    elif w_ring <= 0:
        w_ring = 0

    if n_sali > 0.99:
        n_sali = 1

    Nf = np.append(Nf, n_food)
    Nr = np.append(Nr, n_ring)
    Ns = np.append(Ns, n_sali)
    Wr = np.append(Wr, w_ring)

# Test3
for i in range(10):
    n_food, n_ring = 0, 1
    n_sali = n_food * w_food + n_ring * w_ring
    d_w = lr*w_ring*n_food*n_ring - fr*w_ring*n_ring*(1-n_food) - fr*w_ring*w_food*(1-n_ring)
    # w_ring = w_ring + d_w

    if w_ring > w_food:
        w_ring = w_food
    elif w_ring <= 0:
        w_ring = 0

    if n_sali > 0.99:
        n_sali = 1

    Nf = np.append(Nf, n_food)
    Nr = np.append(Nr, n_ring)
    Ns = np.append(Ns, n_sali)
    Wr = np.append(Wr, w_ring)

# Learning2
for i in range(30):
    n_food, n_ring = 1, 1
    n_sali = n_food * w_food + n_ring * w_ring
    d_w = lr*w_ring*n_food*n_ring - fr*w_ring*n_ring*(1-n_food) - fr*w_ring*w_food*(1-n_ring)
    w_ring = w_ring + d_w

    if w_ring > w_food:
        w_ring = w_food
    elif w_ring <= 0:
        w_ring = 0

    if n_sali > 0.99:
        n_sali = 1

    Nf = np.append(Nf, n_food)
    Nr = np.append(Nr, n_ring)
    Ns = np.append(Ns, n_sali)
    Wr = np.append(Wr, w_ring)

# Forgetting2
for i in range(40):
    n_food, n_ring = 0, 1
    n_sali = n_food * w_food + n_ring * w_ring
    d_w = lr*w_ring*n_food*n_ring - fr*w_ring*n_ring*(1-n_food) - fr*w_ring*w_food*(1-n_ring)
    w_ring = w_ring + d_w

    if w_ring > w_food:
        w_ring = w_food
    elif w_ring <= 0:
        w_ring = 0

    if n_sali > 0.99:
        n_sali = 1

    Nf = np.append(Nf, n_food)
    Nr = np.append(Nr, n_ring)
    Ns = np.append(Ns, n_sali)
    Wr = np.append(Wr, w_ring)

# Test4
for i in range(10):
    n_food, n_ring = 1, 0
    n_sali = n_food * w_food + n_ring * w_ring
    d_w = lr*w_ring*n_food*n_ring - fr*w_ring*n_ring*(1-n_food) - fr*w_ring*w_food*(1-n_ring)
    # w_ring = w_ring + d_w

    if w_ring > w_food:
        w_ring = w_food
    elif w_ring <= 0:
        w_ring = 0

    if n_sali > 0.99:
        n_sali = 1

    Nf = np.append(Nf, n_food)
    Nr = np.append(Nr, n_ring)
    Ns = np.append(Ns, n_sali)
    Wr = np.append(Wr, w_ring)

# Test4
for i in range(10):
    n_food, n_ring = 0, 1
    n_sali = n_food * w_food + n_ring * w_ring
    d_w = lr*w_ring*n_food*n_ring - fr*w_ring*n_ring*(1-n_food) - fr*w_ring*w_food*(1-n_ring)
    # w_ring = w_ring + d_w

    if w_ring > w_food:
        w_ring = w_food
    elif w_ring <= 0:
        w_ring = 0

    if n_sali > 0.99:
        n_sali = 1

    Nf = np.append(Nf, n_food)
    Nr = np.append(Nr, n_ring)
    Ns = np.append(Ns, n_sali)
    Wr = np.append(Wr, w_ring)

plt.subplot(411)
plt.plot(Nf, 'g')
plt.ylabel("N_food")

plt.subplot(412)
plt.plot(Nr, 'b')
plt.ylabel("N_ring")

plt.subplot(413)
plt.plot(Ns, 'r')
plt.ylabel("N_sali")

plt.subplot(414)
plt.plot(Wr)
plt.ylabel("W_ring")
plt.ylabel("W_ring")
plt.show()
print(Nf)
print(Nr)
print(Ns)
print(Wr)
