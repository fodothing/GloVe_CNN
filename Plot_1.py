import matplotlib.pyplot as plt
fig, ax = plt.subplots()

x = [0, 1]
x_1 = [0, 1]

y = [0, 1]
y_1 = [1, 0]

ax.scatter(x, y, marker='o')
ax.scatter(x_1, y_1, marker='x')

plt.show()

plt.figure(1)
fig.savefig('figure_1.png', dpi=900, format='png')