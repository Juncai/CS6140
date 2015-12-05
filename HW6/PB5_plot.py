from matplotlib import pyplot as plt

# plot points
x1 = (1, 2, 2)
y1 = (1, 2, 0)
x2 = (0, 1, 0)
y2 = (0, 0, 1)


plt.scatter(x1, y1, color='red')
plt.scatter(x2, y2, color='blue')

# plot hyperplane
# y = wx + b; w = -1, b = 1.5
x3 = (-0.5, 2)
y3 = (2, -0.5)
plt.plot(x3, y3, c='orange')


plt.title("PB5 plotting")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

