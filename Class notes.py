import random
import time
import numpy as np

N = 10**7
a = []
# list of 10 million numbers
for _ in range(N):
  a.append(random.randint(1, N))

start = time.perf_counter()
a.sort()
end = time.perf_counter()
print(f"classic took {end-start}")

a = np.random.randint(1, N, N)
start = time.perf_counter()
a.sort()
end = time.perf_counter()
print(f"NumPy took {end-start}")

import numpy as np

import numpy as np
a = np.array([1, 2, 3, 4, 5, 6, 7])
print(a, type(a))
print(a+2)
print(a/2)
print(a+a)
print(a**2)

a = np.array([[1, 2, 3, 4, 5],
              [6, 7, 8, 9, 10]])
print(a)
print (type(a))
print(f"{a.ndim}, {a.size}, {a.shape}, {a.dtype}\n")

a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], dtype=np.int16)
print(a)
print(f"{a.ndim}, {a.size}, {a.shape}, {a.dtype}")

a0 = np.zeros((4,10), dtype=int)
a1 = np.ones((2, 3, 4))
print(a0)
print(a1)
a11 = np.ones((2, 3, 4), dtype=np.int16)
print(a11)


a1 = np.array([1, 2, 3])
a2 = np.array([4, 5, 6])
print(f"a1={a1}, a2={a2}")
print(f"a1+a2={a1+a2}, a1*a2={a1*a2}")
a1 = np.array([[1, 2, 3], [4, 5, 6]])
print(a1+a2)
print("\n\n")


print(10/a1)
print(a1 * 2.5)
print(a1 > 3)


a = np.array(
    [[1, 2, 3, 4, 5],
     [6, 7, 8, 9, 10]]
     )
print(a[0][1]) # row and column
print(a[1, 2]) # same thing

a = np.random.randint(10, 100, size=(3, 4, 5))
print(f"a={a}\n")
print(f"a[1] = \n {a[1]}\n")
print(f"a[1, 2] = \n{a[1, 2]}\n")
print(a[1, 1, 4])

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(a)
print(a[1:5])
slice = a[1:5]
slice[0] = 99
print(a)
slice[:] = 99 # careful here with slice = 99
print(a)
print("\n\n")


a = np.arange(12)
a = a.reshape((3, 4))
print(f"a=\n{a}")
print(f"a[:, 1]=\n{a[:, 1]}")
print(f"a[:2, :2]=\n{a[:2, :2]}")
a[:2, :2] = 99
print(f"a=\n{a}")


names = np.array(["Bob", "James", "Mary", "Bob", "Jane", "Bob", "James", "Jane"])
print(names == "Bob")
a = np.arange(24).reshape((8, 3))
print(f"a=\n{a}")
print(f"a[names == 'Bob']=\n {a[names == 'Bob']}")
a[names=="Jane", :2 ] = 99
print(f"a=\n{a}")

print(f"{a[(names == 'Bob') | (names=='Jane')]}") # or=| and=&

a = np.zeros((8, 3), dtype=np.int16)
for i in range(8):
  a[i] = i

print(a)
fancy_index = [1, 3, 5]
print(f"a[fancy] = \n{a[fancy_index]}")

fancy_index2 = [-1, -2, -4]
print(f"a[fancy2] = \n{a[fancy_index2]}")

a = np.arange(24).reshape((8, 3))
print(f"a=\n{a}")
fancy1 = [0, 2, 4, 5, 7, 2]
fancy2 = [0, 0, 1, 2, 0, 1]
print(f"a[fancy_fancy] = \n{a[fancy1, fancy2]}")


import numpy as np
a = np.zeros((3, 3))
a[0, 0] = 1
a[1, 1] = 2
a[2, 2] = 3
b = np.linalg.inv(a)
print(f"a=\n{a}")
print(f"b=\n{b}")
print(f"aXb=\n{a.dot(b)}")
print(f"aXb=\n{a @ b}")
print("\n")

a = np.ones((3, 4))
print(a)
b = a.T
print(b)
print(f"aXb=\n{a @ b}")

! wget -O data.csv https://raw.githubusercontent.com/itb-ie/HomeSchool5/master/MOD_LSTD_E_2021-08-29_gs_720x360.csv

import numpy as np
import matplotlib.pyplot as plt


dat = np.genfromtxt('data.csv', delimiter=',')
dat = np.where(dat<1000, dat, np.NaN)
# dat = np.where(dat<1000, dat, 10)

dat = dat[::-1, :]
lon = np.arange(-180.0, 180.1, 0.5)
lat = np.arange(-90.0, 90.1, 0.5)
date = '2021-08-29 to 2021-09-05'
source = 'https://neo.sci.gsfc.nasa.gov/view.php?datasetId=MOD_LSTD_E&date=2021-09-01'

fig, ax = plt.subplots(constrained_layout=True, figsize=(14, 8))
ax.set_facecolor('0.6')
pc = ax.pcolormesh(lon, lat, dat, shading='auto', cmap='inferno')
ax.set_aspect(1.3)
ax.set_xlabel('Longitude $[^o E]$')
ax.set_ylabel('Latitude $[^o N]$')
fig.colorbar(pc, shrink=0.6, extend='both', label='land temperature $[^oC]$');
ax.set_title(f'source: {source}', loc='left', fontsize='x-small');

# Import seaborn
import seaborn as sns

# Apply the default theme
sns.set_theme()

# Load an example dataset
tips = sns.load_dataset("tips")

# Create a visualization
sns.relplot(
    data=tips,
    x="total_bill", y="tip", col="time",
    hue="smoker", style="sex", size="size",
)

import altair as alt
from vega_datasets import data
source = data.cars()
brush = alt.selection(type='interval')
points = alt.Chart(source).mark_point().encode(
    x='Horsepower:Q',
    y='Miles_per_Gallon:Q',
    color=alt.condition(brush, 'Origin:N', alt.value('lightgray'))
).add_selection(
    brush
)
bars = alt.Chart(source).mark_bar().encode(
    y='Origin:N',
    color='Origin:N',
    x='count(Origin):Q'
).transform_filter(
    brush
)
plot = points & bars
plot

import matplotlib.pylab as plt
x = y = list(range(100))
y2 = list(range(0, 200, 2))
y3 = list(range(0 ,300, 3))
plt.plot(x, y)
plt.plot(x, y2)
plt.plot(x, y3)

plt.show()

import matplotlib.pylab as plt
import random
x = list(range(100))
y = list(range(100))
random.shuffle(y)

print(y)
plt.plot(x, y)

plt.show()

import matplotlib.pylab as plt
x = y = list(range(100))
y2 = list(range(100, 0, -1))

fig, ax = plt.subplots(2,2)
ax[0, 0].plot(x, y)
ax[1, 0].plot(x, y2)
ax[0, 1].plot(x, y2)
ax[1, 1].plot(x, y)



plt.show()

import matplotlib.pylab as plt
x = y = list(range(100))
y2 = [50 for i in range(100)]
y3 = [100-i for i in range(100)]
y3 = [i for i in range(100,0,-1)]
y3 = list(range(100,0,-1))

fig, axs = plt.subplots(2,3, figsize=(20, 10))
for line in range(0,2):
  for col in range(0,3):
    axs[line, col].set_title(f"Title {line+1}/{col+1}")
    axs[line, col].plot(x, y)
    axs[line, col].plot(x, y2)
    axs[line, col].plot(x, y3)

plt.show()

import matplotlib.pylab as plt
import numpy as np

x = np.linspace(0, 200, 50)
y = x**2
y2 = x**3/100

# plt.rcParams.update({'font.size': 16})
plt.figure(figsize = (10, 10))

plt.title("$X^2$ vs $X^3$")
plt.plot(x, y)
plt.plot(x, y2)
plt.show()

import matplotlib.pylab as plt
import numpy as np

x = np.linspace(0, 100, 40)
y1 = x**2
y2 = 10*x

plt.rcParams.update({'font.size': 16})
fig, axis = plt.subplots(1, 3, figsize=(20, 10), sharey=False)


axis[0].set_title("Y =  $X^2$")
axis[0].plot(x, y1)

axis[1].set_title("Y = 10*X")
axis[1].plot(x, y2)

axis[2].set_title("Now together")
axis[2].plot(x, y1)
axis[2].plot(x, y2)


plt.show()

import matplotlib.pylab as plt
import numpy as np

# data
x = np.linspace(0, 5*np.pi, 200)
y1 = np.sin(x)
y2 = np.cos(x)

# figure
plt.figure(figsize=(20, 8))
plt.title("Sin and Cos Graph")

plt.plot(x, y1, "r", label="sin", markersize=20)
plt.plot(x, y2, "k", label="cos", markersize=2)
plt.legend()
plt.show()

#@title Pie Charts

import matplotlib.pyplot as plt

labels = 'Looks like Pacman', 'Does not look like Pacman'
sizes = [84, 16]
colors = ["#f0f000", "#000000"]
plt.rcParams.update({'font.size': 12})

fig, axes = plt.subplots(1, 3, figsize=(20, 10))

axes[0].pie(sizes, startangle=30, colors=colors, explode=(0.1, 0.2))
axes[0].legend(labels, loc="upper left")

axes[1].pie([10, 20, 30, 40], explode=(0, 0.2, 0, 0), labels=("", "My Piece", "", ""))
axes[1].set_title("Madrid Real Estate")

axes[2].pie([10, 10, 35, 30, 15], explode=(0, 0.1, 0, 0, 0.2),
            labels=("", "My Piece", "", "", "Your piece"), shadow=True,
            autopct='%1.1f%%')
plt.show()