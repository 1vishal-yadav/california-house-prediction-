import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv("output.csv")
x = data["predictions"]
y = data["median_house_value"]
plt.scatter(x, y, alpha=0.2)
plt.xlabel("Predictions")
plt.ylabel("Median House Value")
plt.title("Housing Data")
plt.grid(True)
plt.show()