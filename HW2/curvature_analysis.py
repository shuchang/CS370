import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

from scipy import stats

def main():
    with open("curvature.npy", "rb") as f:
        gauss_array = np.load(f)
        mean_array = np.load(f)
    res = stats.pearsonr(gauss_array, mean_array)
    print(res)
    plt.scatter(gauss_array, mean_array)
    plt.xlabel("gaussian curvature")
    plt.ylabel("mean curvature")
    plt.show()
    sns.kdeplot(gauss_array)
    plt.show()
    sns.kdeplot(mean_array)
    plt.show()


if __name__ == "__main__":
    main()