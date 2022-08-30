import numpy as np
import matplotlib.pyplot as plt
path = "/home/sadhanalolla/list_attr_celeba.txt"

with open(path, "r") as f:
    l = f.readlines()
    l = [i.split(" ") for i in l]
    pale_skin = [l[i][20] for i in range(len(l)) if i > 1]
    pale_skin = [int(i) for i in pale_skin if i != ""]
    arr, frequencies = np.unique(pale_skin, return_counts=True)
print(arr, frequencies)
plt.bar(arr, frequencies, tick_label=["Pale Skin", "Dark Skin"])
plt.savefig("frequencies_skin_tone.png")