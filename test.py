import os
import matplotlib.pyplot as plt

folder = r"C:\Users\nguye\Desktop\ComputerVison\face_estimation\data\output\all"
ages = []
for fname in os.listdir(folder):
    if fname.endswith('.jpg.chip.jpg'):
        age = int(fname.split('_')[0])
        ages.append(age)
plt.hist(ages, bins=range(0, 80, 2))
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age distribution in training data')
plt.show()