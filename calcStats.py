import sys
import os
import numpy as np
from statistics import calc_precision, calc_recall, calc_sorensen, class_stats

directory = sys.argv[1]

splits = os.listdir(directory)
splits = [s for s in splits if "split" in s]
wholeMatrix = None

out_str = directory + "\n\n"

for split in sorted(splits):
    mat = np.load("{}/{}/confusion_matrix.npy".format(directory, split))
    if wholeMatrix is None:
        wholeMatrix = mat
    else:
        wholeMatrix += mat
    
    out_str += "{}\n".format(split)
    out_str += "{}\n\n".format(str(calc_sorensen(class_stats(mat))))
   

all_stats = class_stats(wholeMatrix)
out_str += "Whole dataset:\n"
out_str += "Dice\n"
out_str += "{}\n".format(str(calc_sorensen(all_stats)))
out_str += "Prec\n"
out_str += "{}\n".format(str(calc_precision(all_stats)))
out_str += "Recall\n"
out_str += "{}\n".format(str(calc_recall(all_stats)))
print(out_str)
print(all_stats)
print(wholeMatrix)
with open("{}/stats.txt".format(directory), "w") as file:
    file.write(out_str)