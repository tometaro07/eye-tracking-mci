import os
import warnings
import numpy as np
import pickle

seed = 42

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONHASHSEED"] = str(seed)
os.environ["MPLCONFIGDIR"] = os.getcwd() + "/configs/"

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=Warning)

np.random.seed(seed)

PROCCESS_DIR = ""
UNPROCESS_DIR = ""

# LOAD DATA

data = []

for file in os.listdir(UNPROCESS_DIR):
    with open(UNPROCESS_DIR + file, "rb+") as f:
        data += [pickle.load(f)]
    if "patient_s33_9_12" in file:
        print(data[-1][-1].encoding)

# REMOVE MISSING VALUES

encoding = ""
recognition = ""
for i in range(len(data)):
    for j in range(len(data[i])):

        guide = np.any(np.isnan(data[i][j].encoding), axis=1)

        if np.sum(guide) > len(data[i][j].encoding) * 0.2:
            data[i][j].encoding = None
            encoding += f"{data[i][j].group} {data[i][j].subject} {data[i][j].trial}\n"
        else:
            data[i][j].encoding = data[i][j].encoding[~guide]

        guide = np.any(np.isnan(data[i][j].recognition), axis=1)

        if np.sum(guide) > len(data[i][j].recognition) * 0.2:
            data[i][j].recognition = None
            recognition += (
                f"{data[i][j].group} {data[i][j].subject} {data[i][j].trial}\n"
            )
        else:
            data[i][j].recognition = data[i][j].recognition[~guide]

with open("enconding_removed.txt", "w") as output:
    output.write(encoding)

with open("recognition_removed.txt", "w") as output:
    output.write(recognition)

results = {"control": {}, "patient": {}}
lists = {"control": [], "patient": []}

for i in range(len(data)):
    group = data[i][0].group
    subject = int(data[i][0].subject)
    for j in range(len(data[i])):
        if data[i][j].encoding is not None:
            if subject not in list(results[group].keys()):
                results[group][subject] = []
            results[group][subject] += [data[i][j].isCorrect]

new_sub = []
for i in range(len(data)):
    if data[i][0].subject not in new_sub:
        new_sub += [data[i][0].subject]
        lists[data[i][0].group] += [data[i][0].trial.split("_")[0]]


s = {"control": {}, "patient": {}}

for g in results.keys():
    for t in results[g].keys():
        s[g][t] = len(results[g][t])
        results[g][t] = np.mean(results[g][t])

total_s = [sum(list(s["control"].values())), sum(list(s["patient"].values()))]

min_group, min_value = np.argmin(total_s), np.min(total_s)
max_group, max_value = 1 - min_group, total_s[1 - min_group]

min_group = "control" if min_group == 0 else "patient"
max_group = "control" if max_group == 0 else "patient"

max_sub, max_val = list(results[max_group].keys())[
    np.argmax(list(results[max_group].values()))
], np.max(list(results[max_group].values()))

while min_value < max_value - s[max_group][max_sub]:
    print(
        results[max_group].pop(max_sub),
        max_group,
        max_sub,
    )

    max_value -= s[max_group][max_sub]

    s[max_group].pop(max_sub)

    max_sub, max_val = list(results[max_group].keys())[
        np.argmax(list(results[max_group].values()))
    ], np.max(list(results[max_group].values()))

print(max_value, min_value, max_value + min_value)


for i in range(len(data) - 1, -1, -1):
    if (
        int(data[i][0].subject) not in list(s[max_group].keys())
        and data[i][0].group == max_group
    ):
        print(data[i][0].subject, data[i][0].group)
        data.pop(i)

del (
    results,
    encoding,
    output,
    s,
    min_group,
    min_value,
    max_group,
    max_value,
    max_sub,
    max_val,
)
# SAVE DATA
j = 0

for k in range(len(data) - 1, -1, -1):
    group = data[k][0].group
    subject = int(data[k][0].subject // 1)
    alter = "" if data[k][0].subject % 1 == 0 else "_5"
    directory = ""

    for i in range(len(data[k])):
        if data[k][i].encoding is not None:
            data[k][i].build_scanpath(resize=(256, 256))
            data[k][i].build_heatmap(distance=57, angle=1, resize=(256, 256))

    j += 1
    print(j)
    with open(f"{PROCCESS_DIR}{directory}{group}_s{subject}{alter}.pkl", "wb") as file:
        pickle.dump(data.pop(k), file)
