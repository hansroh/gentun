import ast
generations=[]
current_gen=0
with open("geneticcnn" , 'r') as f:
    for line in f.readlines():
        if "Evaluating generation" in line:
            current_gen=int(line.split('#')[1].split('.')[0])-1
            generations.append({"individuals":0,"accuracy":0,"fittest":{}})
        if "Got fitness for" in line:
            generations[current_gen]["individuals"]+=1
        if "S_1" in line:
            generations[current_gen]["fittest"]=ast.literal_eval(line)
        if "Fitness value is" in line:
            generations[current_gen]["accuracy"]=float(line.split(':')[1])

print("Total Individuals:",len(generations)*generations[0]["individuals"])
print("Repeating Individuals:",len(generations)*generations[0]["individuals"]-sum([x["individuals"] for x in generations]))
print("New Individual Trainings:",sum([x["individuals"] for x in generations]))

print("Accuracy Achieved in First Generation:",generations[0]["accuracy"])
print("Maximum Accuracy Achieved:",max([x["accuracy"] for x in generations]))
print("Minimum Accuracy Achieved:",min([x["accuracy"] for x in generations]))
print("Average Accuracy Achieved: %.4f" % (sum([x["accuracy"] for x in generations])/len(generations)))
print("Accuracy Achieved in First Generation:",generations[len(generations)-1]["accuracy"])

import matplotlib.pyplot as plt
import numpy as np
ind = np.arange(len(generations))    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence
p1 = plt.bar(ind, [x["individuals"] for x in generations], width)
plt.ylabel('Individuals')
plt.title('Individual Trainings Over Generataions')
# plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
plt.yticks(np.arange(0, 21, 1))



# p1 = plt.plot(ind, [x["accuracy"] for x in generations])
# plt.ylabel('Cross Validation Accuracy')
# plt.title('Performance Over Generataions')
# # plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
# plt.yticks(np.arange(0.91, 0.92, 0.001))

plt.savefig('individuals.png')
plt.show()