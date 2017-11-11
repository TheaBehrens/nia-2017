import random

num_machines1 = 20
jobs1 = [random.randint(10, 1000) for i in range(0, 200)] +\
        [random.randint(100, 300) for i in range(0, 100)]
with open('makespan_1.dzn', 'w') as data:
    data.write('num_jobs = ' + str(len(jobs1)) + ";\n" +
               'num_machines = ' + str(num_machines1) + ";\n" +
               'jobs = ' + str(jobs1) + ";\n")

num_machines2 = 20
jobs2 = [random.randint(10, 1000) for i in range(0, 150)] +\
        [random.randint(400, 700) for i in range(0, 150)]
with open('makespan_2.dzn', 'w') as data:
    data.write('num_jobs = ' + str(len(jobs2)) + ";\n" +
               'num_machines = ' + str(num_machines2) + ";\n" +
               'jobs = ' + str(jobs2) + ";\n")

num_machines3 = 50
time = 50
jobs3 = [time for i in range(3)]
while len(jobs3) < 100:
        time += 1
        jobs3 += [time for i in range(2)]
with open('makespan_3.dzn', 'w') as data:
    data.write('num_jobs = ' + str(len(jobs3)) + ";\n" +
               'num_machines = ' + str(num_machines3) + ";\n" +
               'jobs = ' + str(jobs3) + ";\n")