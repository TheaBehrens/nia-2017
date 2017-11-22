import csv


def read_tsp(problem):
    tsp = []
    with open(str(problem) + '.tsp') as inputfile:
        for line in csv.reader(inputfile, delimiter=' '):
            line.remove('')
            tsp.append([int(i) for i in line])
    return tsp

def convert_to_dzn(problem):
    tsp = read_tsp(problem)
    with open(str(problem) + '.dzn', 'w') as data:
        data.write('num_nodes = ' + str(len(tsp)) + ';\n')
        data.write('cost_matrix = [')
        for i in tsp:
            line = str(i).replace('[','|').replace(']','')
            data.write(line + ',\n')
        data.write('|]')
