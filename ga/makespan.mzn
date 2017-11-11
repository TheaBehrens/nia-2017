int: num_machines;
int: num_jobs;

array[1..num_jobs] of int: jobs;
array[1..num_jobs] of var 1..num_machines: allocation;
array[1..num_machines] of var int: machine_times;

constraint forall(m in 1..num_machines)(machine_times[m] = sum(j in 1..num_jobs where allocation[j] = m)(jobs[j]));


solve :: int_search([allocation[j] | j in 1..num_jobs], input_order, indomain_min, complete)
   minimize max(machine_times);

output ["total time = ",show(max(fix(machine_times))),";\n"]