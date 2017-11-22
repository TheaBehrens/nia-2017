include "globals.mzn";

int: num_nodes;
set of int: N = 1..num_nodes;
array[N,N] of int: cost_matrix;
array[N] of var N: route; % route will be a circuit (see constraints)

% restrict range of possible costs:
int: min_cost = min(i,j in N where cost_matrix[i,j] > 0)(cost_matrix[i,j]);
int: max_cost = max(i,j in N)(cost_matrix[i,j]);
% array of costs for each leg
array[N] of var min_cost..max_cost: route_costs;

var int: total_cost = sum(route_costs);

constraint alldifferent(route);
constraint circuit(route); % route[i] = j means j is the successor of i in the circuit

% the cost for a leg can be looked up with the index in the route and the value at that index
constraint forall(i in N)(cost_matrix[i, route[i]] = route_costs[i]);

solve :: int_search(route_costs, max_regret, indomain_split, complete) minimize(total_cost);     
%solve minimize(total_cost);

output [show(total_cost)]         