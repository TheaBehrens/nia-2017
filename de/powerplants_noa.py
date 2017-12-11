import numpy as np

# characteristics of the problem
PLANT_KWH = [50000, 600000, 4000000]
PLANT_COSTS = [10000, 80000, 400000]
MAX_PLANTS = [100, 50, 3]


MAX_PRICE = [0.45, 0.25, 0.2]
MAX_DEMAND = [2000000, 30000000, 20000000]

COST_PRICE = 0.6
LARGE = np.inf


def objective_func(solution):
    """Takes a solution vector of the form (e1, e2, e3, s1, s2, s3, p1,
    p2, p3) and determines the profit.

    """
    produce = solution[0:3]  # e1, e2, e3
    sell = solution[3:6]     # s1, s2, s3
    price = solution[6:9]    # p1, p2, p3
    # ensure validity:
    if (np.array(solution) < 0.0).any():
        return -np.inf
    revenue = 0
    prod_cost = 0
    purch_cost = 0
    for plant_type, amount in enumerate(produce):
        prod_cost += cost_func(plant_type, amount)
    for mkt_type, amount in enumerate(sell):
        # revenue : soldQuantity * price
        revenue += (min(demand_func(mkt_type, price[mkt_type]), amount) *
                    price[mkt_type])
    purch_cost = max((sum(sell) - sum(produce)), 0) * COST_PRICE
    cost = prod_cost + purch_cost
    profit = revenue - cost
    return profit


def cost_func(plant_type, amount):
    """Calculates the cost.
    """
    if amount <= 0:
        return 0

    elif amount > PLANT_KWH[plant_type] * MAX_PLANTS[plant_type]:
        return LARGE

    return np.ceil(amount / PLANT_KWH[plant_type]) * PLANT_COSTS[plant_type]


def demand_func(mkt_type, price):
    """Calculates the demand.
    """

    if price > MAX_PRICE[mkt_type]:
        return 0

    elif price <= 0:
        return MAX_DEMAND[mkt_type]

    return (MAX_DEMAND[mkt_type] - (np.square(price) *
                                    MAX_DEMAND[mkt_type] /
                                    np.square(MAX_PRICE[mkt_type])))


def get_bounds():
    bounds = np.array([[0, 10000000],  # e1
                       [0, 10000000],  # e2
                       [0, 10000000],  # e3
                       [0, 10000000],  # s1
                       [0, 10000000],  # s2
                       [0, 10000000],  # s3
                       [0, 0.7],  # p1
                       [0, 0.7],  # p2
                       [0, 0.7]])  # p3
    return bounds
