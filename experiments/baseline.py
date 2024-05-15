import gurobipy as gp
from gurobipy import GRB
import numpy as np
from itertools import chain, combinations
import matplotlib.pyplot as plt
from utils.logger_config import setup_logger

logger = setup_logger()


# Function to generate the power set of features
def powerset(iterable):
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))


# Define f(z_i) as the prediction from a trained model
def f(model, z_i):
    return model.predict_proba(z_i.reshape(1, -1))[0]


class OptimalMeanDifference:

    def __init__(self, model, X_factual, X_counterfactual):
        self.model = model
        self.X_factual = X_factual
        self.X_counterfactual = X_counterfactual

        assert X_factual.shape == X_counterfactual.shape
        self.n, self.p = X_factual.shape

        # Generate the power set and create a mapping
        feature_indices = range(self.p)
        self.subsets = powerset(feature_indices)
        self.num_subsets = len(self.subsets)

        # Compute sizes of each subset
        self.s = [len(subset) for subset in self.subsets]

        self.x, self.r = X_factual, X_counterfactual

        self.g = self._compute_g()

    def _compute_g(self):
        # Compute g_{i, \bb} for each data point and each subset
        g = np.zeros((self.n, self.num_subsets))
        for i in range(self.n):
            r_i = self.r[i]
            f_r_i = f(self.model, r_i)
            for j, subset in enumerate(self.subsets):
                z_i = self.x[i].copy()  # Start with the original row
                for k in subset:
                    z_i[k] = self.r[i][
                        k
                    ]  # Replace values in subset with values from r_i
                f_z_i = f(self.model, z_i)
                g[i, j] = f_z_i - f_r_i  # Compute g_{i, \bb}
        g /= self.n
        return g

    def solve_problem(self, C):
        # Create the model
        self.problem = gp.Model("subset_selection")

        # Set the model to be silent
        self.problem.setParam("OutputFlag", 0)

        # Decision variables
        self.a = self.problem.addVars(
            self.n, self.num_subsets, vtype=GRB.BINARY, name="a"
        )

        # Variable for the objective
        self.eta = self.problem.addVar(vtype=GRB.CONTINUOUS, name="eta")

        # Objective function
        self.problem.setObjective(self.eta, GRB.MINIMIZE)

        # Absolute value constraint
        self.problem.addConstr(
            gp.quicksum(
                self.g[i, j] * self.a[i, j]
                for i in range(self.n)
                for j in range(self.num_subsets)
            )
            <= self.eta,
            "abs_pos",
        )
        self.problem.addConstr(
            gp.quicksum(
                self.g[i, j] * self.a[i, j]
                for i in range(self.n)
                for j in range(self.num_subsets)
            )
            >= -self.eta,
            "abs_neg",
        )

        # Coverage constraint: Ensure at least one subset is selected for each data point
        self.problem.addConstrs(
            (
                gp.quicksum(self.a[i, j] for j in range(self.num_subsets)) == 1
                for i in range(self.n)
            ),
            "coverage",
        )

        # Capacity constraint: Ensure the total size of selected subsets does not exceed C across all data points
        self.problem.addConstr(
            gp.quicksum(
                self.s[j] * self.a[i, j]
                for i in range(self.n)
                for j in range(self.num_subsets)
            )
            <= C,
            "capacity",
        )

        # Optimize the problem
        self.problem.optimize()

        # Extract the solution for the decision variables a
        self.a_solution = np.zeros((self.n, self.num_subsets))

        for i in range(self.n):
            for j in range(self.num_subsets):
                if (
                    self.a[i, j].X > 0.5
                ):  # Gurobi returns the value of the variable, check if it's selected
                    self.a_solution[i, j] = 1

        return {"eta": self.eta.X}

    def display_g(self, colorbar=False):
        plt.imshow(self.g, cmap="viridis")
        if colorbar:
            plt.colorbar()
        plt.xlabel("Subset Index")
        plt.ylabel("Data Point Index")
        plt.title("Heatmap of g")
        plt.show()

    def display_optimal_solutions(self):
        # Print the results
        for v in self.problem.getVars():
            if (
                v.varName != "eta" and v.x > 0.001
            ):  # Only print the variables that are part of the solution
                var_name_parts = v.varName.split("[")[1].rstrip("]").split(",")
                i = int(var_name_parts[0])
                j = int(var_name_parts[1])
                subset = self.subsets[j]
                logger.info(f"Data point {i} uses subset {subset}")

        logger.info(f"Objective Value (eta): {self.eta.X}")

        # Calculate the value of the constraint expression
        total_size = sum(
            self.s[j] * self.a[i, j].X
            for i in range(self.n)
            for j in range(self.num_subsets)
        )
        logger.info(f"Total consumed size = {total_size}")

    def display_a(self, colorbar=False):
        # Visualize the matrix a_solution using a heatmap
        plt.imshow(
            self.a_solution,
            cmap="gray",
        )
        if colorbar:
            plt.colorbar()
        plt.xlabel("Subset Index")
        plt.ylabel("Data Point Index")
        plt.title("Heatmap of Decision Variables a")
        plt.show()

    def compute_intervention_policy(self):
        return self.a_solution / self.a_solution.sum()
