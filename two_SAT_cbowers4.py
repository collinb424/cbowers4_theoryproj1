import csv, time
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def unit_propagate(cnf, assignments):
    """
    Performs unit propagation on the cnf with the current assignments.
    """

    unit_clauses = [clause for clause in cnf if len(clause) == 1]

    while unit_clauses:
        # Get the literal in the unit clause
        # If it was assigned a different value earlier, then there is a conflict
        literal = unit_clauses.pop()[0]
        var = abs(literal)
        val = True if literal > 0 else False
        if var in assignments:
            if assignments[var] != val:
                # Conflict
                return False, None
        
        assignments[var] = val
        new_clauses = []
        for clause in cnf:
            if literal in clause: 
                # Do not include because the literal in the unit clause is in it
                continue
            elif -literal in clause:
                new_clause = [l for l in clause if l != -literal]

                if not new_clause:
                    # Empty clause generated, unsatisfiable
                    return False, None
                
                # New unit clause (this will result in the outer while loop continuing)
                if len(new_clause) == 1 and new_clause not in unit_clauses:
                    unit_clauses.append(new_clause)

                new_clauses.append(new_clause)
            else:
                new_clauses.append(clause)

        cnf = new_clauses

    return cnf, assignments


def pure_literal_assign(cnf, assignments):
    """
    Performs pure literal elimination on the clauses with the current assignments.
    """
    all_literals = [literal for clause in cnf for literal in clause]

    literal_counts = {}
    for literal in all_literals:
        literal_counts[literal] = literal_counts.get(literal, 0) + 1

    # Gather pure literals
    pure_literals = []
    for literal in literal_counts:
        if -literal not in literal_counts:
            pure_literals.append(literal)

    # No pure literals means there is nothing else to do
    if not pure_literals:
        return cnf, assignments
    
    # If the same variable but with the opposite polarity is found to be a pure literal, 
    # and it was assigned a different value earlier, then there is a conflict
    for literal in pure_literals:
        var = abs(literal)
        val = True if literal > 0 else False
        if var in assignments:
            if assignments[var] != val:
                # Conflict detected
                return False, None
        else:
            assignments[var] = val

    # Remove clauses containing pure literals
    new_clauses = []
    for clause in cnf:
        # Check if the clause contains any of the pure literals
        contains_pure_literal = False
        for literal in pure_literals:
            if literal in clause:
                contains_pure_literal = True
                break

        # Only include the clause if it does not contain a pure literal
        if not contains_pure_literal:
            new_clauses.append(clause)

    return new_clauses, assignments

def substitute(cnf, literal):
    """
    Substitutes a variable assignment into the cnf.
    """
    new_cnf = []
    for clause in cnf:
        if literal in clause:
            continue  # Clause is satisfied
        elif -literal in clause:
            new_clause = [l for l in clause if l != -literal]
            if not new_clause:
                return False  # Empty clause generated, so there's a conflict
            new_cnf.append(new_clause)
        else:
            new_cnf.append(clause)

    return new_cnf


def has_empty_clause(cnf):
    return any(len(clause) == 0 for clause in cnf)


def dpll(cnf, variables, assignments):
    """
    The main DPLL recursive function that attempts to find a satisfying assignment.
    """

    # Do Unit propagation and Pure literal elimination
    # If clauses is False, then we know we reached a dead end and have to backtrack
    cnf, assignments = unit_propagate(cnf, assignments)
    if cnf is False: # use the "is" operator to be more specific
        return False, assignments
    cnf, assignments = pure_literal_assign(cnf, assignments)
    if cnf is False:
        return False, assignments
    
    # All clauses satisfied
    if len(cnf) == 0:
        return True, assignments
    # If there is an empty clause, it is unsatisfiable
    if has_empty_clause(cnf):
        return False, assignments
        
    # Check if all variables are assigned
    assigned_vars = set(assignments.keys())
    if assigned_vars == variables:
        return True, assignments
    
    # Choose a variable to assign
    unassigned_vars = variables - assigned_vars
    var = unassigned_vars.pop()

    # Try assigning True
    assignments_true = assignments.copy()
    assignments_true[var] = True
    cnf_true = substitute(cnf, var)
    if cnf_true is not False:
        result, final_assignments = dpll(cnf_true, variables, assignments_true)
        if result:
            return True, final_assignments
        
    # Try assigning False
    assignments_false = assignments.copy()
    assignments_false[var] = False
    cnf_false = substitute(cnf, -var)
    if cnf_false is not False:
        return dpll(cnf_false, variables, assignments_false)
    else:
        return False, assignments
    

def run_and_time(cnf, variables, assignments):
    """
    Run the DPLL algorithm and measure the time taken.
    """
    start_time = time.time()
    result, final_assignments = dpll(cnf, variables, assignments)
    end_time = time.time()
    exec_time = (end_time - start_time) * 1e6  # Convert to microseconds
    return result, exec_time, final_assignments


def parse_clause(line):
    """Parses the given line into a list of literals representing a clause."""
    clause = []
    i = 0
    while (line[i] != '0'):
        clause.append(int(line[i]))
        i += 1
    return clause


def read_data(file_name="check_cbowers4.csv"):
    """Reads the CNF problem data from a CSV file and processes each problem."""
    with open(file_name, mode='r', encoding='utf-8-sig') as file:
        csv_file = csv.reader(file)
        cnf = []
        num_vars, num_clauses = 0, 0
        results = []
        problem_number = 1

        for line in csv_file:
            if line[0] == 'c':
                continue
            elif line[0] == 'p':
                # Problem line with number of variables and clauses
                num_vars, num_clauses = int(line[2]), int(line[3])
                continue

            # Parse the line representing a clause
            clause = parse_clause(line)
            cnf.append(clause)

            # When we have gathered all clauses for the current problem, run DPLL
            if len(cnf) == num_clauses:
                variables = set(abs(literal) for clause in cnf for literal in clause)
                assignments = {}
                result, exec_time, final_assignments = run_and_time(cnf, variables, assignments)
                results.append((num_vars, exec_time, result))
                status = "Satisfiable" if result else "Unsatisfiable"
                output = f"Problem {problem_number}: {status}, Time: {exec_time:.2f} µs"
                if result:
                    output += f", Assignment: {final_assignments}"
                print(output)
                
                cnf = []  # Reset for the next problem
                problem_number += 1
                cnf = []

    return results

def plot_results(results):
    """
    Plots execution times against the number of variables with different markers for satisfiable and unsatisfiable results.
    """
    # Get data points for plotting
    num_vars = np.array([r[0] for r in results])
    time_per_literal = np.array([r[1] for r in results])
    satisfiable = np.array([r[2] for r in results])

    # Split data into satisfiable and unsatisfiable
    satisfiable_vars = num_vars[satisfiable]
    satisfiable_times = time_per_literal[satisfiable]

    unsatisfiable_vars = num_vars[~satisfiable]
    unsatisfiable_times = time_per_literal[~satisfiable]

    # Plotting the data
    plt.figure(figsize=(10, 6))
    plt.scatter(satisfiable_vars, satisfiable_times, color='green', label='Satisfiable', alpha=0.8)
    plt.scatter(unsatisfiable_vars, unsatisfiable_times, color='red', marker='^', label='Unsatisfiable', alpha=0.8)

    # Fit an exponential curve to the unsatisfiable cases for the dashed line
    def exponential(x, a, b):
        return a * np.exp(b * x)

    # Fit to unsatisfiable data points
    params, _ = curve_fit(exponential, unsatisfiable_vars, unsatisfiable_times)
    x_fit = np.linspace(min(num_vars), max(num_vars), 100)
    y_fit = exponential(x_fit, *params)

    # Plot the fitted curve
    plt.plot(x_fit, y_fit, 'k--', label=f'Fit: {params[0]:.2f} * exp({params[1]:.4f} * x)')

    # Set the axis scales and labels
    plt.xscale('log')  # Log scale for better visualization
    plt.xlabel('Number of Variables')
    plt.ylabel('Time (microseconds)')
    plt.title('DPLL: Problem Size vs Time')
    plt.grid(True, which="both", ls="--")
    plt.legend()

    plt.show()


results = read_data()
plot_results(results)
