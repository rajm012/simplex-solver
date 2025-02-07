# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

def create_variable_names(num_vars, num_constraints):
    """Create variable names for decision and slack variables"""
    decision_vars = [f"x{i+1}" for i in range(num_vars)]
    slack_vars = [f"S{i+1}" for i in range(num_constraints)]
    return decision_vars + slack_vars

def simplex(c, A, b):
    """
    Solves a linear programming problem using the Simplex method.
    Maximizes Z = c^T * x subject to A * x <= b and x >= 0.
    """
    num_constraints, num_vars = A.shape
    
    # Create variable names
    var_names = create_variable_names(num_vars, num_constraints)
    
    # Extend c with zeros for slack variables
    c_ext = np.hstack([c, np.zeros(num_constraints)])

    # Construct initial tableau
    tableau = np.hstack([A, np.eye(num_constraints), b.reshape(-1, 1)])
    tableau = np.vstack([tableau, np.hstack([-c_ext, np.zeros(1)])])

    basic_vars = list(range(num_vars, num_vars + num_constraints))
    Cb = np.zeros(num_constraints)
    
    steps = []
    step_count = 0
    
    while np.any(tableau[-1, :-1] < 0):
        step_count += 1
        
        # Create current basic variable names
        current_basic_vars = [var_names[i] for i in basic_vars]
        
        step_info = {
            "step": int(step_count),
            "tableau": tableau.tolist(),
            "basic_vars": current_basic_vars,
            "var_names": var_names,
            "Cj": c_ext.tolist(),  # Include cost coefficients
            "Cb": Cb.tolist()  # Include basic variables' costs
        }
        
        pivot_col = np.argmin(tableau[-1, :-1])
        ratios = tableau[:-1, -1] / tableau[:-1, pivot_col]
        valid_ratios = np.where(tableau[:-1, pivot_col] > 0, ratios, np.inf)
        pivot_row = np.argmin(valid_ratios)

        if valid_ratios[pivot_row] == np.inf:
            return None, None, ["Unbounded solution detected."]

        pivot_element = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_element
        for i in range(len(tableau)):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

        basic_vars[pivot_row] = pivot_col
        Cb[pivot_row] = c_ext[pivot_col]

        steps.append(step_info)

    # Final step
    final_step = {
        "step": step_count + 1,
        "tableau": tableau.tolist(),
        "basic_vars": [var_names[i] for i in basic_vars],
        "var_names": var_names,
        "Cj": c_ext.tolist(),
        "Cb": Cb.tolist()
    }
    steps.append(final_step)

    # Final solution with variable names
    solution_dict = {}
    for i, var in enumerate(basic_vars):
        if var < num_vars:
            solution_dict[var_names[var]] = float(tableau[i, -1])
    
    # Add zeros for non-basic variables
    for i in range(num_vars):
        if f"x{i+1}" not in solution_dict:
            solution_dict[f"x{i+1}"] = 0.0

    return solution_dict, float(tableau[-1, -1]), steps

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    try:
        data = request.get_json()
        
        # Parse input data
        c = np.array(data['objective'], dtype=float)
        A = np.array(data['constraints'], dtype=float)
        b = np.array(data['rhs'], dtype=float)
        
        # Solve using simplex method
        solution, optimal_value, steps = simplex(c, A, b)
        
        if solution is None:
            return jsonify({'error': 'Unbounded solution'})
        
        result = {
            'solution': solution,
            'optimal_value': optimal_value,
            'steps': steps
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)