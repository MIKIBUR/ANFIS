import pandas as pd
import numpy as np
import itertools
import fuzzy_logic
from sklearn.metrics import mean_squared_error

def forward_half_pass(X, mem_funcs, rules):
    """Forward pass through ANFIS layers 1-4"""
    layer_four = np.empty(0,)
    w_sum = []
    
    # Process each input pattern
    for pattern in range(len(X[:,0])):
        
        # Layer 1: Evaluate membership functions for current input pattern
        layer_one = fuzzy_logic.evaluate_mf(X[pattern,:], mem_funcs)
        
        # Layer 2: Calculate rule firing strengths
        # For each rule, get the membership values for each input variable
        mi_alloc = []
        for row in range(len(rules)):
            rule_memberships = []
            for x in range(len(rules[0])):
                # Get membership value for variable x using the membership function specified in the rule
                membership_value = layer_one[x][rules[row][x]]
                rule_memberships.append(membership_value)
            mi_alloc.append(rule_memberships)
        
        # Calculate firing strength for each rule (product of all membership values)
        layer_two_list = []
        for rule_memberships in mi_alloc:
            firing_strength = np.prod(rule_memberships)
            layer_two_list.append(firing_strength)
        layer_two = np.array(layer_two_list).T
        
        # Store firing strengths for all patterns
        if pattern == 0:
            w = layer_two
        else:
            w = np.vstack((w, layer_two))
        
        # Layer 3: Calculate normalized firing strengths
        # Sum all firing strengths for current pattern
        current_w_sum = np.sum(layer_two)
        w_sum.append(current_w_sum)
        
        # Normalize firing strengths by dividing by their sum
        normalized_strengths = layer_two / current_w_sum
        
        if pattern == 0:
            w_normalized = normalized_strengths
        else:
            w_normalized = np.vstack((w_normalized, normalized_strengths))
        
        # Layer 4: Prepare weighted linear combinations
        layer_three = layer_two / current_w_sum
        
        # For each normalized firing strength, multiply by extended input vector [x1, x2, ..., xn, 1]
        extended_input = np.append(X[pattern,:], 1)  # Add bias term
        weighted_combinations = []
        
        for strength in layer_three:
            weighted_combination = strength * extended_input
            weighted_combinations.append(weighted_combination)
        
        # Concatenate all weighted combinations for this pattern
        row_holder = np.concatenate(weighted_combinations)
        layer_four = np.append(layer_four, row_holder)
    
    # Reshape matrices for proper output format
    w = w.T
    w_normalized = w_normalized.T
    layer_four = np.array(np.array_split(layer_four, pattern + 1))
    
    return layer_four, w_sum, w


def lse(A, B, initial_gamma):
    """Least squares estimation for consequent parameters using recursive least squares"""
    coeff_mat = A
    rhs_mat = B
    
    # Initialize covariance matrix S and parameter vector x
    num_params = coeff_mat.shape[1]
    S = np.eye(num_params) * initial_gamma
    x = np.zeros((num_params, 1))
    
    # Process each data point recursively
    for i in range(len(coeff_mat[:,0])):
        # Get current coefficient vector and target value
        a = coeff_mat[i,:]
        b = np.array(rhs_mat[i])
        
        # Convert to matrix format for calculations
        a_matrix = np.matrix(a).transpose()
        a_row_matrix = np.matrix(a)
        b_matrix = np.matrix(b)
        
        # Calculate intermediate terms for covariance update
        S_a = np.dot(S, a_matrix)
        a_S = np.dot(a_row_matrix, S)
        a_S_a = np.dot(a_S, a_matrix)
        denominator = 1 + a_S_a
        
        # Update covariance matrix S
        numerator = np.dot(S_a, a_S)
        S = S - (numerator / denominator)
        
        # Calculate prediction error
        prediction = np.dot(a_row_matrix, x)
        error = b_matrix - prediction
        
        # Update parameter vector x
        correction = np.dot(S, np.dot(a_matrix, error))
        x = x + correction
    
    return x


def backprop(X, Y, mem_funcs, rules, consequents, column_x, columns, the_w_sum, the_w, the_layer_five):
    """Backpropagation for updating membership function parameters"""

    # Initialize parameter gradients for current variable
    param_grp = [0] * len(mem_funcs[column_x])
    
    # Process each membership function for current variable
    for MF in range(len(mem_funcs[column_x])):
        # Initialize parameter array for current membership function
        num_params = len(mem_funcs[column_x][MF][1])
        parameters = np.empty(num_params)
        times_thru = 0
        
        # Process each parameter of current membership function
        param_keys = sorted(mem_funcs[column_x][MF][1].keys())
        for alpha in param_keys:
            # Initialize gradient accumulator for current parameter
            bucket3 = np.empty(len(X))
            
            # Process each training pattern
            for row_x in range(len(X)):
                # Get input value for current variable and pattern
                var_to_test = X[row_x, column_x]
                
                # Create temporary row filled with current variable value
                tmp_row = np.empty(len(mem_funcs))
                tmp_row.fill(var_to_test)
                
                # Initialize gradient accumulator for current pattern
                bucket2 = np.empty(Y.ndim)
                
                # Process each output dimension
                for col_y in range(Y.ndim):
                    # Find rules that use current membership function
                    rules_using_current_mf = []
                    for rule_idx in range(len(rules)):
                        if rules[rule_idx, column_x] == MF:
                            rules_using_current_mf.append(rule_idx)
                    rules_with_alpha = np.array(rules_using_current_mf)
                    
                    # Get columns excluding current variable
                    adj_cols = np.delete(columns, column_x)
                    
                    # Calculate partial derivative of membership function
                    sen_sit = fuzzy_logic.partial_dmf(X[row_x, column_x], mem_funcs[column_x][MF], alpha)

                    # Calculate dW/dAlpha (derivative of firing strength w.r.t. parameter alpha)
                    dw_dalpha_list = []
                    for r in rules_with_alpha:
                        # Calculate product of membership values for other variables
                        product_other_vars = 1.0
                        for c in adj_cols:
                            membership_val = fuzzy_logic.evaluate_mf(tmp_row, mem_funcs)[c][rules[r][c]]
                            product_other_vars *= membership_val
                        
                        dw_contribution = sen_sit * product_other_vars
                        dw_dalpha_list.append(dw_contribution)
                    
                    dw_dalpha = np.array(dw_dalpha_list)
                    
                    # Initialize gradient accumulator for current output dimension
                    bucket1 = np.empty(len(rules[:,0]))
                    
                    # Process each rule (consequent)
                    for consequent in range(len(rules[:,0])):
                        # Calculate consequent output for current rule
                        extended_input = np.append(X[row_x,:], 1.)
                        start_idx = (X.shape[1] + 1) * consequent
                        end_idx = start_idx + (X.shape[1] + 1)
                        consequent_params = consequents[start_idx:end_idx, col_y]
                        f_consequent = np.dot(extended_input, consequent_params)
                        
                        # Calculate contribution to gradient
                        acum = 0
                        
                        # Check if current rule uses the membership function being updated
                        if consequent in rules_with_alpha:
                            rule_position = np.where(rules_with_alpha == consequent)[0]
                            if len(rule_position) > 0:
                                acum = dw_dalpha[rule_position[0]] * the_w_sum[row_x]
                        
                        # Subtract normalized contribution
                        acum = acum - the_w[consequent, row_x] * np.sum(dw_dalpha)
                        
                        # Normalize by squared sum of firing strengths
                        acum = acum / (the_w_sum[row_x] ** 2)
                        
                        # Multiply by consequent output
                        bucket1[consequent] = f_consequent * acum
                    
                    # Sum contributions from all rules
                    sum1 = np.sum(bucket1)
                    
                    # Calculate final gradient contribution for current output dimension
                    if Y.ndim == 1:
                        error = Y[row_x] - the_layer_five[row_x]
                    else:
                        error = Y[row_x, col_y] - the_layer_five[row_x, col_y]
                    
                    bucket2[col_y] = sum1 * error * (-2)  # Factor of -2 from squared error derivative
                
                # Sum contributions from all output dimensions
                sum2 = np.sum(bucket2)
                bucket3[row_x] = sum2
            
            # Sum contributions from all training patterns
            sum3 = np.sum(bucket3)
            parameters[times_thru] = sum3
            times_thru = times_thru + 1
        
        # Store gradients for current membership function
        param_grp[MF] = parameters
    
    return param_grp


def train_anfis(X, Y, mem_funcs, y_scaler, epochs=5, tolerance=1e-5, initial_gamma=300, k=0.05):
    """Train ANFIS using hybrid learning algorithm (LSE + backpropagation)"""

    # Initialize membership function indices for rule generation
    mem_funcs_by_variable = [
        list(range(len(mem_funcs[z])))
        for z in range(len(mem_funcs))
    ]
    
    # Generate all possible rules (Cartesian product of membership function indices)
    rules = np.array(list(itertools.product(*mem_funcs_by_variable)))
    
    # Initialize consequent parameters
    num_consequent_params = Y.ndim * len(rules) * (X.shape[1] + 1)
    consequents = np.zeros(num_consequent_params)
    
    # Initialize training variables
    errors = []
    
    # Check if all variables have the same number of membership functions
    mem_funcs_homo = all(len(mf) == len(mem_funcs[0]) for mf in mem_funcs_by_variable)
    
    convergence = False
    epoch = 1

    while (epoch <= epochs) and (not convergence):
        # Forward pass
        layer_four, w_sum, w = forward_half_pass(X, mem_funcs, rules)
        
        # LSE to update consequent parameters
        layer_five = np.array(lse(layer_four, Y, initial_gamma))
        consequents = layer_five
        
        # Network output (normalized)
        y_pred_norm = np.dot(layer_four, layer_five).ravel()

        # Inverse scale both prediction and ground truth
        y_pred_real = y_scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).ravel()
        y_true_real = y_scaler.inverse_transform(Y.reshape(-1, 1)).ravel()

        # Calculate error on original scale
        error = np.sqrt(mean_squared_error(y_true_real, y_pred_real))
        errors.append(error)

        # Display progress
        dots = ['.  ', '.. ', '...'][epoch % 3]
        print(f"\rCurrent epoch: {epoch}/{epochs} {dots}", end='', flush=True)

        # Convergence check
        if error < tolerance:
            convergence = True

        if not convergence:
            # Backpropagation
            de_dalpha = [
                backprop(X, Y, mem_funcs, rules, consequents, col_x, range(X.shape[1]), w_sum, w, y_pred_norm)
                for col_x in range(X.shape[1])
            ]

            # Adaptive learning rate
            if len(errors) >= 4 and all(errors[i] > errors[i + 1] for i in range(-4, -1)):
                k *= 1.1
            if len(errors) >= 5 and (errors[-1] < errors[-2] and errors[-3] < errors[-2] and
                                     errors[-3] < errors[-4] and errors[-5] > errors[-4]):
                k *= 0.9

            # Compute adaptive eta
            all_gradients = [g for g_layer in de_dalpha for g_mf in g_layer for g in g_mf]
            total_grad_mag = np.abs(np.sum(all_gradients))
            eta = k / total_grad_mag if total_grad_mag != 0 and not np.isinf(total_grad_mag) else k

            # Compute parameter updates
            if not mem_funcs_homo:
                for i in range(len(mem_funcs)):
                    for j in range(len(mem_funcs[i])):
                        for k_idx, param_key in enumerate(sorted(mem_funcs[i][j][1])):
                            update = -eta * de_dalpha[i][j][k_idx]
                            mem_funcs[i][j][1][param_key] += update
            else:
                d_alpha = -eta * np.array(de_dalpha)
                for i in range(len(mem_funcs)):
                    for j in range(len(mem_funcs[i])):
                        for k_idx, param_key in enumerate(sorted(mem_funcs[i][j][1])):
                            mem_funcs[i][j][1][param_key] += d_alpha[i][j][k_idx]

        epoch += 1

    print()  # New line
    return mem_funcs, consequents, np.array(errors)

def predict(X, mem_funcs, consequents, y_scaler):
    """Make predictions using trained ANFIS model and rescale outputs to original target scale."""
    # Recreate membership function indices for rule generation
    mem_funcs_by_variable = [
        list(range(len(mem_funcs[z])))
        for z in range(len(mem_funcs))
    ]
    
    # Regenerate rules (same as in training)
    rules = np.array(list(itertools.product(*mem_funcs_by_variable)))
    
    # Forward pass to get predictions
    layer_four, _, _ = forward_half_pass(X, mem_funcs, rules)
    
    # Calculate final normalized output
    layer_five = np.dot(layer_four, consequents)

    # Rescale output to original target range
    rescaled_output = y_scaler.inverse_transform(layer_five.reshape(-1, 1)).ravel()

    return rescaled_output
