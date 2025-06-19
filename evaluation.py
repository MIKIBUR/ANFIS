import matplotlib.pyplot as plt
import numpy as np
import fuzzy_logic
from sklearn.metrics import mean_squared_error

def visualize_gauss_mfs(mfs,input_features):
    """Plot Gaussian membership functions given their parameters. """
    for i, feature_mfs in enumerate(mfs):
        plt.figure(figsize=(8, 4))
        
        # Determine a reasonable x range: from (min mean - 3*sigma) to (max mean + 3*sigma)
        means = [mf[1]['mean'] for mf in feature_mfs]
        sigmas = [mf[1]['sigma'] for mf in feature_mfs]
        x_min = 0
        x_max = 1
        x_range = np.linspace(x_min, x_max, 1000)
        
        for mf in feature_mfs:
            mean = mf[1]['mean']
            sigma = mf[1]['sigma']
            y = fuzzy_logic.gaussmf(x_range, mean, sigma)
            plt.plot(x_range, y, label=f'μ={mean:.2f}, σ={sigma:.2f}')
        
        plt.title(f"Gaussian MFs for Feature {input_features[i]}")
        plt.xlabel("Input value")
        plt.ylabel("Membership degree")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.draw()

def evaluate(y_train,y_test,train_predictions,test_predictions,training_errors):
    # Calculate RMSE
    rmse_train = np.sqrt(mean_squared_error(y_train, train_predictions.flatten()))
    rmse_test = np.sqrt(mean_squared_error(y_test, test_predictions.flatten()))
    
    print(f"Train RMSE: {rmse_train:.4f}")
    print(f"Test RMSE: {rmse_test:.4f}")
    
    # Print sample predictions
    print("\nSample predictions vs actual:")
    for i in range(min(len(test_predictions), 10)):
        pred_val = test_predictions[i][0] if test_predictions.ndim > 1 else test_predictions[i]
        actual_val = y_test[i]
        percent_err = abs(pred_val - actual_val) / actual_val * 100
        print(f"Predicted: {pred_val:.2f}, Actual: {actual_val:.2f}, Percent Err: {percent_err:.2f}%")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Training results
    plt.subplot(2, 1, 1)
    plt.scatter(range(len(train_predictions)), train_predictions, color='red', label='predicted', alpha=0.6)
    plt.scatter(range(len(y_train)), y_train, color='blue', label='actual', alpha=0.6)
    plt.legend()
    plt.title('Training Set Results')
    plt.xlabel('Sample Index')
    plt.ylabel('Credit Score')
    
    # Test results
    plt.subplot(2, 1, 2)
    plt.scatter(range(len(test_predictions)), test_predictions, color='red', label='predicted', alpha=0.6)
    plt.scatter(range(len(y_test)), y_test, color='blue', label='actual', alpha=0.6)
    plt.legend()
    plt.title('Test Set Results')
    plt.xlabel('Sample Index')
    plt.ylabel('Credit Score')
    
    plt.tight_layout()
    plt.show()
