import matplotlib.pyplot as plt
import numpy as np
import fuzzy_logic
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def plot_mfs(mfs, input_features, plot_label):
    """Plot Gaussian membership functions for all features"""
    x_range = np.linspace(0, 1, 1000)
    num_features = len(mfs)

    # Grid dimensions
    cols = 2
    rows = (num_features + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 5))
    axes = axes.flatten()

    for i, feature_mfs in enumerate(mfs):
        ax = axes[i]
        for mf in feature_mfs:
            mean = mf[1]['mean']
            sigma = mf[1]['sigma']
            y = fuzzy_logic.gaussmf(x_range, mean, sigma)
            ax.plot(x_range, y, label=f'μ={mean:.2f}, σ={sigma:.2f}')
        
        fig.suptitle(plot_label, fontsize=16)
        fig.supxlabel("Input Value", fontsize=12)
        ax.set_title(f"Gaussian MFs for {input_features[i]}")
        ax.set_ylabel("Membership degree")
        ax.legend()
        ax.grid(True)

    # Hide unused subplots
    for i in range(num_features, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.draw()

def plot_training_errors(errors):
    """Plot training errors per epoch"""
    plt.figure(figsize=(6, 4))
    plt.plot(range(len(errors)), errors, 'ro-', label='Training Error')
    plt.ylabel('Error')
    plt.xlabel('Epoch')
    plt.title('Training MSE over Epochs')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.draw()


def evaluate(y_train, y_test, train_predictions, test_predictions, training_errors, training_duration):
    """General model evaluation"""
    # Flatten predictions if needed
    train_predictions = train_predictions.flatten()
    test_predictions = test_predictions.flatten()

    # Calculate error metrics
    rmse_train = np.sqrt(mean_squared_error(y_train, train_predictions))
    mae_train = mean_absolute_error(y_train, train_predictions)
    r2_train = r2_score(y_train, train_predictions)

    rmse_test = np.sqrt(mean_squared_error(y_test, test_predictions))
    mae_test = mean_absolute_error(y_test, test_predictions)
    r2_test = r2_score(y_test, test_predictions)

    # Print metrics
    print("")
    print("Evaluation Metrics:")
    print(f"Training duration: {training_duration:.4f}")
    print("")
    print(f"Train RMSE: {rmse_train:.4f}")
    print(f"Test RMSE: {rmse_test:.4f}")
    print("")
    print(f"Train MAE: {mae_train:.4f}")
    print(f"Test MAE: {mae_test:.4f}")
    print("")
    print(f"Train R² Score: {r2_train:.4f}")
    print(f"Test R² Score: {r2_test:.4f}")

    # Print sample predictions
    print("\nSample predictions vs actual:")
    for i in range(min(len(test_predictions), 10)):
        pred_val = test_predictions[i]
        actual_val = y_test[i]
        percent_err = abs(pred_val - actual_val) / (abs(actual_val) + 1e-8) * 100  # Avoid div by zero
        print(f"Predicted: {pred_val:.2f}, Actual: {actual_val:.2f}, Percent Err: {percent_err:.2f}%")

    # Plot prediction results
    plt.figure(figsize=(14, 10))

    # 1. Training Predictions
    plt.subplot(3, 1, 1)
    plt.scatter(range(len(train_predictions)), train_predictions, color='red', label='Predicted', alpha=0.6)
    plt.scatter(range(len(y_train)), y_train, color='blue', label='Actual', alpha=0.6)
    plt.title('Training Set Predictions')
    plt.xlabel('Sample Index')
    plt.ylabel('Target Value')
    plt.legend()

    # 2. Test Predictions
    plt.subplot(3, 1, 2)
    plt.scatter(range(len(test_predictions)), test_predictions, color='red', label='Predicted', alpha=0.6)
    plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual', alpha=0.6)
    plt.title('Test Set Predictions')
    plt.xlabel('Sample Index')
    plt.ylabel('Target Value')
    plt.legend()

    # 3. Error Metrics Bar Chart
    metrics = ['RMSE', 'MAE', 'R²']
    train_scores = [rmse_train, mae_train, r2_train]
    test_scores = [rmse_test, mae_test, r2_test]

    x = np.arange(len(metrics))
    width = 0.35

    plt.subplot(3, 1, 3)
    plt.bar(x - width/2, train_scores, width, label='Train', color='skyblue')
    plt.bar(x + width/2, test_scores, width, label='Test', color='salmon')
    plt.xticks(x, metrics)
    plt.ylabel('Score')
    plt.title('Error Metrics Comparison')
    plt.legend()

    plt.tight_layout()
    plt.draw()

    # Plot training errors
    plot_training_errors(training_errors)

    plt.show()
