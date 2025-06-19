import argparse
import data_loader
import fuzzy_logic
import evaluation
import anfis_core
import time

if __name__ == "__main__":
    """Main execution"""
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Run ANFIS training and evaluation.")
    parser.add_argument("--rows", type=int, default=200, help="Number of rows to load from the dataset")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--decision", type=str, default="h1", help="Decision type, h1-credit_score, h2-loan_int_rate, h3-years_to_pay")
    
    args = parser.parse_args()

    # Define X and y columns
    input_features = ["person_age", "person_income","person_emp_exp","cb_person_cred_hist_length"]
    decision_feature = "credit_score"

    if args.decision == "h1":
        decision_feature = "credit_score"
    elif args.decision == "h2":
        decision_feature = "loan_int_rate"
    elif args.decision == "h3":
        decision_feature = "years_to_pay"

    # Load data
    X_train, X_test, y_train, y_test, scaler = data_loader.load_data(input_features, decision_feature, args.rows)

    # Generate membership functions
    mf_definitions = fuzzy_logic.generate_gauss_mfs(X_train)

    # Visualize membership functions
    evaluation.plot_mfs(mf_definitions,input_features,'Starting MFs definition for features')

    # Train ANFIS
    start_time = time.time()
    trained_mfs, trained_consequents, training_errors = anfis_core.train_anfis(X_train, y_train, mf_definitions, epochs=args.epochs)
    training_duration = time.time() - start_time 

    evaluation.plot_mfs(trained_mfs,input_features,'Trained MFs definition for features')

    # Make predictions
    train_predictions = anfis_core.predict_anfis(X_train, trained_mfs, trained_consequents)
    test_predictions = anfis_core.predict_anfis(X_test, trained_mfs, trained_consequents)

    # Evaluate the trained model
    evaluation.evaluate(y_train, y_test, train_predictions, test_predictions, training_errors, training_duration)

    # Run interactive CLI
    # data_loader.run_cli_interface(trained_mfs, trained_consequents, scaler)
