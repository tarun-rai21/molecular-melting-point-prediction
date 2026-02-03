import pandas as pd


def make_submission(model, X_test, ids, output_path):
    """
    Generates Kaggle submission file.
    """

    # Import predict function from model.py
    from src.model import predict

    preds = predict(model, X_test)

    submission = pd.DataFrame({
        "id": ids,
        "Tm": preds
    })

    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
