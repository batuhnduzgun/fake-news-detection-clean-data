import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from graphic import plot_learning_curves_plotly, plot_roc_curve_plotly


def create_text_classifier(classifier_type):
    classifiers = {
        "rf": RandomForestClassifier(random_state=42, n_estimators=400),
        "nb": MultinomialNB(),
        "svm": SVC(kernel="linear", probability=True, random_state=42),
        "lr": LogisticRegression(l1_ratio=0, random_state=42),
    }
    return Pipeline(
        [("tfidf", TfidfVectorizer()), ("clf", classifiers[classifier_type])]
    )


def train_and_evaluate(train_df, test_df, classifier_type, cv=5):
    # Create the pipeline
    pipeline = create_text_classifier(classifier_type)

    # Perform cross-validation
    cv_scores = cross_val_score(
        pipeline, train_df["text"], train_df["is_real"], cv=cv, scoring="accuracy"
    )
    print(f"Cross-validation scores ({cv}-fold): {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f}")
    print(f"CV Standard deviation: {cv_scores.std():.4f}")

    # Calculate learning curves
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline,
        train_df["text"],
        train_df["is_real"],
        train_sizes=train_sizes,
        cv=cv,
        n_jobs=-1,
        scoring="accuracy",
    )

    # Plot learning curves
    fig_learning = plot_learning_curves_plotly(train_sizes, train_scores, val_scores)

    # Train on full training set
    pipeline.fit(train_df["text"], train_df["is_real"])

    # Generate and plot ROC curve
    y_score = pipeline.predict_proba(test_df["text"])[:, 1]
    fig_roc = plot_roc_curve_plotly(test_df["is_real"], y_score)

    # Evaluate on training data
    train_predictions = pipeline.predict(train_df["text"])
    train_report = classification_report(
        train_df["is_real"], train_predictions, output_dict=True
    )

    # Evaluate on test data
    test_predictions = pipeline.predict(test_df["text"])
    test_report = classification_report(
        test_df["is_real"], test_predictions, output_dict=True
    )

    return fig_learning, fig_roc, train_report, test_report
