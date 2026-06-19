import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import auc, confusion_matrix, roc_curve


def plot_classification_report(title, train_report_dict, test_report_dict):
    def process_report(report_dict, report_type):
        class_data = []
        metrics_to_plot = ["precision", "recall", "f1-score"]

        for class_name, metrics in report_dict.items():
            if isinstance(metrics, dict) and class_name not in [
                "macro avg",
                "weighted avg",
            ]:
                class_data.append(
                    {
                        "class": class_name,
                        "report_type": report_type,
                        **{metric: metrics[metric] for metric in metrics_to_plot},
                    }
                )
        return class_data

    train_data = process_report(train_report_dict, "Train")
    test_data = process_report(test_report_dict, "Test")
    df = pd.DataFrame(train_data + test_data)

    fig = go.Figure()
    colors = {"precision": "#4363d8", "recall": "#e6194B", "f1-score": "#3cb44b"}
    patterns = {"Train": "", "Test": "/"}

    # Plot metrics interleaved
    for metric, color in colors.items():
        for report_type in ["Train", "Test"]:
            mask = (df["report_type"] == report_type) & df[metric].notna()
            fig.add_trace(
                go.Bar(
                    name=f"{report_type} {metric}",
                    x=df.loc[mask, "class"],
                    y=df.loc[mask, metric],
                    text=df.loc[mask, metric].round(3),
                    textposition="auto",
                    marker_color=color,
                    marker_pattern_shape=patterns[report_type],
                    marker_line_width=1,
                    marker_line_color="rgba(0,0,0,0.3)",
                    hovertemplate=f"<b>%{{x}}</b><br>{metric}: %{{y:.3f}}<br>{report_type}<br>",
                    width=0.15,
                )
            )

    # Add accuracy annotations centered and side by side
    fig.add_annotation(
        x=0.4,
        y=1.15,
        text=f"Train Accuracy: {train_report_dict['accuracy']:.3f}",
        showarrow=False,
        font=dict(size=11, family="Arial", color="#444"),
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="#444",
        borderwidth=1,
        borderpad=4,
        xref="paper",
        yref="paper",
        xanchor="center",
    )

    fig.add_annotation(
        x=0.6,
        y=1.15,
        text=f"Test Accuracy: {test_report_dict['accuracy']:.3f}",
        showarrow=False,
        font=dict(size=11, family="Arial", color="#444"),
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="#444",
        borderwidth=1,
        borderpad=4,
        xref="paper",
        yref="paper",
        xanchor="center",
    )

    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "y": 0.98,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=16, family="Arial", color="#333"),
        },
        barmode="group",
        yaxis=dict(
            range=[0, 1],
            title=dict(text="Score", font=dict(size=12, family="Arial")),
            gridwidth=0.5,
            gridcolor="rgba(128,128,128,0.1)",
            zeroline=True,
            zerolinecolor="rgba(0,0,0,0.2)",
            zerolinewidth=1,
            tickfont=dict(size=10, family="Arial"),
        ),
        xaxis=dict(
            title=dict(text="Class", font=dict(size=12, family="Arial")),
            tickangle=0,
            showgrid=False,
            type="category",
            tickfont=dict(size=10, family="Arial"),
        ),
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.22,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
            font=dict(size=10, family="Arial"),
            groupclick="toggleitem",
            traceorder="normal",
        ),
        margin=dict(t=140, l=25, r=25, b=25),
        hoverlabel=dict(bgcolor="white", font_size=11, font_family="Arial"),
        plot_bgcolor="white",
        showlegend=True,
        bargap=0.08,
        bargroupgap=0.02,
        barcornerradius=3,
        width=800,
        height=450,
        paper_bgcolor="white",
    )

    # current_time = datetime.datetime.now().replace(microsecond=0)
    fig.show()
    # fig.write_image(f"images/{title}_{current_time}.png")


def plot_classification_test_report(title, test_report_dict):
    # Extract class metrics
    class_data = []
    metrics_to_plot = ["precision", "recall", "f1-score"]

    # Process regular classes
    for class_name, metrics in test_report_dict.items():
        if isinstance(metrics, dict) and class_name not in [
            "macro avg",
            "weighted avg",
        ]:
            class_data.append(
                {
                    "class": class_name,
                    **{metric: metrics[metric] for metric in metrics_to_plot},
                }
            )

    # Create DataFrame
    df = pd.DataFrame(class_data)

    # Create figure with secondary y-axis
    fig = go.Figure()

    # Define colors for metrics
    colors = {"precision": "#00CC96", "recall": "#636EFA", "f1-score": "#EF553B"}

    # Add traces for class metrics
    for metric, color in colors.items():
        mask = df[metric].notna()
        fig.add_trace(
            go.Bar(
                name=metric,
                x=df.loc[mask, "class"],
                y=df.loc[mask, metric],
                text=df.loc[mask, metric].round(3),
                textposition="auto",
                marker_color=color,
                hovertemplate="<b>%{x}</b><br>" + f"{metric}: %{{y:.3f}}<br>",
                width=0.25,
            )
        )

    # Add accuracy as annotation
    fig.add_annotation(
        x=0.5,  # Center horizontally
        y=1.15,  # Position above the plot, near column names
        text=f"Accuracy: {test_report_dict['accuracy']:.3f}",
        showarrow=False,
        font=dict(size=14, color="#FFA500"),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="#FFA500",
        borderwidth=2,
        borderpad=4,
        xref="paper",
        yref="paper",
        xanchor="center",
    )

    # Update layout
    fig.update_layout(
        title={
            "text": title,
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        barmode="group",
        yaxis=dict(
            range=[0, 1],
            title="Score",
            gridwidth=1,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=False,
        ),
        xaxis=dict(title="Class", tickangle=0, showgrid=False, type="category"),
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        margin=dict(t=100, l=50, r=50, b=50),
        hoverlabel=dict(bgcolor="white"),
        plot_bgcolor="white",
        showlegend=True,
        bargap=0.15,
        bargroupgap=0.1,
        barcornerradius=15,
    )

    fig.show()


def plot_learning_curves_plotly(train_sizes, train_scores, val_scores):
    """Create a Plotly figure for learning curves."""
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    fig = go.Figure()

    # Training data
    fig.add_trace(
        go.Scatter(
            x=train_sizes,
            y=train_mean,
            mode="lines+markers",
            name="Training Score",
            line=dict(color="blue"),
            error_y=dict(type="data", array=train_std, visible=True),
        )
    )

    # Validation data
    fig.add_trace(
        go.Scatter(
            x=train_sizes,
            y=val_mean,
            mode="lines+markers",
            name="Validation Score",
            line=dict(color="red"),
            error_y=dict(type="data", array=val_std, visible=True),
        )
    )

    fig.update_layout(
        title="Learning Curves",
        xaxis_title="Training Examples",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.05]),
        legend=dict(x=0, y=0),
        template="plotly_white",
    )

    return fig


def plot_confusion_matrix(title, y_true, y_pred):
    """Create an annotated confusion matrix heatmap for binary classification."""
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Sahte (0)", "Gerçek (1)"]

    # Normalized version for color scale
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Annotation text: count + percentage
    annotations = [
        [f"{cm[i][j]}<br>{cm_norm[i][j]:.1%}" for j in range(2)] for i in range(2)
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm_norm,
            x=labels,
            y=labels,
            colorscale=[
                [0.0, "#ffffff"],
                [0.4, "#a8d5ff"],
                [1.0, "#1a56db"],
            ],
            showscale=True,
            colorbar=dict(title="Oran", tickformat=".0%"),
            text=annotations,
            texttemplate="%{text}",
            textfont=dict(size=14, family="Arial"),
            hovertemplate="Gerçek: %{y}<br>Tahmin: %{x}<br>Sayı: %{text}<extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "font": dict(size=15, family="Arial", color="#333"),
        },
        xaxis=dict(
            title=dict(text="Tahmin Edilen Sınıf", font=dict(size=12, family="Arial")),
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            title=dict(text="Gerçek Sınıf", font=dict(size=12, family="Arial")),
            tickfont=dict(size=11),
            autorange="reversed",
        ),
        template="plotly_white",
        width=450,
        height=400,
        margin=dict(t=80, l=100, r=60, b=80),
    )

    fig.show()
    return fig


def plot_dual_confusion_matrix(title, y_true_1, y_pred_1, label_1, y_true_2, y_pred_2, label_2):
    """Side-by-side confusion matrices for two models (Şekil 1 in report)."""
    from plotly.subplots import make_subplots

    cm1 = confusion_matrix(y_true_1, y_pred_1)
    cm2 = confusion_matrix(y_true_2, y_pred_2)
    labels = ["Sahte (0)", "Gerçek (1)"]

    def norm_and_text(cm):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        text = [[f"{cm[i][j]}<br>{cm_norm[i][j]:.1%}" for j in range(2)] for i in range(2)]
        return cm_norm, text

    cm1_norm, text1 = norm_and_text(cm1)
    cm2_norm, text2 = norm_and_text(cm2)

    colorscale = [[0.0, "#ffffff"], [0.4, "#a8d5ff"], [1.0, "#1a56db"]]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[f"<b>{label_1}</b>", f"<b>{label_2}</b>"],
        horizontal_spacing=0.12,
    )

    for col, (cm_norm, text) in enumerate([(cm1_norm, text1), (cm2_norm, text2)], 1):
        fig.add_trace(
            go.Heatmap(
                z=cm_norm,
                x=labels,
                y=labels,
                colorscale=colorscale,
                showscale=(col == 2),
                colorbar=dict(title="Oran", tickformat=".0%", x=1.02),
                text=text,
                texttemplate="%{text}",
                textfont=dict(size=13, family="Arial"),
                hovertemplate="Gerçek: %{y}<br>Tahmin: %{x}<br>%{text}<extra></extra>",
            ),
            row=1,
            col=col,
        )

    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "y": 0.97,
            "x": 0.5,
            "xanchor": "center",
            "font": dict(size=15, family="Arial", color="#333"),
        },
        template="plotly_white",
        width=800,
        height=420,
        margin=dict(t=100, l=80, r=80, b=80),
    )

    for col in [1, 2]:
        fig.update_xaxes(title_text="Tahmin Edilen", row=1, col=col)
        fig.update_yaxes(title_text="Gerçek Sınıf", autorange="reversed", row=1, col=col)

    fig.show()
    return fig


def plot_roc_curve_plotly(y_true, y_score):
    """Create a Plotly figure for ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()

    # Add ROC curve
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC curve (area = {roc_auc:.2f})",
            line=dict(color="darkorange", width=2),
        )
    )

    # Add diagonal line (random classifier)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(color="navy", width=2, dash="dash"),
        )
    )

    fig.update_layout(
        title="Receiver Operating Characteristic",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain="domain"),
        legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1),
        width=700,
        height=500,
        template="plotly_white",
    )

    return fig
