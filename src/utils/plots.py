import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score


def plot_numerical_distribution(data, numerical_columns, target_column=None, figsize=(15, 10), kde=True, color='skyblue'):
    """
    Plot the distribution of numerical columns in subplots, optionally grouped by the target variable.

    Parameters:
    - data: DataFrame containing the numerical columns.
    - numerical_columns: List of numerical column names to plot.
    - target_column: Name of the target variable column. Default is None.
    - figsize: Size of the overall figure. Default is (15, 10).
    - kde: If True, plot the kernel density estimate along with the histogram. Default is True.
    - color: Color of the histogram bars. Default is 'skyblue'.

    Returns:
    - None
    """

    num_plots = len(numerical_columns)
    nrows = math.ceil(num_plots / 3)
    ncols = min(3, num_plots)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    axes = axes.flatten()

    for i, column in enumerate(numerical_columns):
        if target_column:
            sns.histplot(data=data, x=column, hue=target_column, ax=axes[i], kde=kde, palette='Set2', edgecolor='black', bins=30)
        else:
            sns.histplot(data=data[column], ax=axes[i], kde=kde, color=color, edgecolor='black', bins=30)

        axes[i].set_title(column)
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

    for i in range(num_plots, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()



def plot_boxplots(data, numerical_columns, target_column, pos_color, neg_color, figsize=(15, 10), showfliers = False):
    """
    Plot boxplots for numerical columns with hue grouped by the target variable in subplots.

    Parameters:
    - data: DataFrame containing the data.
    - numerical_columns: List of numerical column names to plot.
    - target_column: Name of the target variable column.
    - pos_color: Color for the positive class (e.g., '#f54900').
    - neg_color: Color for the negative class (e.g., '#0a664f').
    - figsize: Size of the overall figure. Default is (15, 10).

    Returns:
    - None
    """

    num_plots = len(numerical_columns)
    nrows = math.ceil(num_plots / 3)
    ncols = min(3, num_plots)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    axes = axes.flatten()

    colors = {0: neg_color, 1: pos_color}

    for i, column in enumerate(numerical_columns):
        sns.boxplot(x=target_column, y=column, data=data, ax=axes[i], palette=colors,showfliers = showfliers)
        axes[i].set_title(column)
        axes[i].set_xlabel(target_column)
        axes[i].set_ylabel(column)

    for i in range(num_plots, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def plot_categorical_variables(df, categorical_columns,hue=None, figsize=(15, 15), color = 'Set2'):
    """
    Plot the relationship between multiple categorical features and the target variable in subplots.

    Parameters:
    - df: DataFrame containing the data.
    - categorical_columns: List of categorical column names to plot.
    - hue: Optional name of the hue variable for grouping. Default is None.
    - figsize: Size of the overall figure. Default is (15, 10).

    Returns:
    - None
    """
    num_plots = len(categorical_columns)
    nrows = math.ceil(num_plots / 3)
    ncols = min(3, num_plots)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    axes = axes.flatten()

    for i, column in enumerate(categorical_columns):
        ax = axes[i]
        if hue:
            sns.countplot(x=column, hue=hue, data=df, ax=ax, palette=color)
            
        else:
            sns.countplot(x=column, data=df, ax=ax)
        ax.set_title(column)
        ax.set_xlabel(None)
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=60)
        ax.legend(title=hue, loc='upper right') if hue else None

    for i in range(num_plots, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_probs, auc_score=None, title = 'Receiver Operating Characteristic (ROC) Curve', image_path = None):
    """
    Plot the Receiver Operating Characteristic (ROC) curve for binary classification.

    Parameters:
    - y_true : array-like of shape (n_samples,)
        True binary labels.
    - y_probs : array-like of shape (n_samples,)
        Target scores or probabilities of the positive class.
    - auc_score : float, optional (default=None)
        Area under the ROC curve (AUC) score. If not provided, it will be computed from y_true and y_probs.

    Returns:
    None
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    
    if auc_score is None:
        auc_score = roc_auc_score(y_true, y_probs)
    
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)

    if image_path:
        plt.savefig(image_path)
    else:
        plt.show()



def plot_feature_importance(model, feature_names, top_n=10, figsize=(10, 6)):
    """
    Plot feature importance for a Random Forest model.

    Parameters:
    - model: trained Random Forest model
    - feature_names: list of feature names
    - top_n: number of top features to display (default: 10)
    - figsize: tuple specifying the figure size (default: (10, 6))
    """
    
    importances = model.feature_importances_
    indices = importances.argsort()[-top_n:]

    plt.figure(figsize=figsize)
    plt.title("Feature Importances")
    plt.barh(range(len(indices)), importances[indices], color="b", align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.gca().invert_yaxis()
    plt.show()