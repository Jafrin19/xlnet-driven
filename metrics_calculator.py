import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

class MetricsCalculator:
    def __init__(self, labels, text_widget=None):
        self.labels = labels
        self.text = text_widget  # Tkinter Text widget handle

        self.precision = []
        self.recall = []
        self.fscore = []
        self.accuracy = []

        self.metrics_df = pd.DataFrame(columns=['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
        self.class_report_df = pd.DataFrame()
        self.class_performance_dfs = {}

        if not os.path.exists('results'):
            os.makedirs('results')

    def calculate_metrics(self, algorithm, predict, y_test, y_score=None):
        categories = self.labels

        # Overall metrics
        a = accuracy_score(y_test, predict) * 100
        p = precision_score(y_test, predict, average='macro') * 100
        r = recall_score(y_test, predict, average='macro') * 100
        f = f1_score(y_test, predict, average='macro') * 100

        self.accuracy.append(a)
        self.precision.append(p)
        self.recall.append(r)
        self.fscore.append(f)

        metrics_entry = pd.DataFrame({
            'Algorithm': [algorithm],
            'Accuracy': [a],
            'Precision': [p],
            'Recall': [r],
            'F1-Score': [f]
        })
        self.metrics_df = pd.concat([self.metrics_df, metrics_entry], ignore_index=True)

        # Print to Text widget if available
        if self.text:
            self.text.insert('end', f"{algorithm} Accuracy  : {a:.4f}\n")
            self.text.insert('end', f"{algorithm} Precision : {p:.4f}\n")
            self.text.insert('end', f"{algorithm} Recall    : {r:.4f}\n")
            self.text.insert('end', f"{algorithm} FScore    : {f:.4f}\n\n")

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, predict)
        CR = classification_report(y_test, predict, target_names=[str(c) for c in categories], output_dict=True)

        if self.text:
            self.text.insert('end', f"{algorithm} Classification Report\n")
            self.text.insert('end', classification_report(y_test, predict, target_names=[str(c) for c in categories]) + "\n")

        cr_df = pd.DataFrame(CR).transpose()
        cr_df['Algorithm'] = algorithm
        self.class_report_df = pd.concat([self.class_report_df, cr_df], ignore_index=False)

        # Per-class performance
        for category in categories:
            class_entry = pd.DataFrame({
                'Algorithm': [algorithm],
                'Precision': [CR[str(category)]['precision'] * 100],
                'Recall': [CR[str(category)]['recall'] * 100],
                'F1-Score': [CR[str(category)]['f1-score'] * 100],
                'Support': [CR[str(category)]['support']]
            })

            if str(category) not in self.class_performance_dfs:
                self.class_performance_dfs[str(category)] = pd.DataFrame(columns=['Algorithm', 'Precision', 'Recall', 'F1-Score', 'Support'])

            self.class_performance_dfs[str(category)] = pd.concat([self.class_performance_dfs[str(category)], class_entry], ignore_index=True)

        # Confusion Matrix Plot
        plt.figure() 
        ax = sns.heatmap(
            conf_matrix,
            xticklabels=categories,
            yticklabels=categories,
            annot=True,
            cmap="viridis",
            fmt="g",
            cbar=False       )

        ax.set_ylim([0, len(categories)])  # Keeps all rows visible
        plt.title(algorithm + " Confusion Matrix", fontsize=14, pad=15)
        plt.ylabel('True Class', fontsize=12)
        plt.xlabel('Predicted Class', fontsize=12)

        plt.tight_layout()  # Adjusts subplots to fit within figure area

        # ✅ Save figure without cropping anything
        plt.savefig(
            f"results/{algorithm.replace(' ', '_')}_confusion_matrix.png",
            dpi=300,
            bbox_inches='tight'  # Ensures full content is saved
        )
        plt.show()

        if y_score is not None:

            # Convert output labels to numeric if needed
            class_indices = {c: i for i, c in enumerate(categories)}
            if isinstance(y_test[0], str):     # string labels → convert
                class_indices = {c: i for i, c in enumerate(categories)}
                y_test_num = np.array([class_indices[y] for y in y_test])
            else:                              # already numeric → use directly
                y_test_num = np.array(y_test)
            # One-vs-Rest binarization
            y_test_bin = label_binarize(y_test_num, classes=range(len(categories)))

            # Number of classes
            n_classes = y_test_bin.shape[1]

            fpr = {}
            tpr = {}
            roc_auc = {}

            # ---------------------------------------------
            # COMPUTE ROC FOR EACH CLASS
            # ---------------------------------------------
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # ---------------------------------------------
            # MICRO-AVERAGE ROC (global)
            # ---------------------------------------------
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            # ---------------------------------------------
            # PLOT ROC CURVES
            # ---------------------------------------------
            plt.figure(figsize=(10, 8))

            # Plot each class ROC
            for i in range(n_classes):
                plt.plot(
                    fpr[i], tpr[i],
                    label=f'{categories[i]} (AUC = {roc_auc[i]:.2f})',
                    linewidth=2
                )

            # Plot micro-average
            plt.plot(
                fpr["micro"], tpr["micro"],
                linestyle='--', linewidth=3,
                label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})'
            )

            plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
            plt.title(f"{algorithm} ROC Curves (One-vs-Rest)")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"results/{algorithm.replace(' ', '_')}_roc_curve.png")
            plt.show()    
            

    def plot_classification_graphs(self):
        melted_df = pd.melt(self.metrics_df, id_vars=['Algorithm'],
                            value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                            var_name='Parameters', value_name='Value')

        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='Parameters', y='Value', hue='Algorithm', data=melted_df)
        plt.title('Classifier Performance Comparison', fontsize=14, pad=10)
        plt.ylabel('Score (%)', fontsize=12)
        plt.xlabel('Metrics', fontsize=12)
        plt.xticks(rotation=0)
        plt.legend(title='Algorithms', bbox_to_anchor=(1.05, 1), loc='upper left')

        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', padding=3)

        os.makedirs('results', exist_ok=True)
        plt.tight_layout()
        plt.savefig('results/classifier_performance.png')
        plt.show()

        # Class-specific bar plots
        for class_name, class_df in self.class_performance_dfs.items():
            melted_class_df = pd.melt(class_df, id_vars=['Algorithm'],
                                      value_vars=['Precision', 'Recall', 'F1-Score'],
                                      var_name='Parameters', value_name='Value')

            plt.figure(figsize=(12, 6))
            ax = sns.barplot(x='Parameters', y='Value', hue='Algorithm', data=melted_class_df)
            plt.title(f'Class {class_name} Performance Comparison', fontsize=14, pad=10)
            plt.ylabel('Score (%)', fontsize=12)
            plt.xlabel('Metrics', fontsize=12)
            plt.xticks(rotation=0)
            plt.legend(title='Algorithms', bbox_to_anchor=(1.05, 1), loc='upper left')

            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f', padding=3)

            plt.tight_layout()
            plt.savefig(f'results/class_{class_name}_performance.png')
            plt.show()

        # Return formatted metrics table
        melted_df_new = self.metrics_df[['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].copy()
        melted_df_new = melted_df_new.round(3)
        return melted_df_new
