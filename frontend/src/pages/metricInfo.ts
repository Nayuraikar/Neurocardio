export const METRIC_INFO: Record<string, { name: string; definition: string; reading: string }> = {
  precision: { name: 'Precision', definition: 'Of the windows the model called this class, the share that really are.', reading: 'Low precision means many false alarms for that class. It depends on how common the class is.' },
  recall: { name: 'Recall', definition: 'Of the windows that really are this class, the share the model found.', reading: 'Low recall means the class is usually missed. This is the main weakness for S and F.' },
  specificity: { name: 'Specificity', definition: 'Of the windows that are not this class, the share the model did not call it.', reading: 'It looks high for rare classes even when the model rarely finds them, so read it next to recall.' },
  f1: { name: 'F1', definition: 'Harmonic mean of precision and recall.', reading: 'Macro-F1 averages the four class F1 scores with equal weight, so rare classes count as much as N.' },
  roc_auc: { name: 'ROC-AUC', definition: 'How well the class score ranks members above non-members, across all thresholds.', reading: 'A good ranking does not mean the default decision finds the class; compare with recall.' },
  pr_auc: { name: 'PR-AUC', definition: 'Area under the precision-recall curve.', reading: 'The reference level for a random ranking is the class prevalence, not 0.5.' },
}
