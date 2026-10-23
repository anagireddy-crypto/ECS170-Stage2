from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from local_code.base_class.evaluate import evaluate


class Evaluate_Classification_Stage2(evaluate):
    def __init__(self):
        super().__init__('classification scores', 'Accuracy and weighted scores')

    def evaluate(self):
        true_y = self.data['true_y']
        pred_y = self.data['pred_y']

        return {
            'accuracy': accuracy_score(true_y, pred_y),
            'weighted_precision': precision_score(true_y, pred_y, average='weighted', zero_division=0),
            'weighted_recall': recall_score(true_y, pred_y, average='weighted', zero_division=0),
            'weighted_f1': f1_score(true_y, pred_y, average='weighted', zero_division=0),
        }
