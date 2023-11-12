from torch.nn.functional import softmax
import torch
from torchmetrics.classification.accuracy import Accuracy

class CER():
    def __init__(self, dim=-1):
        self.dim = -1
    
    @torch.no_grad()
    def __call__(self, preds, targets):
        outputs = torch.argmax(input=softmax(input=preds, dim=-1), dim=-1)
        correct_predictions = outputs == targets
        acc = torch.mean(correct_predictions.float())
        return 1 - acc.item()