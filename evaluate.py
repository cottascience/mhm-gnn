import torch, sys
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
import scipy.sparse as sp

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret

def logit(train_x, train_y, test_x, test_y):

    dim = train_x.size(1)
    nb_classes = len(torch.unique(train_y))
    log = LogReg(dim, nb_classes)

    opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
    xent = nn.CrossEntropyLoss(weight=1/train_y.bincount().float())

    for _ in range(100):
        log.train()
        opt.zero_grad()

        logits = log(train_x)
        loss = xent(logits, train_y)

        loss.backward()
        opt.step()

    logits = log(test_x)
    preds = torch.argmax(logits, dim=1)
    acc = balanced_accuracy_score(test_y.cpu().detach().numpy(), preds.cpu().detach().numpy())

    return acc
