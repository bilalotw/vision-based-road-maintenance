import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, cohen_kappa_score

def severity_metrics(y_true,y_pred):
    return {"accuracy":float(accuracy_score(y_true,y_pred)),
            "macro_f1":float(f1_score(y_true,y_pred,average="macro")),
            "weighted_f1":float(f1_score(y_true,y_pred,average="weighted")),
            "ordinal_mae":float(mean_absolute_error(y_true,y_pred)),
            "quadratic_weighted_kappa":float(cohen_kappa_score(y_true,y_pred,weights="quadratic"))}

def expected_calibration_error(probs,labels,n_bins=15):
    probs=np.asarray(probs); labels=np.asarray(labels); conf=probs.max(1); pred=probs.argmax(1)
    ece=0.0
    for lo,hi in zip(np.linspace(0,1,n_bins+1)[:-1],np.linspace(0,1,n_bins+1)[1:]):
        m=(conf>lo)&(conf<=hi)
        if m.any(): ece += m.mean()*abs((pred[m]==labels[m]).mean()-conf[m].mean())
    return float(ece)
