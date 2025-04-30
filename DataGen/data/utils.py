from DataGen.data.datagen import DataGenerator
from sklearn import metrics


def AUCFromDistributions(dist1, dist2):
    dg = DataGenerator(dist1, dist2, 0.5)
    n=5000
    x, y = dg.pn_data(n)[0:2]
    posterior_x = dg.pn_posterior_balanced(x)
    fpr, tpr, thresholds = metrics.roc_curve(y, posterior_x)
    aucpn = metrics.auc(fpr, tpr)
    return aucpn