from giniStatistics import giniStat_marignal
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
gCov, gCor = giniStat_marignal(X,y)
print(gCov)
print(gCor)