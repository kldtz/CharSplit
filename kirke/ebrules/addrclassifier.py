from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer


class LogRegModel:
    def __init__(self,model = None, vec= None, featureselector= None):
      if (model is None):
          self.model = LogisticRegression()
          self.vec = DictVectorizer()
          self.allFeatures = []
          self.allCorrect = []
          self.posTagsDict = defaultdict(list)
          self.currTimestamp = str(datetime.now())
      else:
        self.model = model
        self.vec = vec
        self.featSelect = featureselector

    def extract_features(self, addr):
        featureset = {}
        

    def fitModel(self, train, labels):
        X = []
        for addr in train:
            X.append(self.extract_features(addr))

        self.featSelect = SelectFromModel(RandomForestClassifier()).fit(X,y)

    def predict(self, addr):
        features = self.extract_features(addr)

def parse_train(fname):
    train_data = []
    train_labels = []
    with open(fname) as train:
        for line in train:
            data, label = line.split("\t")
            train_data.append(data)
            train_labels.append(label.strip())
    return train_data, train_labels


def main():
    model = LogRegModel()
    train_data, train_labels = parse_train("addr_annots.tsv")
    model.fitModel(train, labels)
