import gokart
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

class LoadData(gokart.TaskOnKart):
    def output(self):
        return self.make_target('./iris_data.csv')

    def run(self):
        iris = load_iris()
        data_array = iris.data
        feature = iris.feature_names

        target = iris.target

        df_data = pd.DataFrame(data_array, columns=feature)
        df_target = pd.DataFrame(target, columns=['target'])
        df = pd.concat([df_data, df_target], axis=1)
        self.dump(df)
        
class PreTreatment(gokart.TaskOnKart):
    data_df = gokart.TaskInstanceParameter()

    # データの出力とscalerの出力
    def output(self):
        return {
            'scaled_data': self.make_target('./iris_data_scaled.csv'),
            'scaler': self.make_target('./scaler.pkl')
        }

    def run(self):
        df: pd.DataFrame = self.load(self.data_df)
        scaler = StandardScaler()
        X = df.drop('target', axis=1)
        scaler.fit(X)
        scaled_data = scaler.transform(X)
        self.dump(pd.DataFrame(scaled_data, columns=X.columns), 'scaled_data')
        self.dump(scaler, 'scaler')

class TrainModel(gokart.TaskOnKart):
    data_df = gokart.TaskInstanceParameter()
    pre_treatment = gokart.TaskInstanceParameter()

    def output(self):
        return self.make_target('./model.pkl')

    def run(self):
        df: pd.DataFrame = self.load(self.data_df)
        X: pd.DataFrame = self.load(self.pre_treatment)['scaled_data']
        y: pd.Series = df['target']

        clf = RandomForestClassifier()
        clf.fit(X, y)
        self.dump(clf)

class MLPipleLine(gokart.TaskOnKart):
    def requires(self):
        data = LoadData()
        pre_treatment = PreTreatment(data_df=data)
        train_model = TrainModel(data_df=data, pre_treatment=pre_treatment)
        return {'data': data, 'pre_treatment': pre_treatment, 'model': train_model}

    def output(self):
        return self.make_target('./model.csv')
    
    def run(self):
        X: pd.DataFrame = self.load('pre_treatment')['scaled_data']
        data: pd.DataFrame = self.load('data')
        target = data['target']

        model = self.load('model')
        result = model.predict(X)

        r2_result = r2_score(target, result)
        self.dump(pd.Series(r2_result, name='R2'))

if __name__ == '__main__':
    task = MLPipleLine()
    output = gokart.build(task)
    print(output)
