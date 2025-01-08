import gokart
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# タスク1: データの読み込み
class LoadData(gokart.TaskOnKart):
    def output(self):
        # データの保存先を指定
        return self.make_target('./iris_data.csv')

    def run(self):
        # Irisデータセットをロード
        iris = load_iris()
        data_array = iris.data  # 特徴量データ
        feature = iris.feature_names  # 特徴量名
        target = iris.target  # ターゲットデータ

        # 特徴量とターゲットをDataFrameに変換
        df_data = pd.DataFrame(data_array, columns=feature)
        df_target = pd.DataFrame(target, columns=['target'])
        df = pd.concat([df_data, df_target], axis=1)  # 特徴量とターゲットを結合
        self.dump(df)  # データを保存

# タスク2: データの前処理
class PreTreatment(gokart.TaskOnKart):
    data_df = gokart.TaskInstanceParameter()  # 入力データタスクを指定

    def output(self):
        # スケーリング済みデータとスケーラーの保存先を指定
        return {
            'scaled_data': self.make_target('./iris_data_scaled.csv'),
            'scaler': self.make_target('./scaler.pkl')
        }

    def run(self):
        df: pd.DataFrame = self.load(self.data_df)  # 入力データを読み込み
        scaler = StandardScaler()  # 標準化用のスケーラーを初期化
        X = df.drop('target', axis=1)  # 特徴量データを抽出
        scaler.fit(X)  # スケーラーをフィット
        scaled_data = scaler.transform(X)  # データをスケーリング

        # スケーリング済みデータとスケーラーを保存
        self.dump(pd.DataFrame(scaled_data, columns=X.columns), 'scaled_data')
        self.dump(scaler, 'scaler')

# タスク3: モデルの学習
class TrainModel(gokart.TaskOnKart):
    data_df = gokart.TaskInstanceParameter()  # 入力データタスクを指定
    pre_treatment = gokart.TaskInstanceParameter()  # 前処理タスクを指定

    def output(self):
        # 学習済みモデルの保存先を指定
        return self.make_target('./model.pkl')

    def run(self):
        df: pd.DataFrame = self.load(self.data_df)  # 入力データを読み込み
        X: pd.DataFrame = self.load(self.pre_treatment)['scaled_data']  # スケーリング済みデータを取得
        y: pd.Series = df['target']  # ターゲットデータを取得

        clf = RandomForestClassifier()  # ランダムフォレストモデルを初期化
        clf.fit(X, y)  # モデルを学習
        self.dump(clf)  # 学習済みモデルを保存

# タスク4: 全体のパイプライン
class MLPipleLine(gokart.TaskOnKart):
    def requires(self):
        # 各タスクを順序立てて指定
        data = LoadData()  # データ読み込みタスク
        pre_treatment = PreTreatment(data_df=data)  # 前処理タスク
        train_model = TrainModel(data_df=data, pre_treatment=pre_treatment)  # モデル学習タスク
        return {'data': data, 'pre_treatment': pre_treatment, 'model': train_model}

    def output(self):
        # 最終結果の保存先を指定
        return self.make_target('./model.csv')

    def run(self):
        # 前処理済みデータと元データを読み込み
        X: pd.DataFrame = self.load('pre_treatment')['scaled_data']
        data: pd.DataFrame = self.load('data')
        target = data['target']  # ターゲットデータを取得

        # 学習済みモデルを読み込み
        model = self.load('model')
        result = model.predict(X)  # データを予測

        # R2スコアを計算
        r2_result = r2_score(target, result)
        self.dump(pd.Series(r2_result, name='R2'))  # R2スコアを保存

# メイン処理
if __name__ == '__main__':
    task = MLPipleLine()  # パイプラインタスクを初期化
    output = gokart.build(task)  # パイプラインを実行
    print(output)  # 実行結果を表示