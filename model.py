import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

def train_model():
    df=pd.read_csv("shopping_data.csv")

    X=df[["price","salary","expenses","usage","need_score"]]

    y_safety=df["safety"]
    y_regret=df["regret"]

    X_train, X_test, y_train_s, y_test_s=train_test_split(X,y_safety,test_size=0.2, random_state=42)
    _, _, y_train_r, y_test_r=train_test_split(X,y_regret,test_size=0.2, random_state=42)

    model_s=Ridge()
    model_r=Ridge()

    model_s.fit(X_train,y_train_s)
    model_r.fit(X_train,y_train_r)

    with open("model.pkl","wb")as f:
        pickle.dump((model_s, model_r),f)
    print("Model trained and saved")

if __name__=="__main__":
    train_model()
