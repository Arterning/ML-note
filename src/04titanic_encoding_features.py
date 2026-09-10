"""演示 Titanic 数据的序数、独热、标签编码与特征工程。"""

from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "titanic.csv"

RARE_TITLES = [
    "Lady", "Countess", "Capt", "Col", "Don", "Dr",
    "Major", "Rev", "Sir", "Jonkheer", "Dona",
]


def demonstrate_encoders(dataframe):
    ordinal = dataframe[["Fare"]].copy()
    ordinal["fare_category"] = pd.cut(
        ordinal["Fare"], bins=3, labels=["Low", "Mid", "High"]
    )
    ordinal_encoder = OrdinalEncoder(categories=[["Low", "Mid", "High"]])
    ordinal["fare_cat_encoded"] = ordinal_encoder.fit_transform(
        ordinal[["fare_category"]]
    )
    print("序数编码（Low < Mid < High）：")
    print(ordinal[["fare_category", "fare_cat_encoded"]].head())

    towns = dataframe["Embarked"].fillna("Unknown").map(
        {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown", "Unknown": "Unknown"}
    )
    onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded = onehot_encoder.fit_transform(towns.to_frame(name="embark_town"))
    encoded_frame = pd.DataFrame(
        encoded,
        columns=onehot_encoder.get_feature_names_out(["embark_town"]),
        index=dataframe.index,
    )
    print("\n独热编码生成的列：")
    print(list(encoded_frame.columns))
    print(encoded_frame.head())

    label_encoder = LabelEncoder()
    sex_encoded = label_encoder.fit_transform(dataframe["Sex"])
    print("\n性别标签编码映射：")
    print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
    print("前 5 个编码值:", sex_encoded[:5])


def create_features(dataframe):
    """对训练集和测试集使用完全相同的特征构造规则。"""
    result = dataframe.copy()
    result["family_size"] = result["SibSp"] + result["Parch"] + 1
    result["is_alone"] = (result["family_size"] == 1).astype(int)
    result["title"] = result["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    result["title"] = result["title"].replace(RARE_TITLES, "Rare")
    result["title"] = result["title"].replace(
        {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
    )
    result["HasCabin"] = result["Cabin"].notna().astype(int)
    result.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)
    return result


def demonstrate_train_test_processing(dataframe):
    X_train, X_test = train_test_split(
        dataframe.drop(columns="Survived"), test_size=0.2, random_state=42
    )
    X_train = X_train.copy()
    X_test = X_test.copy()

    for column, strategy in [("Age", "median"), ("Embarked", "most_frequent")]:
        imputer = SimpleImputer(strategy=strategy)
        X_train[[column]] = imputer.fit_transform(X_train[[column]])
        X_test[[column]] = imputer.transform(X_test[[column]])

    train_features = create_features(X_train)
    test_features = create_features(X_test)

    scaler = StandardScaler()
    numeric_columns = ["Age", "Fare"]
    train_features[numeric_columns] = scaler.fit_transform(train_features[numeric_columns])
    test_features[numeric_columns] = scaler.transform(test_features[numeric_columns])

    print("\n特征工程后训练集形状:", train_features.shape)
    print(train_features[["family_size", "is_alone", "title", "HasCabin"]].head())
    print("\n标准化后训练集 Age/Fare 均值:")
    print(train_features[numeric_columns].mean().round(6))
    print("测试集使用训练集上拟合的填充器和缩放器。")


def main():
    dataframe = pd.read_csv(DATA_PATH)
    dataframe.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)
    demonstrate_encoders(dataframe)
    demonstrate_train_test_processing(dataframe)


if __name__ == "__main__":
    main()
