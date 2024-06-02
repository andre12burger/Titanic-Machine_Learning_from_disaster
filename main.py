import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Criar os data frames a partir dos arquivos .csv
result_data = pd.read_csv('gender_submission.csv')
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# TRATAMENTO DE DADOS
# Tratamento de valores ausentes na coluna 'Age' usando a mediana
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())

# Tratamento de valores ausentes na coluna 'Embarked' usando a moda
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])
test_data['Embarked'] = test_data['Embarked'].fillna(test_data['Embarked'].mode()[0])

# Tratamento de valores ausentes na coluna 'Fare' usando a mediana
train_data['Fare'] = train_data['Fare'].fillna(train_data['Fare'].median())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())




# Substituindo dados não numéricos por dummies
train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'])
test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'])


# Dados a serem trabalhados para fazer a predição
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
x_train = train_data[features]
y_train = train_data['Survived'] #alvo do treino
x_test = test_data[features]

# Treinamento do modelo
model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=1)
model.fit(x_train, y_train)

# Avaliação do modelo
scores = cross_val_score(model, x_train, y_train, cv=5)
print(f'Accuracy: {scores.mean():.2f} ± {scores.std():.2f}')

# Fazer predições no conjunto de teste
y_pred = model.predict(x_test)

# Preparar o arquivo para submissão
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': y_pred
})

# Salvar o arquivo de submissão
submission.to_csv('submission.csv', index=False)


# Comparação e avaliação dos csv gerado e o real
submission_data = pd.read_csv('submission.csv')

# Comparar a segunda coluna dos dois arquivos
real_results = result_data.iloc[:, 1]
predicted_results = submission_data.iloc[:, 1]

# Verificar se os resultados estão corretos
correct_predictions = real_results == predicted_results

# Calcular a porcentagem de acertos
accuracy = correct_predictions.mean() * 100

print(f'Porcentagem de acertos: {accuracy:.2f}%')