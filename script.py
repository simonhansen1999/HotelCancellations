import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv('datasets/hotel_bookings.csv')

df.head()
df.info()

df['is_canceled'].value_counts(0)
df['is_canceled'].value_counts(1)

df['reservation_status'].value_counts(0)
df['reservation_status'].value_counts(1)

grouped = df.groupby('arrival_date_month')['is_canceled'].mean()
grouped.sort_values()

object_columns = ['is_canceled', 'lead_time', 'arrival_date_year', 'arrival_date_month', 'arrival_date_week_number']

df[object_columns].head()

drop_columns = [
    'country', 'agent', 'company', 'reservation_status_date',
    'arrival_date_week_number', 'arrival_date_day_of_month', 'arrival_date_year'
]

df = df.drop(labels=drop_columns, axis=1)

print(df.head())

pd.set_option('future.no_silent_downcasting', True)
df['meal'] = df['meal'].replace({'Undefined':0, 'SC':0, 'BB':1, 'HB':2, 'FB':3})

print(df['meal'].head())

one_hot_columns = [
    'arrival_date_month', 'market_segment', 'distribution_channel',
    'deposit_type', 'customer_type', 'assigned_room_type', 'reserved_room_type'
]
df = pd.get_dummies(df, columns=one_hot_columns, dtype=int)

print(df.head())

# Remove target columns
remove_cols = ['is_canceled', 'reservation_status']

# Select training features
train_features = [x for x in df.columns if x not in remove_cols]

X = torch.tensor(df[train_features].values, dtype=torch.float)
y = torch.tensor(df['is_canceled'].values, dtype=torch.float).view(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.80,
                                                    test_size=0.20,
                                                    random_state=42) 
print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

torch.manual_seed(42)

model = nn.Sequential(
    nn.Linear(65, 36),
    nn.ReLU(),
    nn.Linear(36, 18),
    nn.ReLU(),
    nn.Linear(18, 1),
    nn.Sigmoid()
)

loss = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

num_epochs = 1000
for epoch in range(num_epochs):
    predictions = model(X_train)
    BCELoss = loss(predictions, y_train)
    BCELoss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if (epoch + 1) % 100 == 0:
        predicted_labels = (predictions >= 0.5).int()
        accuracy = accuracy_score(y_train, predicted_labels)
        print(f'Epoch [{epoch+1}/{num_epochs}], BCELoss: {BCELoss.item():.4f}, Accuracy: {accuracy.item():.4f}')

model.eval()

with torch.no_grad():
    test_predictions = model(X_test)
    test_predicted_labels = (test_predictions >= 0.5).int()
    accuracy = accuracy_score(y_test, test_predicted_labels)
    report = classification_report(y_test, test_predicted_labels)
    
print(f'Accuracy: {accuracy.item():.4f}')
print(report)

df['reservation_status'] = df['reservation_status'].replace({'Check-Out':2, 'Canceled':1, 'No-Show':0})

print(df['reservation_status'].head())

X = torch.tensor(df[train_features].values, dtype=torch.float)
y = torch.tensor(df['reservation_status'].values, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.80,
                                                    test_size=0.20,
                                                    random_state=42) 
print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

torch.manual_seed(42)

multiclass_model = nn.Sequential(
    nn.Linear(65, 65),
    nn.ReLU(),
    nn.Linear(65, 36),
    nn.ReLU(),
    nn.Linear(36, 3),
)

loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 500
for epoch in range(num_epochs):
    predictions = multiclass_model(X_train)
    BCELoss = loss(predictions, y_train)
    BCELoss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if (epoch + 1) % 100 == 0:
        predicted_labels = torch.argmax(predictions, dim=1)
        accuracy = accuracy_score(y_train, predicted_labels)
        print(f'Epoch [{epoch+1}/{num_epochs}], BCELoss: {BCELoss.item():.4f}, Accuracy: {accuracy.item():.4f}')

model.eval()

with torch.no_grad():
    multiclass_predictions = multiclass_model(X_test)
    multiclass_predicted_labels = torch.argmax(multiclass_predictions, dim=1)
    accuracy = accuracy_score(y_test, multiclass_predicted_labels)
    report = classification_report(y_test, multiclass_predicted_labels)
    
print(f'Accuracy: {accuracy.item():.4f}')
print(report)