# Results (Part 2: CNNs on complete drawings)

Test set: 12,276 drawings not seen in training, balanced over 9 regions (chance = 11.1%).

| Method | Region accuracy | Top-3 |
|---|---|---|
| Always guess most common region | 11.1% | |
| Logistic regression (28x28 pixels) | 13.7% | |
| small CNN | 15.5% | 41.0% |
| deep CNN | 17.0% | 44.3% |

## CNN accuracy by true region

| Region | small CNN | deep CNN |
|---|---|---|
| North America | 17.1% | 37.9% |
| Latin America | 8.7% | 5.7% |
| Western Europe | 5.6% | 0.3% |
| Eastern Europe | 28.8% | 28.7% |
| Middle East & Africa | 4.6% | 6.1% |
| South Asia | 26.0% | 32.3% |
| East Asia | 41.1% | 34.2% |
| Southeast Asia | 1.8% | 1.5% |
| Oceania | 5.9% | 5.8% |

## CNN accuracy by object

| Object | small CNN | deep CNN |
|---|---|---|
| power outlet | 18.8% | 20.9% |
| tree | 15.4% | 19.5% |
| chair | 17.7% | 17.7% |
| cup | 15.2% | 17.5% |
| car | 15.0% | 16.6% |
| house | 15.7% | 16.4% |
| face | 13.0% | 15.6% |
| bread | 15.2% | 15.6% |
| bus | 15.5% | 15.0% |
| mailbox | 13.1% | 14.4% |


## Object recognition (small CNN, what is drawn)

Test accuracy: **93.0%** over 10 objects (chance = 10%).

Confusion matrix (deep CNN): `confusion_matrix.png`. Example drawings: `samples.png`.
