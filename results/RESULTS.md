# Results (Part 2: CNNs on complete drawings)

Test set: 12,276 drawings not seen in training, balanced over 9 regions (chance = 11.1%).

| Method | Region accuracy | Top-3 |
|---|---|---|
| Always guess most common region | 11.1% | |
| Logistic regression (28x28 pixels) | 13.7% | |
| small CNN | 16.7% | 44.8% |
| deep CNN | 18.2% | 46.4% |

## CNN accuracy by true region

| Region | small CNN | deep CNN |
|---|---|---|
| North America | 26.7% | 34.3% |
| Latin America | 11.0% | 6.3% |
| Western Europe | 5.1% | 2.7% |
| Eastern Europe | 33.4% | 33.4% |
| Middle East & Africa | 2.9% | 6.8% |
| South Asia | 22.9% | 18.8% |
| East Asia | 40.4% | 48.2% |
| Southeast Asia | 0.9% | 6.9% |
| Oceania | 7.3% | 5.9% |

## CNN accuracy by object

| Object | small CNN | deep CNN |
|---|---|---|
| power outlet | 18.5% | 21.8% |
| house | 16.0% | 19.9% |
| tree | 16.7% | 19.8% |
| chair | 18.3% | 18.8% |
| bus | 16.8% | 18.2% |
| cup | 17.7% | 17.5% |
| bread | 15.5% | 16.7% |
| face | 13.9% | 16.7% |
| car | 16.7% | 16.5% |
| mailbox | 16.9% | 15.9% |


## Object recognition (small CNN, what is drawn)

Test accuracy: **97.0%** over 10 objects (chance = 10%).

Confusion matrix (deep CNN): `confusion_matrix.png`. Example drawings: `samples.png`.
