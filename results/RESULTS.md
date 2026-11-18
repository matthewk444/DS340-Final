# Results (Part 2: initial CNN, complete drawings)

Test set: 12,276 drawings not seen in training, balanced over 9 regions (chance = 11.1%).

| Method | Region accuracy |
|---|---|
| Always guess most common region | 11.1% |
| Logistic regression (28x28 pixels) | 13.7% |
| **CNN** | **16.7%** (top-3: 44.8%) |

## CNN accuracy by true region

| Region | Accuracy |
|---|---|
| North America | 26.7% |
| Latin America | 11.0% |
| Western Europe | 5.1% |
| Eastern Europe | 33.4% |
| Middle East & Africa | 2.9% |
| South Asia | 22.9% |
| East Asia | 40.4% |
| Southeast Asia | 0.9% |
| Oceania | 7.3% |

## CNN accuracy by object

| Object | Accuracy |
|---|---|
| power outlet | 18.5% |
| chair | 18.3% |
| cup | 17.7% |
| mailbox | 16.9% |
| bus | 16.8% |
| tree | 16.7% |
| car | 16.7% |
| house | 16.0% |
| bread | 15.5% |
| face | 13.9% |

Confusion matrix: `confusion_matrix.png`. Example drawings: `samples.png`.
