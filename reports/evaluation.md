# Evaluation report

Training examples: 548 across 16 intents. Held-out test phrases: 130. NLTK preprocessing active: True.

## Baseline vs improved

| Pipeline | 5-fold CV accuracy | Held-out accuracy | Held-out macro F1 |
|---|---|---|---|
| Baseline (lowercase + word TF-IDF) | 65.5% | 83.1% | 0.836 |
| Improved (NLTK stem + word & char TF-IDF) | 77.2% | 98.5% | 0.984 |

Confidence threshold 0.3: 2 of 130 held-out phrases would be routed to retrieval/fallback.

## Per-intent report (held-out, improved pipeline)

```
                 precision    recall  f1-score   support

   bot_identity       1.00      0.88      0.93         8
   capabilities       0.89      1.00      0.94         8
        dataset       1.00      1.00      1.00         8
     evaluation       1.00      1.00      1.00         8
        goodbye       1.00      1.00      1.00         8
       greeting       0.89      1.00      0.94         8
    how_are_you       1.00      0.88      0.93         8
intent_response       1.00      1.00      1.00         8
      interface       1.00      1.00      1.00         8
           joke       1.00      1.00      1.00         8
      libraries       1.00      1.00      1.00         8
 nlp_definition       1.00      1.00      1.00         8
   out_of_scope       1.00      1.00      1.00        10
  preprocessing       1.00      1.00      1.00         8
         thanks       1.00      1.00      1.00         8
       training       1.00      1.00      1.00         8

       accuracy                           0.98       130
      macro avg       0.99      0.98      0.98       130
   weighted avg       0.99      0.98      0.98       130

```

## Confusion matrix (rows = true, columns = predicted)

```
labels: bot_identity, capabilities, dataset, evaluation, goodbye, greeting, how_are_you, intent_response, interface, joke, libraries, nlp_definition, out_of_scope, preprocessing, thanks, training
[[ 7  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  8  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  8  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  8  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  8  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  8  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  1  7  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  8  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  8  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  8  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  8  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  8  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0 10  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  8  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  8  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  8]]
```

## Misclassified held-out phrases (2)

- "is all good with you bot": expected `how_are_you`, got `greeting` (35%)
- "what do i call you": expected `bot_identity`, got `capabilities` (39%)