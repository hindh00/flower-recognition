# Evaluation results

- **Top-1 test accuracy**: 0.9618
- **Top-5 test accuracy**: 0.9959

## Most confused class pairs

| Count | True class | Predicted as |
|---|---|---|
| 2 | snapdragon | sweet pea |
| 1 | wild pansy | columbine |
| 1 | water lily | camellia |
| 1 | trumpet creeper | wallflower |
| 1 | sword lily | thorn apple |
| 1 | sword lily | corn poppy |
| 1 | sweet william | siam tulip |
| 1 | sweet william | great masterwort |
| 1 | sweet william | garden phlox |
| 1 | sweet william | azalea |
| 1 | sweet pea | toad lily |
| 1 | sweet pea | cape flower |
| 1 | snapdragon | wallflower |
| 1 | siam tulip | lotus lotus |
| 1 | prince of wales feathers | bee balm |

## Full classification report

```
                           precision    recall  f1-score   support

            pink primrose       1.00      1.00      1.00         6
hard-leaved pocket orchid       1.00      1.00      1.00         9
         canterbury bells       1.00      0.83      0.91         6
                sweet pea       0.75      0.75      0.75         8
         english marigold       1.00      0.78      0.88         9
               tiger lily       1.00      1.00      1.00         6
              moon orchid       0.86      1.00      0.92         6
         bird of paradise       0.93      1.00      0.96        13
                monkshood       0.86      0.86      0.86         7
            globe thistle       0.86      1.00      0.92         6
               snapdragon       0.91      0.77      0.83        13
              colt's foot       0.93      1.00      0.96        13
              king protea       1.00      1.00      1.00         7
            spear thistle       0.88      1.00      0.93         7
              yellow iris       1.00      1.00      1.00         7
             globe-flower       1.00      1.00      1.00         6
        purple coneflower       1.00      1.00      1.00        13
            peruvian lily       0.93      1.00      0.96        13
           balloon flower       1.00      1.00      1.00         7
    giant white arum lily       1.00      1.00      1.00         8
                fire lily       1.00      1.00      1.00         6
        pincushion flower       1.00      1.00      1.00         9
               fritillary       1.00      1.00      1.00        14
               red ginger       1.00      1.00      1.00         6
           grape hyacinth       0.83      0.83      0.83         6
               corn poppy       0.67      0.67      0.67         6
 prince of wales feathers       1.00      0.83      0.91         6
         stemless gentian       0.91      1.00      0.95        10
                artichoke       1.00      0.92      0.96        12
            sweet william       0.90      0.69      0.78        13
                carnation       1.00      0.62      0.77         8
             garden phlox       0.83      0.83      0.83         6
         love in the mist       1.00      1.00      1.00         7
            mexican aster       1.00      1.00      1.00         6
         alpine sea holly       1.00      1.00      1.00         6
     ruby-lipped cattleya       1.00      1.00      1.00        11
              cape flower       0.94      1.00      0.97        16
         great masterwort       0.89      1.00      0.94         8
               siam tulip       0.83      0.83      0.83         6
              lenten rose       1.00      1.00      1.00        10
           barbeton daisy       0.95      1.00      0.97        19
                 daffodil       1.00      1.00      1.00         9
               sword lily       1.00      0.90      0.95        20
               poinsettia       1.00      1.00      1.00        14
         bolero deep blue       0.80      0.67      0.73         6
               wallflower       0.94      1.00      0.97        30
                 marigold       1.00      1.00      1.00        10
                buttercup       0.91      0.91      0.91        11
              oxeye daisy       1.00      1.00      1.00         7
         common dandelion       1.00      0.93      0.96        14
                  petunia       1.00      0.97      0.99        39
               wild pansy       1.00      0.92      0.96        13
                  primula       1.00      1.00      1.00        14
                sunflower       1.00      1.00      1.00         9
              pelargonium       1.00      1.00      1.00        11
       bishop of llandaff       1.00      1.00      1.00        17
                    gaura       1.00      1.00      1.00        10
                 geranium       0.94      1.00      0.97        17
            orange dahlia       1.00      1.00      1.00        10
       pink-yellow dahlia       1.00      1.00      1.00        17
         cautleya spicata       1.00      1.00      1.00         7
         japanese anemone       1.00      0.88      0.93         8
         black-eyed susan       1.00      1.00      1.00         8
               silverbush       1.00      1.00      1.00         8
        californian poppy       1.00      1.00      1.00        16
             osteospermum       0.90      1.00      0.95         9
            spring crocus       0.86      1.00      0.92         6
             bearded iris       1.00      1.00      1.00         8
               windflower       1.00      1.00      1.00         8
               tree poppy       1.00      1.00      1.00         9
                  gazania       1.00      1.00      1.00        12
                   azalea       0.93      0.93      0.93        15
               water lily       0.97      0.97      0.97        29
                     rose       0.96      1.00      0.98        26
              thorn apple       0.95      1.00      0.97        18
            morning glory       0.94      1.00      0.97        16
           passion flower       1.00      1.00      1.00        38
              lotus lotus       0.91      1.00      0.95        21
                toad lily       0.86      1.00      0.92         6
                anthurium       1.00      1.00      1.00        16
               frangipani       1.00      1.00      1.00        25
                 clematis       1.00      1.00      1.00        17
                 hibiscus       1.00      0.90      0.95        20
                columbine       0.92      0.92      0.92        13
              desert-rose       1.00      0.89      0.94         9
              tree mallow       1.00      1.00      1.00         8
                 magnolia       1.00      0.89      0.94         9
                 cyclamen       1.00      1.00      1.00        23
               watercress       1.00      1.00      1.00        28
               canna lily       1.00      0.77      0.87        13
              hippeastrum       0.86      1.00      0.92        12
                 bee balm       0.90      0.90      0.90        10
                ball moss       1.00      1.00      1.00         7
                 foxglove       0.96      0.96      0.96        25
            bougainvillea       1.00      1.00      1.00        19
                 camellia       0.80      0.86      0.83        14
                   mallow       0.77      1.00      0.87        10
          mexican petunia       1.00      1.00      1.00        13
                 bromelia       1.00      1.00      1.00         9
           blanket flower       1.00      1.00      1.00         7
          trumpet creeper       0.88      0.88      0.88         8
          blackberry lily       1.00      1.00      1.00         7

                 accuracy                           0.96      1229
                macro avg       0.96      0.95      0.95      1229
             weighted avg       0.96      0.96      0.96      1229
```

![Confusion matrix](images/confusion_matrix.png)
