The MXNet implementation of CNN_sentence reached 79% of test accuracy after 3 epochs.


```python
Loading data...
Training: 9595, Testing: 1067, Vocabulary: 8000
Configuring CNN model...
Initializing weights on gpu(0)
TextCNN(
  (embedding): Embedding(8000 -> 128, float32)
  (conv1): Conv_Max_Pooling(
    (conv): Conv1D(None -> 100, kernel_size=(3,), stride=(1,))
    (pooling): GlobalMaxPool1D(size=(1,), stride=(1,), padding=(0,), ceil_mode=True)
  )
  (conv2): Conv_Max_Pooling(
    (conv): Conv1D(None -> 100, kernel_size=(4,), stride=(1,))
    (pooling): GlobalMaxPool1D(size=(1,), stride=(1,), padding=(0,), ceil_mode=True)
  )
  (conv3): Conv_Max_Pooling(
    (conv): Conv1D(None -> 100, kernel_size=(5,), stride=(1,))
    (pooling): GlobalMaxPool1D(size=(1,), stride=(1,), padding=(0,), ceil_mode=True)
  )
  (dropout): Dropout(p = 0.5)
  (fc1): Dense(None -> 2, linear)
)
Training and evaluating...
Epoch   1, Train_loss:    0.42, Train_acc 82.51%, Test_loss:   0.51, Test_acc 74.41%, Time: 0:00:29 *
Epoch   2, Train_loss:    0.23, Train_acc 92.80%, Test_loss:   0.46, Test_acc 77.98%, Time: 0:00:56 *
Epoch   3, Train_loss:    0.11, Train_acc 97.35%, Test_loss:   0.51, Test_acc 79.19%, Time: 0:01:23 *
Epoch   4, Train_loss:    0.05, Train_acc 98.86%, Test_loss:   0.63, Test_acc 78.63%, Time: 0:01:50
Epoch   5, Train_loss:   0.025, Train_acc 99.52%, Test_loss:   0.79, Test_acc 76.94%, Time: 0:02:17
Epoch   6, Train_loss:   0.012, Train_acc 99.80%, Test_loss:   0.93, Test_acc 76.94%, Time: 0:02:44
Epoch   7, Train_loss:   0.007, Train_acc 99.89%, Test_loss:    1.0, Test_acc 76.01%, Time: 0:03:11
Epoch   8, Train_loss:  0.0053, Train_acc 99.93%, Test_loss:    1.2, Test_acc 75.82%, Time: 0:03:38
Epoch   9, Train_loss:  0.0026, Train_acc 99.99%, Test_loss:    1.2, Test_acc 76.85%, Time: 0:04:05
Epoch  10, Train_loss:  0.0018, Train_acc 100.00%, Test_loss:    1.3, Test_acc 76.48%, Time: 0:04:32
Testing...
Test accuracy:  79.19%, F1-Score:  79.18%
Precision, Recall and F1-Score...
             precision    recall  f1-score   support

        POS       0.79      0.78      0.79       524
        NEG       0.79      0.80      0.80       543

avg / total       0.79      0.79      0.79      1067

Confusion Matrix...
[[408 116]
 [106 437]]
Time usage: 0:00:01
```
