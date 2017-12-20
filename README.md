# CNN for sentence classification

This example demonstrates the use of Conv1D for CNN text classification. Original paper could be found at: https://arxiv.org/abs/1408.5882

This is the baseline model: CNN-rand, on MR dataset.

The model is implemented in two frameworks:

- cnn_mxnet.py: MXNET/Gluon API
- cnn_pytorch: PyTorch

We didn't implement cross validation, but simply run python mr_cnn.py for multiple times, the average accuracy is close to 76%.

#### MXNET Result:

```bash
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
Epoch   1, Train_loss:    0.42, Train_acc 83.12%, Test_loss:   0.53, Test_acc 72.73%, Time: 0:00:29 *
Epoch   2, Train_loss:    0.22, Train_acc 93.63%, Test_loss:   0.48, Test_acc 77.23%, Time: 0:00:56 *
Epoch   3, Train_loss:    0.11, Train_acc 96.73%, Test_loss:   0.59, Test_acc 77.13%, Time: 0:01:23
Epoch   4, Train_loss:   0.048, Train_acc 98.90%, Test_loss:   0.75, Test_acc 76.10%, Time: 0:01:51
Epoch   5, Train_loss:   0.021, Train_acc 99.66%, Test_loss:   0.91, Test_acc 75.45%, Time: 0:02:18
Epoch   6, Train_loss:   0.011, Train_acc 99.87%, Test_loss:    1.1, Test_acc 76.01%, Time: 0:02:46
Epoch   7, Train_loss:   0.006, Train_acc 99.95%, Test_loss:    1.2, Test_acc 76.10%, Time: 0:03:13
Epoch   8, Train_loss:  0.0036, Train_acc 99.97%, Test_loss:    1.3, Test_acc 76.29%, Time: 0:03:41
Epoch   9, Train_loss:  0.0024, Train_acc 99.98%, Test_loss:    1.4, Test_acc 76.38%, Time: 0:04:08
Epoch  10, Train_loss:  0.0019, Train_acc 99.98%, Test_loss:    1.5, Test_acc 76.38%, Time: 0:04:36
Testing...
Test accuracy:  77.23%, F1-Score:  77.21%
Precision, Recall and F1-Score...
             precision    recall  f1-score   support

        POS       0.78      0.76      0.77       525
        NEG       0.77      0.79      0.78       542

avg / total       0.77      0.77      0.77      1067

Confusion Matrix...
[[397 128]
 [115 427]]
Time usage: 0:00:01
POS
NEG
```

#### PyTorch Result:

```bash
Loading data...
Training: 9595, Testing: 1067, Vocabulary: 8000
Configuring CNN model...
TextCNN(
  (embedding): Embedding(8000, 128)
  (convs): ModuleList(
    (0): Conv1d (128, 100, kernel_size=(3,), stride=(1,))
    (1): Conv1d (128, 100, kernel_size=(4,), stride=(1,))
    (2): Conv1d (128, 100, kernel_size=(5,), stride=(1,))
  )
  (dropout): Dropout(p=0.5)
  (fc1): Linear(in_features=300, out_features=2)
)
Training and evaluating...
Epoch   1, Train_loss:    0.57, Train_acc 76.20%, Test_loss:   0.63, Test_acc 65.42%, Time: 0:00:09 *
Epoch   2, Train_loss:    0.46, Train_acc 83.75%, Test_loss:    0.6, Test_acc 67.01%, Time: 0:00:15 *
Epoch   3, Train_loss:    0.35, Train_acc 87.68%, Test_loss:   0.58, Test_acc 69.35%, Time: 0:00:21 *
Epoch   4, Train_loss:    0.28, Train_acc 89.46%, Test_loss:   0.61, Test_acc 69.54%, Time: 0:00:27 *
Epoch   5, Train_loss:    0.23, Train_acc 90.62%, Test_loss:   0.66, Test_acc 70.29%, Time: 0:00:33 *
Epoch   6, Train_loss:    0.13, Train_acc 96.47%, Test_loss:   0.63, Test_acc 73.01%, Time: 0:00:39 *
Epoch   7, Train_loss:   0.069, Train_acc 98.83%, Test_loss:   0.63, Test_acc 73.48%, Time: 0:00:45 *
Epoch   8, Train_loss:    0.05, Train_acc 99.33%, Test_loss:   0.69, Test_acc 73.66%, Time: 0:00:51 *
Epoch   9, Train_loss:   0.031, Train_acc 99.74%, Test_loss:   0.71, Test_acc 74.51%, Time: 0:00:56 *
Epoch  10, Train_loss:   0.027, Train_acc 99.66%, Test_loss:   0.79, Test_acc 74.13%, Time: 0:01:02
Epoch  11, Train_loss:   0.019, Train_acc 99.78%, Test_loss:   0.82, Test_acc 74.60%, Time: 0:01:08 *
Epoch  12, Train_loss:   0.014, Train_acc 99.91%, Test_loss:   0.85, Test_acc 73.66%, Time: 0:01:14
Epoch  13, Train_loss:   0.013, Train_acc 99.90%, Test_loss:   0.94, Test_acc 74.70%, Time: 0:01:20 *
Epoch  14, Train_loss:   0.013, Train_acc 99.86%, Test_loss:    1.0, Test_acc 74.04%, Time: 0:01:26
Epoch  15, Train_loss:  0.0089, Train_acc 99.94%, Test_loss:    1.0, Test_acc 74.98%, Time: 0:01:32 *
Epoch  16, Train_loss:  0.0066, Train_acc 99.98%, Test_loss:    1.0, Test_acc 73.95%, Time: 0:01:38
Epoch  17, Train_loss:  0.0057, Train_acc 99.99%, Test_loss:    1.1, Test_acc 75.45%, Time: 0:01:44 *
Epoch  18, Train_loss:  0.0051, Train_acc 100.00%, Test_loss:    1.1, Test_acc 75.16%, Time: 0:01:50
Epoch  19, Train_loss:  0.0053, Train_acc 99.97%, Test_loss:    1.2, Test_acc 75.45%, Time: 0:01:56
Epoch  20, Train_loss:   0.005, Train_acc 99.95%, Test_loss:    1.3, Test_acc 75.45%, Time: 0:02:02
Testing...
Test accuracy:  75.45%, F1-Score:  75.37%
Precision, Recall and F1-Score...
             precision    recall  f1-score   support

        POS       0.77      0.76      0.77       568
        NEG       0.73      0.75      0.74       499

avg / total       0.75      0.75      0.75      1067

Confusion Matrix...
[[432 136]
 [126 373]]
Time usage: 0:00:00
POS
NEG
```

The result of MXNET is better than the PyTorch version, and it converges faster.
