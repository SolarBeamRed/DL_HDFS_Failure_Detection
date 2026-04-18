# ANOMALY CAPTURE IN HDFS LOGS
#### Python tool to detect potential anomalous blocks through regex parsing and LSTM based sequence analysis of events in the log  file<br>

The repo contains the entire modular project, which includes training scripts, notebooks used, tuning reports,
 relevant figures, model configuration, checkpoints as well as the final trained model. Model can directly be used for inference from main.py<br> 

Results:<br>
**ROC-AUC score on test data: 0.99998873**
```text
                  Classification Report                   
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┓
┃ Class        ┃ Precision ┃ Recall ┃ F1-score ┃ Support ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━┩
│ 0.0          │ 1.00      │ 1.00   │ 1.00     │ 111645  │
│ 1.0          │ 1.00      │ 0.99   │ 1.00     │ 3368    │
│ macro avg    │ 1.00      │ 1.00   │ 1.00     │ 115013  │
│ weighted avg │ 1.00      │ 1.00   │ 1.00     │ 115013  │
└──────────────┴───────────┴────────┴──────────┴─────────┘
```


___

## Installation and Usage
If inference is the only objective, and you do not want to train the model, then log dataset need not be installed. Follow the below steps:
1. Clone repo and navigate into directory:
~~~
git clone https://github.com/SolarBeamRed/DL_HDFS_Failure_Detection.git
cd DL_HDFS_Failure_Detection
~~~
2. Run the main python script
~~~
python main.py
~~~
3. Optionally run model summary to confirm presence of model by running option 3
4. Perform inference by choosing option 4 and providing path of log file<br><br>

If you also wish to train the model from scratch and run evaluation, then just run the "Train model"
 option from main.py. It will automatically handle downloading and placing data in the
 right place. You also have the option to download manually and place files as suggested in
 project structure below in this README, or src/utils.config.py
___

###  Available User Operations

#### 1. Save Anomalies as CSV
- Saves only anomalous blocks
- Includes:
  - `blk_id`
  - `anomaly_score`

---

#### 2. View Detected Anomalies
- Option to:
  - Display all anomalous blocks
  - Display top *N* anomalous blocks (sorted by severity)

---

#### 3. Save All Predictions
- Saves complete results to CSV
- Includes:
  - `blk_id`
  - `anomaly_score`
  - `prediction`

---

### Performance Features
- Batched inference using `DataLoader`
- GPU acceleration (if available)


---
___

##  About the project itself:
### TLDR
After learning some theory about RNNs and LSTMs, I wanted to implement them in Pytorch to try
 things out. Was split between this idea and another one revolving around DNA. Chose to stick to this idea because I thought
 it was more practical, and also parsing always looked cool to me. Preprocessed data was available, but I wanted to
 work on raw logs, so did not use the preprocessed data. Prolly spent more than half the time 
 on building clean pipelines for parsing and transforming raw log data. Then used LSTM to train on 
sequence of events for each block. Tried a baseline model, which already gave shockingly good results, 
then tuned hyperparameters for a bunch of trials, and ended up with a model that I am very satisfied with.
Everything after that was just structuring, writing .py scripts and improving visuals etc. Cool project, regex was quite fun

## Objective
**Build a tool for swift parsing and detection of anomalous blocks from HDFS logs:**
- Build robust pipeline for parsing raw log data into dataframes of blk_id and event sequences
- Perform preprocessing on dataframes to make them suitable for training and inference
- Build a powerful LSTM model to detect anomalies based on the event sequence of the block
- Tuned model using Optuna to obtain even better performance
- Provide user input inference out of the box

## Dataset
**HDFS_v1 log file from LOGPAI**
https://github.com/logpai/loghub/blob/master/HDFS#hdfs_v1<br>
The log file itself is 1.5GB in size, and contains 11,175,630 lines of text.
The corresponding labels for the log file are also available from the same source.
<br><br>
Note: Dataset is **not included** in the repo because of its large size. You don't have to download anything
manually though, scripts are present to automatically download the dataset and place it in the right directories
automatically when training model
___

## Approach
### General Approach
- Parse log file to obtain only useful information and clean out metadata
- Create dataframe from cleaned log file
- Apply necessary preprocessing on the dataframe to make it suitable for training and inference
- Build and observe performance on a baseline LSTM
- Tune hyperparameters using Optuna
-  Final training with early stopping and checkpointing.

### Parsing 
1. Extract tuples of blk_id and message from each useful line, [stored into a tuple]
2. Normalise messages to create templates. [Stored again in tuples of (blk_id, normalised_msg)]
3. Encode messages: Assign integer ID to each message according to its template. [Stored again in tuples of (blk_id, event_id)]
4. Put events of same blk_id together [Stored in a dictionary with keys as blk_id and values as list of events]
5. Create dataframe of above dictionary, and append label columns to the dataframe

### Preprocessing on Dataframe
Perform following transforms:<br>
1. Choose a max length, and truncate all sequences to fit max_length
2. Add 0 padding using torch.nn.utils.rnn.pad_sequence on X
3. Convert labels to torch tensors with dtype=torch.float32 

### Model
Bidirectional stacked LSTM with 2 layers. An embedding layer of embed_dim=128 is used
 before LSTM layers. <br>
Model Architecture Summary:
```text
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
TunedModel                               --
├─Embedding: 1-1                         7,040
├─LSTM: 1-2                              198,656
├─Linear: 1-3                            129
=================================================================
Total params: 205,825
Trainable params: 205,825
Non-trainable params: 0
=================================================================
```

### Training 
- Loss: BCEWithLogitsLoss
- Optimizer: Adam (tuned parameters)
- Early stopping based on validation loss  
- Model checkpointing (best model saved)<br>  
Training loop for the best model took around 20 minutes to complete on an RTX3050.
Early stopping was triggered at epoch 24, and the best epoch was epoch 14.

### Tuning
Used Optuna for hyperparameter tuning. Tuned the following parameters:
- Learning rate
- Weight decay
- Choice of optimizer (Adam vs RMSProp vs SGD)
- Beta values for Adam
- Embedding size
- Hidden state size of LSTMs
- Dropout rate<br><br>
Study consisted of 80 trials, and usage of MedianPruner() for pruning.
The entire study optimisation consumed around 8-10 minutes of time.

### Results
On test data, following results were observed:<br>
**ROC-AUC score: 0.99998873**
```text
                  Classification Report                   
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┓
┃ Class        ┃ Precision ┃ Recall ┃ F1-score ┃ Support ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━┩
│ 0.0          │ 1.00      │ 1.00   │ 1.00     │ 111645  │
│ 1.0          │ 1.00      │ 0.99   │ 1.00     │ 3368    │
│ macro avg    │ 1.00      │ 1.00   │ 1.00     │ 115013  │
│ weighted avg │ 1.00      │ 1.00   │ 1.00     │ 115013  │
└──────────────┴───────────┴────────┴──────────┴─────────┘
```

### Project structure
```text
project-root/
├── src/                     # Core training pipeline
│   ├── models/              # LSTM model architecture, training, evaluation
│   ├── data/                # Data preprocessing, encoding, sequence building
│   └── utils/               # Configs, helpers, constants
│
├── notebooks/               # Experimentation and hyperparameter tuning
├── reports/                 # Figures, Optuna study DB, results CSVs
├── configs/                 # Configuration files (config.json)
├── checkpoints/             # Saved models and training checkpoints
├── datasets/                # Dataset directory (auto-downloaded if missing)
│   └── HDFS_v1/
│       ├── HDFS.log
│       └── preprocessed/
│           └── anomaly_label.csv
│
├── main.py                  # Entry point (CLI menu)
├── requirements.txt         # Dependencies
└── README.md
```

### Reflections
Building pipelines was satisfying. Using Regex to parse raw unstructured log files felt pretty cool ngl. Definitely spent
 more time working on the pipeline than I did on the model itself. Building the Model was pretty cool, 
and I was genuinely surprised when I saw the baseline model surpassing 0.99 roc-auc score. 
I genuinely thought there must have been some form of leakage or serious overlooking of some structural
 decision, because I thought there was no way the model could be this good, not with the baseline at least.
 But even after extensive inspection, I could not really find any leakage or logical overlooking. Just
 to make sure, I shuffled training labels and checked the score. It came out to be 0.58, which is more or less what the score
 has to be on random labels, so yeah, was pleasantly surprised to learn how good LSTMs can be for this particular 
problem. After tuning, the model improved even more, and compared to CNN projects, this took
 way less time to tune and train as well, so there was less doing nothing while model was being trained.
 Overall, cool project, had good fun building pipelines and playing around with Regex. Rad project
