This is the repository of the WWW 2024 industry track paper: Collaborative-Enhanced Prediction of Spending on Newly Downloaded Mobile Games under Consumption Uncertainty.

In this repository, we divide the code into two parts. One is the research structure to help the researchers accelerate their research. As the original private data can not be released, we take the steam dataset as an example. We have processed the original data from the website: https://steam.internet.byu.edu/ along with the scripts to process the dataset. Besides, we also provide the final pre-processed data(https://pan.baidu.com/s/1O_-BN5GvS_XI4fj4aGxwAA?pwd=ubki) we used in the experimental section. One thing should be noticed that is, we don't introduce any details about this dataset in the paper. 
The following is the overall structure of the proposed research structure:
1. entry.py is the entry script. You can execute the following code python entry.py --model=PMF --config=xxx to start the training process
2. train.py is used to train the specific model
3. config is a directory, which stores the configure parameters of all mdoels and corresponding dataset
4. models store the pytorch-based neural network structures of different models 
5. data_process.py is used to process the experimental datset. 
6. evaluations.py stores different kinds of metrics

And another is the colab_ltv.ipynb file, to implement the collaborative enhaned spending money prediction model in the private data.