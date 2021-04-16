# APPLYING ML TECHNIQUES IN CL - Final project: "Waseem" Revisited

## Problem statement
For this project, we will repeat the experiment by Waseem and Hovy (2016). All
groups who choose this project will jointly retrieve tweets from the last 2-3 
months using the hashtag list of Waseem and Hovy 
([see paper](https://www.aclweb.org/anthology/N16-2013.pdf)), and then each 
project member will annotate 150-200 tweets (depending on how many groups 
choose this project). We will then train and optimize (parameters, features) a
classifier of our choice on the Waseem data, and then classify the
new data. The working hypothesis is that the results will be lower than those 
reported by Waseem and Hovy. Then the next step will be to investigate two 
possible causes of the discrepancies (e.g., different levels of explicit abuse,
different distribution of original hashtags).

## Installation Steps

- Clone this repository
- Install the dependencies from ```requirements.txt``` in a virtual
 environment or globally using the command ```pip install -r requirements.txt```
- In order to fetch Tweets through Twitter API, a ```credentials.py``` file
  is needed and it must never be committed to git. The file needs the following
  variables with string values to be defined:
    - ```consumer_key``` (a.k.a. API Key)
    - ```consumer_secret``` (a.k.a. API Secret)
    - ```access_token```
    - ```access_token_secret```
    
## Scripts

### ```retrieve_tweets.py```
This script uses Twitter API to fetch all tweets matching a certain search
query (specified by ```search_query``` variable) within a time range
(specified by ```date_start``` and ```date_end``` variables). The fetched Tweets 
are stored in a CSV file in the ```Data ``` folder. A stratified sample of
these tweets will then be extracted for annotating as our test set . No 
arguments are needed to be passed while running this script. This script 
requires the ```credentials.py``` file mentioned in the installation section for
accessing the Twitter API.