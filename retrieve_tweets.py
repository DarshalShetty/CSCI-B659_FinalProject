from credentials import consumer_key, consumer_secret, access_token, access_token_secret
import tweepy as tw
import csv

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

# Define the search term and the date_since date as variables
search_words = '''
MKR OR "asian drive" OR feminazi OR immigrant OR nigger OR sjw OR WomenAgainstFeminism OR 
blameonenotall OR "islam terrorism" OR notallmen OR victimcard OR "victim card" OR "arab terror" OR 
"gamergate" OR jsil OR racecard OR "race card" -filter:retweets
'''
date_since = "2021-01-01"

# Collect tweets
tweets = tw.Cursor(api.search,
                   q=search_words,
                   lang="en",
                   tweet_mode="extended",
                   since=date_since).items(1000)

# Collect a list of tweets
tweetlist = [[tweet.user.screen_name, tweet.full_text, tweet.id] for tweet in tweets]

with open('Data/tweets.csv', 'w') as out_csv:
    w = csv.writer(out_csv)
    w.writerow(["user.screen_name", "full_text", "id"])
    for tweet in tweetlist:
        w.writerow(tweet)
