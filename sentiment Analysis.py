import tweepy           # Accessing Twitter's API
import pandas as pd     # To manipulate dataset
import numpy as np      # Used for statistical computation

# For plotting and visualization:
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns


consumer_key = '3s2IdCz9jX2Z6pej0dtebKrgl'
consumer_secret = '5604qFrlBpIcZblQpo3iKGReFgYC9ZltdhSOJedS1gc0pwzLH7'
access_secret = 'lHiG1xT6yJsuevCF9BVYnrDgNM4f1zsTHhfKItdMoCsKk'
access_token = '723396400206114816-YIt4W7Rj2zKbT60tyWT3Ex8byWnlSLR'


def twitter_setup():

    # Authentication and access using keys:
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)

    # Return API with authentication:
    api = tweepy.API(auth)
    return api
	
extractor = twitter_setup()

# We create a tweet list as follows:
tweets = extractor.user_timeline(screen_name="zomato", count=198)
print("Total tweets extracted: {}.\n".format(len(tweets)))

# We print the most recent 5 tweets:
print("5 recent tweets:\n")
for tweet in tweets[:5]:
    print(tweet.text)
    print()	
	

data = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])
display(data.head(10))

data['len']  = np.array([len(tweet.text) for tweet in tweets])
data['ID']   = np.array([tweet.id for tweet in tweets])
data['Date'] = np.array([tweet.created_at for tweet in tweets])
data['Source'] = np.array([tweet.source for tweet in tweets])
data['Likes']  = np.array([tweet.favorite_count for tweet in tweets])
data['RTs']    = np.array([tweet.retweet_count for tweet in tweets])

fav_max = np.max(data['Likes'])
rt_max  = np.max(data['RTs'])

fav = data[data.Likes == fav_max].index[0]
rt  = data[data.RTs == rt_max].index[0]

# Max FAVs:
print("Most Liked Tweet: \n{}".format(data['Tweets'][fav]))
print("Total Number of Tweets: {}".format(fav_max))
print("{} characters.\n".format(data['len'][fav]))

# Max RTs:
print("Most re-tweeted Tweet: \n{}".format(data['Tweets'][rt]))
print("Total Number of re-tweets: {}".format(rt_max))
print("{} characters.\n".format(data['len'][rt]))

from textblob import TextBlob
import re

def clean_tweet(tweet):
    '''
    Utility function to clean the text in a tweet by removing 
    links and special characters using regex.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def analize_sentiment(tweet):

    sentiment_analysis = TextBlob(clean_tweet(tweet))
    if sentiment_analysis.sentiment.polarity > 0:
        return 1
    elif sentiment_analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1
		
# We create a column with the result of the analysis:
data['SA'] = np.array([ analize_sentiment(tweet) for tweet in data['Tweets'] ])

# We display the updated dataframe with the new column:
display(data.head(10))


positive_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] > 0]
neutral_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] == 0]
negative_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] < 0]


print("Percentage of positive tweets: {}%".format(len(positive_tweets)*100/len(data['Tweets'])))
print("Percentage of neutral tweets: {}%".format(len(neutral_tweets)*100/len(data['Tweets'])))
print("Percentage of negative tweets: {}%".format(len(negative_tweets)*100/len(data['Tweets'])))		