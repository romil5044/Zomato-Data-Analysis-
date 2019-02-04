import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# importing the data
proj= pd.read_csv("C:/Users/Romil Godha/Desktop/zomato/zomato.csv", encoding = "ISO-8859-1")

# Merging a dataset with country code file
Country = pd.read_excel('C:/Users/Romil Godha/Desktop/zomato/Country-Code.xlsx')
merger = pd.merge(proj, Country,on='Country Code')

  
# printing the rows and columns number
print('Number of rows and columns:', proj.shape )

# printing the data types of columns
print('Type of values', proj.dtypes)

# printing the name of columns
print('Column Labels', proj.columns.values.tolist())

# Taking only the necessary columns for analysis
proj_sub = pd.DataFrame(proj, columns=['Restaurant Name','City','Cuisines','Average Cost for two', 'Currency', 'Has Online delivery', 'Is delivering now', 'Has Table booking',  'Price range', 'Aggregate rating', 'Rating text', 'Votes', 'Longitude', 'Latitude'])

# Renaming columns
proj_sub.columns= ['Restaurant_Name','City','Cuisines','Average_Cost_for_two', 'Currency', 'Has_Online_delivery', 'Is_delivering_now', 'Has_Table_booking',  'Price_range', 'Aggregate_rating', 'Rating_text', 'Votes', 'Longitude', 'Latitude']

 # Removing aggregate rating with value equal to 0
proj_sub =proj_sub.loc[proj_sub['Aggregate rating']>0]

# printing the rows and columns number
print('Number of rows and columns:', proj_sub.shape )

# identifying the descriptive stats of data
proj_sub.describe()


#Grouping country code with data to find the countries with maximum restaurants 
#India has the most number of restaurants so our analysis was based on India
country_grp= (merger.groupby(['Country'], as_index=False)['Restaurant ID'].count())
country_grp.columns = ['Country', 'Number of Resturants']
country_grp= country_grp.sort_values(['Number of Resturants'],ascending=False)

# Graphing the number of restaurants by country
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.barplot(country_grp['Country'], country_grp['Number of Resturants'])
plt.xticks(rotation = 90)
plt.show()


# Identifying the best cities with good aggregate ratings 

wealthgroupmean = proj_sub.groupby(['City']).Aggregate_rating.mean()
result = wealthgroupmean.sort_values('Aggregate_rating',ascending=False)


# Plot of top 6 cities
ax1 = plt.subplot2grid((3,3),(2,0))
new = result['Aggregate_rating'].reset_index()
sns.barplot(x = 'City', y = 'Aggregate_rating', data = new.head(6), ax = ax1)
ax1.tick_params(axis='x', rotation=70)
ax1.set_title('Top 6 # City', size = 12)
ax1.set_ylim([0, new['Aggregate_rating'].head(1).values+5])
for i, val in enumerate(new['Aggregate_rating'].head(6)):
    ax.text(i, val+50, val, color = 'grey', ha = 'center')
plt.show()


#Identifying count of restaurants in top 6 cities around the world
# plot for top 6 cities
ax = plt.subplot2grid((3,3),(2,0))
cnt = proj_sub['City'].value_counts().reset_index()
cnt.rename(columns = {'index':'City', 'City':'cnt'}, inplace = True)
sns.barplot(x = 'City', y = 'cnt', data = cnt.head(6), ax = ax)
ax.tick_params(axis='x', rotation=70)
ax.set_title('Top 6 # City', size = 12)
ax.set_ylim([0, cnt['cnt'].head(1).values+500])
for i, val in enumerate(cnt['cnt'].head(6)):
    ax.text(i, val+50, val, color = 'grey', ha = 'center')
plt.show()

# What are the top 10 cuisines around the world
Cusines=(proj_sub.groupby(['Cuisines'], as_index=False)['Restaurant_Name'].count())
Cusines.columns = ['Cuisines', 'Number of Resturants']
Cusines['Mean Rating']=(proj_sub.groupby(['Cuisines'], as_index=False)['Aggregate_rating'].mean())['Aggregate_rating']
PopularCusines= Cusines.sort_values(['Number of Resturants'],ascending=False).head(10)

#India analysis by taking data for India only
ind= pd.read_csv("C:/Users/Romil Godha/Desktop/zomato/India.csv", encoding = "ISO-8859-1")

India = pd.DataFrame(ind, columns=['Restaurant Name','City','Cuisines','Average Cost for two', 'Currency', 'Has Online delivery', 'Is delivering now', 'Has Table booking',  'Price range', 'Aggregate rating', 'Rating text', 'Votes', 'Longitude', 'Latitude'])
India.columns= ['Restaurant_Name','City','Cuisines','Average_Cost_for_two', 'Currency', 'Has_Online_delivery', 'Is_delivering_now', 'Has_Table_booking',  'Price_range', 'Aggregate_rating', 'Rating_text', 'Votes', 'Longitude', 'Latitude']

# Which cities have the costliest food in India?
India_food=(India.groupby(['City'], as_index=False)['Average_Cost_for_two'].mean())
India_food.columns = ['City', 'Cost_of_food']
India_food= India_food.sort_values(['Cost_of_food'],ascending=False)

#Bar plot for costliest cities
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.barplot(India_food['City'], India_food['Cost_of_food'])
plt.xticks(rotation = 90)
plt.show()


# Which cities have the highest aggregate rating in India?
India_rating=(India.groupby(['City'], as_index=False)['Aggregate_rating'].mean())
India_rating.columns = ['City', 'Average Rating']
India_rating= India_rating.sort_values(['Average Rating'],ascending=False)

#Bar plot for costliest cities
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.barplot(India_rating['City'], India_rating['Average Rating'], color= 'blue')
plt.xticks(rotation = 90)
plt.show()

#The price range distribution for India
India_range=(India.groupby(['Price_range'], as_index=False)['Restaurant_Name'].count())
India_range.columns = ['Price range', 'Number of Restaurants']
India_range= India_rating.sort_values(['Number of Restaurants'],ascending=False)

#Bar plot for price range distribution
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.barplot(India_range['Price range'], India_range['Number of Restaurants'], color= 'grey')
plt.xticks(rotation = 90)
plt.show()




# What factors make a restaurant successful
# Machine learning 
df =pd.read_csv('India.csv')
df.head()
df1= df.drop(['Restaurant Name','Country Code','Address','Locality Verbose','Longitude','Latitude','Cuisines','Currency'],axis=1)
df1.head()

sns.pairplot(df1,hue='Rating text')

df1["Has Table booking"] = df1["Has Table booking"].astype('category')
df1["Has Table booking"] = df1["Has Table booking"].cat.codes
df1.head(5)

df1["Has Online delivery"] = df1["Has Online delivery"].astype('category')
df1["Has Online delivery"] = df1["Has Online delivery"].cat.codes

df1["Is delivering now"] = df1["Is delivering now"].astype('category')
df1["Is delivering now"] = df1["Is delivering now"].cat.codes

df1["Switch to order menu"] = df1["Switch to order menu"].astype('category')
df1["Switch to order menu"] = df1["Switch to order menu"].cat.codes

df1["Rating color"] = df1["Rating color"].astype('category')
df1["Rating color"] = df1["Rating color"].cat.codes

sns.pairplot(df1,hue='Rating text')

from sklearn.svm import SVC
model=SVC()
model.fit(X_train,y_train)
pred = model.predict(X_test)
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(x_train,y_train)
pred5 = knn.predict(X_test)
print(confusion_matrix(y_test,pred5))
print('\n')
print(classification_report(y_test,pred5))

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


