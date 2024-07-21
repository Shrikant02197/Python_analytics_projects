import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pandas import json_normalize
import datetime
month_order = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']
day_order = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
df = pd.read_csv("C:\\Users\\LENOVO\\OneDrive\\Documents\\Data Science\\Projects & Assignmets\\Files\\data\\ted_main.csv")
print(df.columns)

#I'm just going to reorder the columns in the order I've
#listed the features for my convenience (and OCD).
df = df[['name','title','description','main_speaker','speaker_occupation','film_date','num_speaker','duration',
         'published_date','comments','tags','languages','ratings','related_talks','url','views','event']]
#Before we go any further, let us convert the Unix timestamps
#into a human readable format.
df['film_date'] = df['film_date'].apply(lambda x:
                datetime.datetime.fromtimestamp( int(x)).strftime('%d-%m-%Y'))
df['published_date'] = df['published_date'].apply(lambda x:
                datetime.datetime.fromtimestamp( int(x)).strftime('%d-%m-%Y'))
print(df.head())

#Analysis 1: Most Viewed Talks of All Time
pop_talks = df[['title','main_speaker','views','film_date']].sort_values('views',ascending=True)[:15]
print(pop_talks)

#Analysis 2. Let us make a bar chart to visualise these 15 talks in terms of the number of views they garnered.
pop_talks['abbr'] = pop_talks['main_speaker'].apply(lambda x: x[:3])
sns.set_style('whitegrid')
plt.figure(figsize=(10,6))
sns.barplot(x='abbr',y='views', data=pop_talks)
plt.xlabel('Main Speaker (Abbreviated)')
plt.ylabel('Views')
plt.title('Top 15 Most Viewed TED Talks')
#plt.show()


#Analysis 3: let us investigate the summary statistics and the distibution of the views garnered on various TED Talks.
plt.figure(figsize=(10, 6))
sns.distplot(df['views'], bins=50, kde=False, label='All Talks')
sns.distplot(df[df['views'] > 0.4e7]['views'], bins=20, kde=False, label='Views > 4 Million')
plt.xlabel('Views')
plt.ylabel('Frequency')
plt.title('Distribution of Views on TED Talks')
plt.legend()
#plt.show()

#Analysis 4:
print(df.describe())

#Analysis 5: Performing textual analysis of comments

df['comments'].describe()
sns.distplot(df['comments'], bins=50, kde=False,label='All comments')
sns.distplot(df[df['comments'] < 500]['comments'], bins=20, kde=False, label='Comment less than 500')
#plt.show()

#Analysis 6: if the number of views is correlated with the number of comments. We should think that this is the case as
#more popular videos tend to have more comments.

sns.jointplot(x='views', y='comments', data=df)
#plt.show()

df[['views','comments']].corr()


#Analysis 7: Let us now check the number of views and comments on the 10 most commented TED Talks of all time
df_1 = df[['title','main_speaker','views','comments']].sort_values('comments',ascending=False).head(10)
print(df_1.iloc[:,1:])

#Analysis 8: #discussion quotient which is simply the ratio of the number of comments to the number of views
df['dis_quo'] = df['comments']/df['views']
df_2 = df[['title','main_speaker','views','comments','dis_quo','film_date']].sort_values('dis_quo',ascending=False).head(10)
print(df_2.iloc[:,0:])

###Analysing TED Talks by the month and the year
df['month'] = df['film_date'].apply(lambda x:
            month_order[int(x.split('-')[1]) - 1]) #x.split('-')[1] to get ['YYYY','MM','DD'] to (1 is for MM)
month_df = pd.DataFrame(df['month'].value_counts()).reset_index()
month_df.columns = ["month",'talks']
sns.barplot(x='month',y='talks',data=month_df,order=month_order)
#plt.show()


df_x = df[df['event'].str.contains('TEDx')]
x_month_df = pd.DataFrame(df_x['month'].value_counts()).reset_index()
x_month_df.columns = ['month','talks']
sns.barplot(x='month',y='talks',data=x_month_df,order=month_order)
#plt.show()

##the most popular days for conducting TED and TEDx conferences.
def getday(x):
    day, month, year = (int(i) for i in x.split('-'))
    answer = datetime.date(year, month, day).weekday()
    return day_order[answer]
df['day'] = df['film_date'].apply(getday)
day_df = pd.DataFrame(df['day'].value_counts()).reset_index()
day_df.columns = ['day','talk']
sns.barplot(x='day',y='talk',data=day_df,order=day_order)
#plt.show()
#Let us now visualize the number of TED talks through the years

df['year'] = df['film_date'].apply(lambda x: x.split('-')[2])
year_df = pd.DataFrame(df['year'].value_counts()).reset_index()
year_df.columns = ['year','talks']
plt.figure(figsize=(18,5))
sns.pointplot(x='year',y='talks',data=year_df)
#plt.show()

#let us construct a heat map that shows us the number of talks by month and year.
months = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5,
          'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11,
          'Dec': 12}
hmap_df = df.copy()
hmap_df['film_date'] = hmap_df['film_date'].apply(lambda x:
month_order[int(x.split('-')[1]) - 1] + " " + str(x.split('-')[2]))
hmap_df = pd.pivot_table(hmap_df[['film_date', 'title']],index='film_date', aggfunc='count').reset_index()
hmap_df['month_num'] = hmap_df['film_date'].apply(lambda x:months[x.split()[0]])
hmap_df['year'] = hmap_df['film_date'].apply(lambda x:x.split()[1])
hmap_df = hmap_df.sort_values(['year', 'month_num'])
hmap_df = hmap_df[['month_num', 'year', 'title']]
hmap_df = hmap_df.drop_duplicates(['month_num', 'title'])
hmap_df['year'] = hmap_df['year'].astype(int)
hmap_df = hmap_df.pivot(index='month_num', columns='title', values='year')
hmap_df = hmap_df.fillna(0)
f, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(hmap_df, annot=True, linewidths=.5, ax=ax,fmt='n', yticklabels=month_order)
#plt.show()
#TED- Speaker
speaker_df = df.groupby('main_speaker').count().reset_index()[['main_speaker','comments']]
speaker_df.columns = ['main_speaker','appearances']
speaker_df = speaker_df.sort_values('appearances',ascending=False)
speaker_df.head(10)
occupation_df = df.groupby('speaker_occupation').count().reset_index()[['speaker_occupation', 'comments']]
occupation_df.columns = ['occupation', 'appearances']
occupation_df = occupation_df.sort_values('appearances',ascending=False)
#Do some professions tend to attract a larger number of viewers?
f, ax = plt.subplots(nrows=1,ncols=1,figsize=(15,8))
sns.boxplot(x='speaker_occupation',y='views',
            data=df[df['speaker_occupation'].isin(occupation_df.head(10)['occupation'])], palette='muted', hue='speaker_occupation',ax=ax)
ax.set_ylim([0, 0.4e7])
#plt.show()
#Finally, let us check the number of talks which have had more than one speaker.
df['num_speaker'].value_counts()
df_3 = df[df['num_speaker'] == 5][['title', 'description','main_speaker', 'event']]
print(df_3)
#TED Events
#Which TED Events tend to hold the most number of TED.com upload worthy events?
event_df = df[['title','event']].groupby('event').count().reset_index()
event_df.columns = ['event','talks']
event_df = event_df.sort_values('talks',ascending=False)
df_4 = event_df.head(10)
print(df_4)

#TED Themes
import ast
df['tags'] = df['tags'].apply(lambda x: ast.literal_eval(x))
s = df.apply(lambda x: pd.Series(x['tags']),axis=1).stack().reset_index(level=1,drop=True)
s.name = 'theme'
theme_df = df.drop('tags',axis=1).join(s)
df_5 = theme_df.head()
len(theme_df['theme'].value_counts())
pop_themes = pd.DataFrame(theme_df['theme'].value_counts()).reset_index()
pop_themes.columns = ['theme','talks']
pop_themes.head(10)
plt.figure(figsize=(15,5))
sns.barplot(x='theme',y='talks',data=pop_themes.head(10))
#plt.show()

themes = list(pop_themes.head(8)['theme'])
themes.remove('TEDx')#pop_theme_talks =theme_df[theme_df['theme'].isin(pop_themes.head(10)['theme'])]

pop_theme_talks = theme_df[theme_df['theme'].isin(pop_themes.head(10)['theme'])]
ctab = pd.crosstab([pop_theme_talks['year']],pop_theme_talks['theme']).apply(lambda x: x/x.sum(),axis=1)
ctab[themes].plot(kind='bar',stacked=True,colormap='rainbow',figsize=(12,8)).legend(loc='center left',bbox_to_anchor=(1,0.5))
#plt.show()
fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(15, 8))
sns.boxplot(x='theme', y='views', data=pop_theme_talks, palette="muted", ax =ax)
ax.set_ylim([0, 0.4e7])
#Talk Duration and Word Counts¶
#Convert to minutes
df['duration'] = df['duration']/60
df['duration'].describe()
df[df['duration'] == 2.25]
df[df['duration'] == 87.6]
sns.jointplot(x='duration',y='views',data=df[df['duration'] < 25])
plt.xlabel('Duration')
plt.ylabel('Views')
plt.show()


