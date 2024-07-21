import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
order_df = pd.read_csv("C:\\Users\\LENOVO\\OneDrive\\Documents\\OnlineRetail-master\\order_items.csv")
review_df = pd.read_csv("C:\\Users\\LENOVO\\OneDrive\\Documents\\OnlineRetail-master\\order_reviews.csv")
print(order_df.shape)
print(review_df.shape)
print(review_df.shape)
print([review_df.columns])
#Q1. Some customers didn't write a review. But why are they happy or dissatisfied?
blank_reviews = review_df['review_comment_message'].isnull() | (review_df['review_comment_message'] =='')
review_df['satisfied_or_not'] = (review_df['review_score'] >= 3) | (review_df['review_score'].isnull() & blank_reviews)
satisfaction_count = review_df['satisfied_or_not'].value_counts()

review_df['satisfied_or_not'] = review_df['satisfied_or_not'].apply(lambda x: 'Satified' if x else 'Not-satisfied')
#categorised based on rating score
print(satisfaction_count)
plt.figure(figsize=(20,15))
sns.countplot(data=review_df,x='satisfied_or_not',hue='satisfied_or_not')
plt.title('Customer Satisfaction')
plt.xlabel('Satisfaction')
plt.ylabel('Count')
plt.show()
"""

order_df = pd.read_csv("C:\\Users\\LENOVO\\OneDrive\\Documents\\OnlineRetail-master\\order_items.csv")
#Display all the column names
print(list(order_df.columns))

order_df = order_df[['order_id','product_id','price']]

prod_df = pd.read_csv("C:\\Users\\LENOVO\\OneDrive\\Documents\\OnlineRetail-master\\products.csv")
print(list(prod_df.columns))

prod_df = prod_df[['product_id','product_category_name']]

cat_df = pd.read_csv("C:\\Users\\LENOVO\\OneDrive\\Documents\\OnlineRetail-master\\product_category_name.csv")
print(list(cat_df.columns))

cat_df = cat_df.rename(columns={'1 product_category_name':'product_category_name',
                                '2 product_category_name_english':'product_category'})
print(list(cat_df.columns))

#Final Dataset - merge tab1 and tab2
data = pd.merge(order_df,prod_df,on='product_id',how='left')

data = pd.merge(data,cat_df,on='product_category_name',how='left')

print(list(data.columns))

for col in data.columns:
    pct_missing = np.mean(data[col].isnull())
    print(f"{col} - {round(pct_missing * 100)}%")
data['product_category'] = data['product_category'].fillna('Unknown')

print(f"Number of rows: \n\n order_items [{order_df.count()}], \n\n MergedData [{data.count()}]")

from matplotlib.ticker import PercentFormatter

df = data[['price','product_category']]
df.set_index(data['product_category'])

print(df.head(100))
# what is the most in demand product category?
sns.countplot(df['product_category'],order=df['product_category'].value_counts().index)
plt.title("Product Categories based on Demand".title(),fontsize=20)
plt.ylabel('Count'.title(),fontsize=14)
plt.xlabel('Product category'.title(),fontsize=14)
plt.xticks(rotation=90,fontsize=12)
plt.yticks(fontsize=5)
plt.show()

# Which categories generates high sales-Pareto
#Sort the values in the ascending order
quant_variable = df['price']
by_variable = df['product_category']

column = 'price'
group_by = 'product_category'

df = df.groupby(group_by)[column].sum().reset_index()
df = df.sort_values(by=column,ascending=True)
df['cumpercentage'] = df[column].cumsum()/df[column].sum()*100
fig, ax = plt.subplots(figsize=(20,5))
ax.bar(df[group_by],df[column],color='C0')
ax2 = ax.twinx()
ax2.plot(df[group_by],df['cumpercentage'],color='C1',marker="D",ms=7)
ax2.yaxis.set_major_formatter(PercentFormatter())
ax.tick_params(axis="x",rotation=90)
ax.tick_params(axis="y",colors="C0")
ax.tick_params(axis="y",colors="C1")
plt.title('Product Categories based on Sales'.title(),fontsize=20)
plt.show()

#Variation 2
# Plotting above graph with only top 40 categories, rest as other caterogies
total = quant_variable.sum()

df = df.groupby(group_by)[column].sum().reset_index()

df = df.sort_values(by=column,ascending=False)
df['cumpercentage'] = df[column].cumsum()/df[column].sum()*100
threshold = df[column].cumsum()/5 #20%

df_above_threshold = df[df['cumpercentage'] < threshold]
df = df_above_threshold
df_below_threshold = df[df['cumpercentage'] >= threshold]
sum = total - df[column].sum()
restbarcumsum = 100 - df_above_threshold['cumpercentage'].max()
rest = pd.Series(['OTHERS', sum, restbarcumsum], index=[group_by, column,'cumpercentage'])

df = pd.concat([df, rest],ignore_index=True)
#df.index = df[group_by]
df = df.sort_values(by='cumpercentage',ascending=True)
fig, ax = plt.subplots()
ax.bar(df.index, df[column],color='C0')
ax2 = ax.twinx()
ax2.plot(df.index, df['cumpercentage'], color='C1',marker="D",ms=7)
ax2.yaxis.set_major_formatter(PercentFormatter())
ax.tick_params(axis="x",color='C0',labelrotation=90)
ax.tick_params(axis="y",colors="C0")
ax.tick_params(axis="y",colors="C1")
plt.title('Product Categories based on Sales - 2'.title(),fontsize=20)
plt.show()

from scipy import stats

order_df = pd.read_csv("C:\\Users\\LENOVO\\OneDrive\\Documents\\OnlineRetail-master\\orders.csv")
print(list(order_df.columns))

X = pd.to_datetime(order_df['order_delivered_customer_date']) - pd.to_datetime(order_df['order_purchase_timestamp'])
print(X)

for i in range(0, len(X)):
    X[i] = X[i].days
plt.figure(figsize=(20,8))
sns.barplot(x=X.value_counts().sort_values(ascending=False).head(30).index,
            y=X.value_counts().sort_values(ascending=False).head(30).values)
plt.xlabel("Delivery Days")
plt.ylabel("Frequency")
plt.show()

info = X.describe()


print(f"Mean Value of Delivery Days: {np.mean(X):.1f}")
print(f"Median Value of Delivery Days: {np.median(X)}")
print(f"Mode Value of Delivery Days: {stats.mode(X)}")
print(f"Standard Deviation in Delivery Days: {X.std():.1f}")

order_df = pd.read_csv("C:\\Users\\LENOVO\\OneDrive\\Documents\\OnlineRetail-master\\orders.csv")
print(list(order_df.columns))

order_df['order_purchase_timestamp'] = pd.to_datetime(order_df['order_purchase_timestamp'])
order_df['order_delivered_customer_date'] = pd.to_datetime(order_df['order_delivered_customer_date'])

order_rev_df = pd.read_csv("C:\\Users\\LENOVO\\OneDrive\\Documents\\OnlineRetail-master\\order_reviews.csv")

print(list(order_rev_df.columns))
order_rev_df['review_creation_date'] = pd.to_datetime(order_rev_df['review_creation_date'])
order_rev_df['review_answer_timestamp'] = pd.to_datetime(order_rev_df['review_answer_timestamp'])

reviews = pd.merge(order_df, order_rev_df, on='order_id',how='left')
wehavecount = reviews['order_id'].count()

to_drop = ['review_id','order_id','customer_id','review_comment_title',
           'order_approved_at','order_delivered_carrier_date','order_estimated_delivery_date']
reviews.drop(columns=to_drop, inplace=True)

from datetime import datetime
sns.set()
COLOR_5S = '#0571b0'
COLOR_1S = '#ca0020'
REVIEWS_PALETTE = sns.color_palette((COLOR_1S, '#d57b6f','#c6c6c6', '#7f9abc', COLOR_5S))

sns.set_style('darkgrid',{'axes.facecolor':'#eeeeee'})
resize_plot = lambda: plt.gcf().set_size_inches(12,8)
p_5s = len(reviews[reviews['review_score'] == 5]) * 100/len(reviews)
p_1s = len(reviews[reviews['review_score'] == 1]) * 100/len(reviews)

first_dt = reviews['review_creation_date'].min()
last_dt = reviews['review_creation_date'].max()
avg_s = reviews['review_score'].mean()

print(len(reviews), 'reviews')
print('First:',first_dt)
print('Last:',last_dt)
print(f'5star: {p_5s:.1f}')
print(f'1star: {p_1s:.1f}')
print(f'Average: {avg_s:.1f}')

sns.catplot(x='review_score',
            kind='count',
            hue='review_score',
            data=reviews,
            palette=REVIEWS_PALETTE).set(xlabel='Review Score',ylabel='Number of Reviews');
plt.title('Score Distribution')
plt.show()

reviews['review_creation_delay'] = (reviews['review_creation_date'] - reviews['order_purchase_timestamp']).dt.days

sns.scatterplot(x='order_purchase_timestamp',
                y='review_creation_delay',
                hue='review_score',
                palette=REVIEWS_PALETTE,
                data=reviews).set(
                xlabel='Purchase Date',
                ylabel='Review Creation Delay (days)',
                xlim=(datetime(2016,8,1), datetime(2018, 12, 31))
);
resize_plot()
plt.title('Review Created Date compared to purchase date')
plt.show()

reviews['year_month'] = reviews['order_purchase_timestamp'].dt.to_period('M')
reviews_timeseries = reviews[reviews['review_creation_delay'] > 0].groupby('year_month')['review_score'].agg(['count','mean'])
ax = sns.lineplot(x=reviews_timeseries.index.to_timestamp(),
                  y='count',
                  data=reviews_timeseries,
                  color='#984ea3',
                  label='count')
ax.set(xlabel='Purchase Month',ylabel='Number of Reviews')
sns.lineplot(x=reviews_timeseries.index.to_timestamp(),
             y='mean',
             data=reviews_timeseries,
             ax=ax.twinx(),
             color='#ff7f00',
             label='mean').set(ylabel='Average Review Score');
resize_plot()
plt.title('Review group by Month')
plt.show()

reviews['review_length'] = reviews['review_comment_message'].str.len()
reviews[['review_score','review_length','review_comment_message']].head()

g = sns.FacetGrid(data=reviews, col='review_score',
                  hue='review_score',palette=REVIEWS_PALETTE)
g.map(plt.hist, 'review_length',bins=40)
g.set_xlabels('Comment Length')
g.set_ylabels('Number of Reviews')
plt.gcf().set_size_inches(12, 5)
plt.title("Size of the comments")
plt.show()

ax = sns.catplot(x='order_status',
                 kind='count',
                 hue='review_score',
                 data=reviews[reviews['order_status'] != 'delivered'],
                 palette=REVIEWS_PALETTE).set(
                xlabel="Order Status",
                ylabel="Number of Reviews"
);
plt.title('Order Status and customer rating')
resize_plot()
plt.show()

import unicodedata
import nltk
def remove_accents(text):
    return unicodedata.normalize('NFKD',text).encode('ascii',errors='ignore').decode('utf-8')

STOP_WORDS = set(remove_accents(w) for w in nltk.corpus.stopwords.words('portuguese'))
STOP_WORDS.remove('nao')

def comments_to_words(comment):
    lowered = comment.lower()
    normalized = remove_accents(lowered)
    tokens = nltk.tokenize.word_tokenize(normalized)
    words = tuple(t for t in tokens if t not in STOP_WORDS and t.isalpha())
    return words
def words_to_ngrams(words):
    unigrams, bigrams, trigrams = [], [], []
    for comment_words in words:
        unigrams.extend(comment_words)
        bigrams.extend(' '.join(bigram) for bigram in nltk.bigrams(comment_words))
        trigrams.extend(' '.join(trigram) for trigram in nltk.trigrams(comment_words))
    return unigrams, bigrams, trigrams

def plot_freq(tokens, color):
    resize_plot = lambda: plt.gcf().set_size_inches(12, 5)
    resize_plot()
    nltk.FreqDist(tokens).plot(25, cumulative=False, color=color)
sns.set()

COLOR_5S = '#0571b0'
COLOR_1S = '#ca0020'
REVIEWS_PALETTE = sns.color_palette((COLOR_1S, '#d57b6f','#c6c6c6', '#7f9abc', COLOR_5S))

sns.set_style('darkgrid',{'axes.facecolor':'#eeeeee'})
resize_plot = lambda: plt.gcf().set_size_inches(12, 5)

commented_reviews = reviews[reviews['review_comment_message'].notnull()].copy()
commented_reviews['review_comment_words'] = commented_reviews['review_comment_message'].apply(comments_to_words)

reviews_5s = commented_reviews[commented_reviews['review_score'] == 5]
reviews_1s = commented_reviews[commented_reviews['review_score'] == 1]

unigrams_5s, bigrams_5s, trigrams_5s = words_to_ngrams(reviews_5s['review_comment_words'])
unigrams_1s, bigrams_1s, trigrams_1s = words_to_ngrams(reviews_1s['review_comment_words'])

plot_freq(unigrams_5s, COLOR_5S)
plot_freq(bigrams_5s, COLOR_5S)
plot_freq(trigrams_5s, COLOR_5S)

plot_freq(unigrams_1s, COLOR_1S)
plot_freq(bigrams_1s, COLOR_1S)
plot_freq(trigrams_1s, COLOR_1S)