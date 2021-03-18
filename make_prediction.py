import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression

# from tdi_read_to_pandas import read_linkedin_to_pandas, read_facebook_to_pandas, read_stock_to_pandas


def clean_currency(x):
    """ If the value is a string, then remove currency symbol and delimiters
    otherwise, the value is numeric and can be converted
    """
    if isinstance(x, str):
        return(x.replace('$', '').replace(',', ''))
    return(x)


def select_LinkedIn_data(lk_df, company_str, year):
    st = datetime.datetime(year-1, 12, 31, tzinfo=datetime.timezone.utc)
    ed = datetime.datetime(year+1, 1, 1, tzinfo=datetime.timezone.utc)

    lk_dt_df = lk_df.loc[(lk_df['company_name'] == company_str)
                         & (lk_df['as_of_date'] > st)
                         & (lk_df['as_of_date'] < ed)
                         ]
    return lk_dt_df.groupby(lk_dt_df['as_of_date'].dt.strftime('%m'))['followers_count'].mean().sort_index() 
    


# def make_LinkedIn_Plot(lk_dt_df):
#     plt.scatter(lk_dt_df['as_of_date'], lk_dt_df['followers_count'])
#     plt.show()
#     variable = input('input something!: ')

#     return 


def select_FB_data(fb_df, company_str, year):
    start = datetime.datetime(year-1, 12, 31, tzinfo=datetime.timezone.utc)
    end = datetime.datetime(year+1, 1, 1, tzinfo=datetime.timezone.utc)

    fb_dt_df = fb_df.loc[(fb_df['username'] == company_str)
                         & (fb_df['time'] > start)
                         & (fb_df['time'] < end)
                         ]
    return fb_dt_df.groupby(fb_dt_df['time'].dt.strftime('%m'))['likes'].mean().sort_index()


# def make_FB_Plot(fb_df):

#     plt.scatter(fb_dt_df['time'], fb_dt_df['likes'])
#     plt.show()
#     variable = input('input something!: ')

#     return fb_dt_df.groupby(fb_dt_df['time'].dt.strftime('%m'))['likes'].mean().sort_values()
#     # %m


# year = 2017
def select_stock_data(st_df, year):
    st = datetime.datetime(year-1, 12, 31, tzinfo=datetime.timezone.utc)
    ed = datetime.datetime(year+1, 1, 1, tzinfo=datetime.timezone.utc)

    st_dt_df = st_df.loc[(st_df['timestamp'] > st)
                         & (st_df['timestamp'] < ed)
                         ]

    return st_dt_df.groupby(st_dt_df['timestamp'].dt.strftime('%m'))['open'].mean().sort_index()


### a scatter plot can be made when debugging
def make_stock_plot(st_df, year):
    st = datetime.datetime(year-1, 12, 31, tzinfo=datetime.timezone.utc)
    ed = datetime.datetime(year+1, 1, 1, tzinfo=datetime.timezone.utc)

    st_dt_df = st_df.loc[(st_df['timestamp'] > st)
                         & (st_df['timestamp'] < ed)
                         ]
    plt.plot(st_dt_df['timestamp'], st_dt_df['open'])
    plt.show()
    variable = input('input something!: ')

    return 


def get_company_str(company_symbol):
    df = pd.read_csv('./nasdaq100_lk_fb_names.csv')

    # if df['Symbol'].str.contains(company_symbol).any():
    # use list here as the above statement will let part of the string pass as well, like "AM", rather than AMD
    if company_symbol in df['Symbol'].tolist():
        lk_bool = pd.isnull(df.loc[df.Symbol == company_symbol, 'LK']).bool()
        fb_bool = pd.isnull(df.loc[df.Symbol == company_symbol, 'FB']).bool()
        if lk_bool or fb_bool:
            return None, None
        else:
            return df.loc[df.Symbol == company_symbol, 'LK'].item(), df.loc[df.Symbol == company_symbol, 'FB'].item() 
    else:
        return None, None
            

def data_for_prediction(stock_company_symbol, fb_company_str, lk_company_str, year):
    print('Print stock_company_symbol, fb_company_str, lk_company_str:')
    print(stock_company_symbol, fb_company_str, lk_company_str)

    linkedin_dir = './datalab/datalab_linkedin_cleaned_more.csv'
    lk_df = pd.read_csv(linkedin_dir)
    # convert date to UTC
    lk_df['as_of_date']= pd.to_datetime(lk_df['as_of_date'], utc = True)

    lk_month_df = select_LinkedIn_data(lk_df, lk_company_str, year)

    # make_LinkedIn_Plot(lk_month_df[1:-1])
    
    facebook_dir = './datalab/datalab_facebook_cleaned_more.csv'
    fb_df = pd.read_csv(facebook_dir)
    # convert date to UTC
    fb_df['time']= pd.to_datetime(fb_df['time'], utc = True)
    fb_month_df = select_FB_data(fb_df, fb_company_str, year)

    # this line has selected stock company
    stock_dir = './nasdaq100/' + stock_company_symbol + '.csv'
    st_df = pd.read_csv(stock_dir)
    st_df['timestamp']= pd.to_datetime(st_df['timestamp'], utc = True)
    st_df['open'] = st_df['open'].apply(clean_currency).astype('float')
    ### when debugging the algorithm, uncomment the following line
    # make_stock_plot(st_df)
    st_month_df = select_stock_data(st_df, year+1)

    return lk_month_df, fb_month_df, st_month_df



def data_next_for_prediction(fb_company_str, lk_company_str, year):

    linkedin_dir = './datalab/datalab_linkedin_cleaned_more.csv'
    lk_df = pd.read_csv(linkedin_dir)
    # convert date to UTC
    lk_df['as_of_date']= pd.to_datetime(lk_df['as_of_date'], utc = True)
    lk_month_df = select_LinkedIn_data(lk_df, lk_company_str, year)
    
    facebook_dir = './datalab/datalab_facebook_cleaned_more.csv'
    fb_df = pd.read_csv(facebook_dir)
    # convert date to UTC
    fb_df['time']= pd.to_datetime(fb_df['time'], utc = True)
    fb_month_df = select_FB_data(fb_df, fb_company_str, year)

    return lk_month_df, fb_month_df


def stock_next_for_validation(stock_company_symbol, year):
    # this line has selected stock company
    stock_dir = './nasdaq100/' + stock_company_symbol + '.csv'
    st_df = pd.read_csv(stock_dir)
    st_df['timestamp']= pd.to_datetime(st_df['timestamp'], utc = True)
    st_df['open'] = st_df['open'].apply(clean_currency).astype('float')
    ### when debugging the algorithm, uncomment the following line
    # make_stock_plot(st_df)
    st_month_df = select_stock_data(st_df, year)

    return st_month_df.tolist()


def make_prediction(lk_month_df, fb_month_df, st_month_df, lk_month_df_next, fb_month_df_next):
    x_merged = lk_month_df.to_frame().join(fb_month_df.to_frame())    # print (x_merged)
    X = x_merged
    print('X for prediction:')
    print (X)
    y = st_month_df
    
    lr = LinearRegression()  # make an instance of the model 
    lr.fit(X, y)             # fit the model
    r2_str = str(round(lr.score(X, y), 2))
    print('fit r2_score: ', r2_str )
    print('fit coef_: ', lr.coef_)

    # get X_next from historical data
    x_next_merged = lk_month_df_next.to_frame().join(fb_month_df_next.to_frame())
    X_next = x_next_merged
    
    # feed to make prediction for next year
    y_pred = lr.predict(X_next)

    # fit y_pred to get ratings
    month_list = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
    slope, intercept = np.polyfit(month_list, y_pred, 1)
    print('slope of y_pred: ', slope)
    
    rating = None

    if slope < -0.1:
        rating = "Decreasing"
    elif slope > 0.1: 
        rating = "Increasing"
    else:
        rating = "Remaining same"

    return y_pred.tolist(), rating, r2_str
    
    # plt.plot(X, y, 'o', color = 'k', label='training data')
    # plt.plot(X, y_pred, color='#42a5f5ff', label='model prediction')
    # plt.xlabel('Total square feet of above ground living area')
    # plt.ylabel('Home price ($)')
    # plt.legend();
    # plt.show()


def compare_predict_true(y_pred, y_true, symbol):
    x_month = [i for i in range(1, 13)]

    plt.plot(x_month, y_pred, '.', markersize=12, label='prediction')
    plt.plot(x_month, y_true, '.', markersize=12, label='true')
    plt.legend()

    # fig.suptitle('test title', fontsize=20)
    plt.xlabel('Month')#, fontsize=18)
    plt.ylabel('Price in $')#, fontsize=16)
    plt.title(symbol, fontdict = {'fontsize' : 16})

    plt.savefig(symbol + '.png')
    plt.clf()
    return 


def get_company_prediction(company_symbol):
    lk,fb = get_company_str(company_symbol)
    if lk and fb:
        # get data for train
        lk_month_df, fb_month_df, st_month_df = data_for_prediction(company_symbol, fb, lk, 2016)
        if lk_month_df.shape[0] == fb_month_df.shape[0] == st_month_df.shape[0]:
            # get data_next for prediction
            lk_month_df_next, fb_month_df_next = data_next_for_prediction(fb, lk, 2017)
            # check if data_next has error when using for prediction
            if lk_month_df_next.shape[0] == fb_month_df_next.shape[0]:
                y_pred, rating, r2_str = make_prediction(lk_month_df, fb_month_df, st_month_df, lk_month_df_next, fb_month_df_next)
                # ## added temperary
                # y_true = stock_next_for_validation(company_symbol, 2018)
                # compare_predict_true(y_pred, y_true, company_symbol)
                # ##
                return (y_pred, rating, r2_str, 2018)
            else:
                return None

        else:
            return None

    else:
        return None

def main():
    # symbol = input('input something!: ')
    # lk,fb = get_company_str(symbol)
    # if s1 and s2:
    #     print ('s1 is: ' + lk)
    #     print ('s2 is: ' + fb, type(fb))
    # else:
    #     print('lack of company social media information')

    lk_month_df, fb_month_df, st_month_df = data_for_prediction('MSFT', 'Microsoft', 'Microsoft')
    if lk_month_df.shape[0] == fb_month_df.shape[0] == st_month_df.shape[0]:
        make_prediction(lk_month_df, fb_month_df, st_month_df)
    else:
        print("lack of social media data")

if __name__ == "__main__":
    main()
