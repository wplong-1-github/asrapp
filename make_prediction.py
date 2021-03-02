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
    # year for train = 2016
    # year for tain = 2017
    print(lk_df,company_str)
    st = datetime.datetime(year-1, 12, 31, tzinfo=datetime.timezone.utc)
    ed = datetime.datetime(year+1, 1, 1, tzinfo=datetime.timezone.utc)

    lk_dt_df = lk_df.loc[(lk_df['company_name'] == company_str)
                         & (lk_df['as_of_date'] > st)
                         & (lk_df['as_of_date'] < ed)
                         ]
    """
    debug
    print (lk_dt_df[['as_of_date', 'followers_count']].head(20), '\n', lk_dt_df[['as_of_date', 'followers_count']].tail(20))
    print (lk_dt_df.groupby(lk_dt_df['as_of_date'].dt.strftime('%m'))['followers_count'].mean().sort_values() )

    plt.scatter(lk_dt_df['as_of_date'], lk_dt_df['followers_count'])
    plt.show()
    variable = input('input something!: ')
    """
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
    # print (fb_dt_df[['time', 'likes']].head(20), '\n', fb_dt_df[['time', 'likes']].tail(20))
    # print (fb_dt_df.groupby(fb_dt_df['time'].dt.strftime('%m'))['likes'].mean().sort_values() )

    # plt.scatter(fb_dt_df['time'], fb_dt_df['likes'])
    # plt.show()
    # variable = input('input something!: ')

    return fb_dt_df.groupby(fb_dt_df['time'].dt.strftime('%m'))['likes'].mean().sort_index()


# def make_FB_Plot(fb_df):

#     plt.scatter(fb_dt_df['time'], fb_dt_df['likes'])
#     plt.show()
#     variable = input('input something!: ')

#     return fb_dt_df.groupby(fb_dt_df['time'].dt.strftime('%m'))['likes'].mean().sort_values()
#     # %m


def select_stock_data(st_df):
    st = datetime.datetime(2016, 12, 31, tzinfo=datetime.timezone.utc)
    ed = datetime.datetime(2018, 1, 1, tzinfo=datetime.timezone.utc)

    st_dt_df = st_df.loc[(st_df['timestamp'] > st)
                         & (st_df['timestamp'] < ed)
                         ]
    # print(st_dt_df[' Open'].head())
    # print (st_dt_df[['Date', ' Open']].head(20), '\n', st_dt_df[['Date', ' Open']].tail(20))
    # print (st_dt_df.groupby(st_dt_df['Date'].dt.strftime('%m'))[' Open'].mean().sort_values() )

    # plt.scatter(st_dt_df['Date'], st_dt_df[' Open'])
    # plt.show()
    # variable = input('input something!: ')

    return st_dt_df.groupby(st_dt_df['timestamp'].dt.strftime('%m'))['open'].mean().sort_index()


def make_stock_plot(st_df):
    st = datetime.datetime(2016, 12, 31, tzinfo=datetime.timezone.utc)
    ed = datetime.datetime(2018, 1, 1, tzinfo=datetime.timezone.utc)

    st_dt_df = st_df.loc[(st_df['timestamp'] > st)
                         & (st_df['timestamp'] < ed)
                         ]
    plt.plot(st_dt_df['timestamp'], st_dt_df['open'])
    plt.show()
    variable = input('input something!: ')

    return 


def get_company_str(company_symbol):
    print(company_symbol)
    df = pd.read_csv('./nasdaq100_lk_fb_names.csv')
    print(df.head()),
    # print(df.Symbol)
    # df_row = df.loc[df['Symbol'] == company_symbol]
    # print(df_row)
    # print(df['Symbol'] == company_symbol)


    # if df['Symbol'].str.contains(company_symbol).any():
    # use list here as the above statement will let part of the string pass as well, like "AM", rather than AMD
    if company_symbol in df['Symbol'].tolist():
        print('passed 1st if')
        # df = df.loc[df['Symbol'] == company_symbol]
        lk_bool = pd.isnull(df.loc[df.Symbol == company_symbol, 'LK']).bool()
        fb_bool = pd.isnull(df.loc[df.Symbol == company_symbol, 'FB']).bool()
        print(lk_bool, type(lk_bool))
        print(fb_bool, type(fb_bool))
        if lk_bool or fb_bool:
            return None, None
        else:
            return df.loc[df.Symbol == company_symbol, 'LK'].item(), df.loc[df.Symbol == company_symbol, 'FB'].item() 
    else:
        return None, None
            

def data_for_prediction(stock_company_symbol, fb_company_str, lk_company_str):
    print(stock_company_symbol, fb_company_str, lk_company_str)

    # # 'Microsoft'
    # stock_company_symbol = company_str

    # # company_str):
    # # company_str = 'AMD'
    # fb_company_str = company_str
    # lk_company_str = company_str
    # # 'Microsoft'
    # stock_company_symbol = company_str
    # # 'msft'

    linkedin_dir = './datalab/datalab_linkedin_cleaned_more.csv'
    lk_df = pd.read_csv(linkedin_dir)
    # convert date to UTC
    lk_df['as_of_date']= pd.to_datetime(lk_df['as_of_date'], utc = True)

    lk_month_df = select_LinkedIn_data(lk_df, lk_company_str, 2016)
    print (type(lk_month_df))
    print (lk_month_df)

    # make_LinkedIn_Plot(lk_month_df[1:-1])
    
    print('----------------------------------------')
    facebook_dir = './datalab/datalab_facebook_cleaned_more.csv'
    fb_df = pd.read_csv(facebook_dir)
    # convert date to UTC
    fb_df['time']= pd.to_datetime(fb_df['time'], utc = True)
    fb_month_df = select_FB_data(fb_df, fb_company_str, 2016)
    print(type(fb_month_df))
    print(fb_month_df)

    print('----------------------------------------')
    stock_dir = './nasdaq100/' + stock_company_symbol + '.csv'
    st_df = pd.read_csv(stock_dir)
    st_df['timestamp']= pd.to_datetime(st_df['timestamp'], utc = True)
    st_df['open'] = st_df['open'].apply(clean_currency).astype('float')
    # make_stock_plot(st_df)
    st_month_df = select_stock_data(st_df)
    print(st_month_df)

    # print(st_month_df['timestamp'])
    # print(st_month_df['open'])

    return lk_month_df, fb_month_df, st_month_df



def data_next_for_prediction(fb_company_str, lk_company_str):
    print(fb_company_str, lk_company_str)

    linkedin_dir = './datalab/datalab_linkedin_cleaned_more.csv'
    lk_df = pd.read_csv(linkedin_dir)
    # convert date to UTC
    lk_df['as_of_date']= pd.to_datetime(lk_df['as_of_date'], utc = True)

    lk_month_df = select_LinkedIn_data(lk_df, lk_company_str, 2017)
    print (type(lk_month_df))
    print (lk_month_df)

    # make_LinkedIn_Plot(lk_month_df[1:-1])
    
    print('----------------------------------------')
    facebook_dir = './datalab/datalab_facebook_cleaned_more.csv'
    fb_df = pd.read_csv(facebook_dir)
    # convert date to UTC
    fb_df['time']= pd.to_datetime(fb_df['time'], utc = True)
    fb_month_df = select_FB_data(fb_df, fb_company_str, 2017)
    print(type(fb_month_df))
    print(fb_month_df)

    # print(st_month_df['timestamp'])
    # print(st_month_df['open'])

    return lk_month_df, fb_month_df


def make_prediction(lk_month_df, fb_month_df, st_month_df, lk_month_df_next, fb_month_df_next):
    # print(lk_month_df.to_frame().info())
    # print(lk_month_df.to_frame())
    x_merged = lk_month_df.to_frame().join(fb_month_df.to_frame())    # print (x_merged)
    X = x_merged
    print (X)
    y = st_month_df
    
    lr = LinearRegression()  # make an instance of the model 
    lr.fit(X, y)             # fit the model

    # get X_next from historical data
    x_next_merged = lk_month_df_next.to_frame().join(fb_month_df_next.to_frame())
    X_next = x_next_merged
    print (X_next)
    
    # feed to make prediction for next year
    y_pred = lr.predict(X_next)
    print (y_pred)
    print (type(y_pred))
    print (lr.coef_)

    # fit y_pred to get ratings
    month_list = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
    slope, intercept = np.polyfit(month_list, y_pred, 1)
    print(slope)
    
    rating = None

    if slope < -0.1:
        rating = "Decreasing"
    elif slope > 0.1: 
        rating = "Increasing"
    else:
        rating = "Remaining same"

    return y_pred.tolist(), rating 
    
    # plt.plot(X, y, 'o', color = 'k', label='training data')
    # plt.plot(X, y_pred, color='#42a5f5ff', label='model prediction')
    # plt.xlabel('Total square feet of above ground living area')
    # plt.ylabel('Home price ($)')
    # plt.legend();
    # plt.show()

def get_company_prediction(company_symbol):
    lk,fb = get_company_str(company_symbol)
    if lk and fb:
        # get data for train
        lk_month_df, fb_month_df, st_month_df = data_for_prediction(company_symbol, fb, lk)
        if lk_month_df.shape[0] == fb_month_df.shape[0] == st_month_df.shape[0]:
            # get data_next for prediction
            lk_month_df_next, fb_month_df_next = data_next_for_prediction(fb, lk)
            # check if data_next has error when using for prediction
            if lk_month_df_next.shape[0] == fb_month_df_next.shape[0]:
                y_pred, rating = make_prediction(lk_month_df, fb_month_df, st_month_df, lk_month_df_next, fb_month_df_next)
                return y_pred, rating
            else:
                return [], None

        else:
            return [], None

    else:
        return [], None

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
