import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm 


spotify_payments_time_chunks = ['0-7','7-10','10-15','15-24']




spotify_play_global = pd.read_csv("output_spotify_play.csv")
spotify_payments_global = pd.read_csv("output_square_payments.csv")
spotify_play_global['key'] = spotify_play_global['key'].str.replace("Jesus", "8BSTTGBX5Z7VM")

key_list_play = list(spotify_play_global['key'].value_counts().index)
key_list_payments = list(spotify_payments_global['location'].value_counts().index)

if len(key_list_play) >= len(key_list_payments):
    main_list = key_list_play
else:
    main_list = key_list_payments

for key in main_list:
    try:
        spotify_play     = spotify_play_global[spotify_play_global['key'] == key]
        if len(spotify_play) < 1:
            print("Key:", key, "wasn't found in play data")
            raise Exception
        print(key)
        #print(spotify_play)

        spotify_play['timestamp'] = pd.to_datetime(spotify_play['timestamp'])

        spotify_play['hour'] = spotify_play['timestamp'].dt.hour
        spotify_play['date'] = spotify_play['timestamp'].dt.date
        playlist_df = spotify_play['context_externalUrls'].dropna().groupby([spotify_play['date'],spotify_play['hour']]).agg(lambda x:x.value_counts().index[0]).reset_index()
        
        variables_mean_calculation = ['track_album_popularity', 'track_features_acousticness', 'track_features_danceability', 'track_features_duration', 'track_features_energy', 'track_features_instrumentalness', 'track_features_loudness_N', 'track_features_mode', 'track_features_speechiness', 'track_features_tempo', 'track_features_valence','track_explicitLyrics_bOOL']

        spotify_play['track_explicitLyrics_bOOL'] = spotify_play['track_explicitLyrics_bOOL'].astype("float64")
        
        meaned_spotify_play = spotify_play[variables_mean_calculation].groupby([spotify_play['date'],spotify_play['hour']]).mean().reset_index()
        max_spotify_play = spotify_play[variables_mean_calculation].groupby([spotify_play['date'],spotify_play['hour']]).max().reset_index().rename(columns=lambda x: x + '_max').rename(columns={"date_max":"date","hour_max":"hour"})
        min_spotify_play = spotify_play[variables_mean_calculation].groupby([spotify_play['date'],spotify_play['hour']]).min().reset_index().rename(columns=lambda x: x + '_min').rename(columns={"date_min":"date","hour_min":"hour"})
        std_spotify_play = spotify_play[variables_mean_calculation].groupby([spotify_play['date'],spotify_play['hour']]).std().reset_index().rename(columns=lambda x: x + '_std').rename(columns={"date_std":"date","hour_std":"hour"})

        df_1 = spotify_play[variables_mean_calculation].groupby([spotify_play['date'],spotify_play['hour']]).max().reset_index() 
        df_2 = spotify_play[variables_mean_calculation].groupby([spotify_play['date'],spotify_play['hour']]).min().reset_index()
        hourly_diff_spotify_play = pd.concat( [(df_1[variables_mean_calculation]-df_2[variables_mean_calculation]), df_1[['date','hour']]],axis=1).rename(columns=lambda x: x + '_diff').rename(columns={"date_diff":"date","hour_diff":"hour"})

        spotify_end = meaned_spotify_play
        spotify_end = pd.merge(spotify_end, max_spotify_play, how='left', on=['date','hour'])
        spotify_end = pd.merge(spotify_end, min_spotify_play, how='left', on=['date','hour'])
        spotify_end = pd.merge(spotify_end, std_spotify_play, how='left', on=['date','hour'])
        spotify_end = pd.merge(spotify_end, hourly_diff_spotify_play, how='left', on=['date','hour'])

        spotify_end.to_csv(str(key+"_music_variables.csv"))


        spotify_payments = spotify_payments_global[spotify_payments_global['location'] == key]
        if len(spotify_payments) < 1:
            print("Key:", key, "wasn't found in payments data")
            raise Exception
    except Exception as e:
        print(e)
        continue



    try:
        """
        spotify_play['timestamp'] = pd.to_datetime(spotify_play['timestamp'])

        spotify_play['hour'] = spotify_play['timestamp'].dt.hour
        spotify_play['date'] = spotify_play['timestamp'].dt.date
        playlist_df = spotify_play['context_externalUrls'].dropna().groupby([spotify_play['date'],spotify_play['hour']]).agg(lambda x:x.value_counts().index[0]).reset_index()
        print(spotify_play)

        variables_mean_calculation = ['track_album_popularity', 'track_features_acousticness', 'track_features_danceability', 'track_features_duration', 'track_features_energy', 'track_features_instrumentalness', 'track_features_loudness_N', 'track_features_mode', 'track_features_speechiness', 'track_features_tempo', 'track_features_valence','track_explicitLyrics_bOOL']

        spotify_play['track_explicitLyrics_bOOL'] = spotify_play['track_explicitLyrics_bOOL'].astype("float64")

        meaned_spotify_play = spotify_play[variables_mean_calculation].groupby([spotify_play['date'],spotify_play['hour']]).mean().reset_index()
        max_spotify_play = spotify_play[variables_mean_calculation].groupby([spotify_play['date'],spotify_play['hour']]).max().reset_index().rename(columns=lambda x: x + '_max').rename(columns={"date_max":"date","hour_max":"hour"})
        min_spotify_play = spotify_play[variables_mean_calculation].groupby([spotify_play['date'],spotify_play['hour']]).min().reset_index().rename(columns=lambda x: x + '_min').rename(columns={"date_min":"date","hour_min":"hour"})
        std_spotify_play = spotify_play[variables_mean_calculation].groupby([spotify_play['date'],spotify_play['hour']]).std().reset_index().rename(columns=lambda x: x + '_std').rename(columns={"date_std":"date","hour_std":"hour"})

        df_1 = spotify_play[variables_mean_calculation].groupby([spotify_play['date'],spotify_play['hour']]).max().reset_index() 
        df_2 = spotify_play[variables_mean_calculation].groupby([spotify_play['date'],spotify_play['hour']]).min().reset_index()
        hourly_diff_spotify_play = pd.concat( [(df_1[variables_mean_calculation]-df_2[variables_mean_calculation]), df_1[['date','hour']]],axis=1).rename(columns=lambda x: x + '_diff').rename(columns={"date_diff":"date","hour_diff":"hour"})
        """



        spotify_payments['timedate'] = spotify_payments['id'].str.split("$").str[0]


        spotify_payments['timedate'] = pd.to_datetime(spotify_payments['timedate'])

        spotify_payments['hour'] = spotify_payments['timedate'].dt.hour
        spotify_payments['date'] = spotify_payments['timedate'].dt.date
        spotify_payments['weekday'] = spotify_payments['timedate'].dt.weekday





        spotify_payments["time_chunk"] = 0
        """
        for i in range(0, len(spotify_payments_time_chunks)):
            chunk = spotify_payments_time_chunks[i]
            chunk_start = chunk.split("-")[0]
            chunk_end = chunk.split("-")[1]
            spotify_payments.loc[(spotify_payments['hour'] >= int(chunk_start)) & (spotify_payments['hour'] < int(chunk_end)), "time_chunk"] = i
        """
        spotify_payments.loc[(spotify_payments['hour'] >= int(0)) & (spotify_payments['hour'] < int(2)), "time_chunk"] = 1
        spotify_payments.loc[(spotify_payments['hour'] >= int(2)) & (spotify_payments['hour'] < int(17)), "time_chunk"] = 2
        spotify_payments.loc[(spotify_payments['hour'] >= int(17)) & (spotify_payments['hour'] < int(20)), "time_chunk"] = 3
        spotify_payments.loc[(spotify_payments['hour'] >= int(20)) & (spotify_payments['hour'] < int(23)), "time_chunk"] = 3



        spotify_payments['discount_percent'] = spotify_payments['discount_N']/spotify_payments['collected']
        spotify_payments['discount'] = spotify_payments['discount'].fillna(0).replace(np.nan,0)
        spotify_payments.loc[spotify_payments['discount'] != 0, 'discount_percent'] = spotify_payments['discount']/spotify_payments['collected']


        spotify_payments['credit_card_sales'] = 0
        spotify_payments.loc[spotify_payments['tenders_0_type'] == "CREDIT_CARD", 'credit_card_sales'] = 1

        quantity_columns = [x for x in spotify_payments.columns if (x.find("items_") != -1) & (x.find("_quantity") != -1)]
        spotify_payments['number_of_items'] = spotify_payments[quantity_columns].sum(axis=1).fillna(0).replace(np.nan, 0)

        spotify_payments['average_price_per_item'] = spotify_payments['collected']/spotify_payments['number_of_items']
        spotify_payments['average_price_per_item'] = spotify_payments['average_price_per_item'].fillna(0).replace(np.nan, 0)


        modifier_columns = []
        for column in quantity_columns:
            # print(column)
            columns_name = column + "_modifier_type_calculation"
            modifier_column = column.replace("quantity", "modifiers") + "_0_type"
            try:
                spotify_payments[columns_name] = np.nan
                spotify_payments.loc[spotify_payments[modifier_column].astype("str") == "modifier", columns_name] = spotify_payments[column]
                modifier_columns.append(columns_name)
            except KeyError:
                continue
        spotify_payments['modifier_number_of_items'] = spotify_payments[modifier_columns].sum(axis=1)
        spotify_payments['modifier_share'] = spotify_payments['modifier_number_of_items']/spotify_payments['number_of_items']
        spotify_payments['modifier_share'] = spotify_payments['modifier_share'].fillna(0).replace(np.nan, 0)

        variation_columns = []
        for column in quantity_columns:
            columns_name = column + "_variation_type_calculation"
            variation_column = column.replace("quantity", "variation")
            try:
                spotify_payments[columns_name] = np.nan
                spotify_payments.loc[(spotify_payments[variation_column].astype("str") == "Regular") | (spotify_payments[variation_column].astype("str") == "Regular Price"), columns_name] = spotify_payments[column]
                variation_columns.append(columns_name)
            except KeyError:
                continue
        spotify_payments['regular_number_of_items'] = spotify_payments[variation_columns].sum(axis=1)
        spotify_payments['regular_share'] = spotify_payments['regular_number_of_items']/spotify_payments['number_of_items']
        spotify_payments['regular_share'] = spotify_payments['regular_share'].fillna(0).replace(np.nan, 0)




        spotify_payments_mean_variables = ['regular_share', 'regular_number_of_items','modifier_number_of_items', 'modifier_share', 'weekday', 'discount_percent', 'discount', 'credit_card_sales', 'number_of_items', 'average_price_per_item', 'time_chunk']

        spotify_payments_hourly_mean = spotify_payments[spotify_payments_mean_variables].groupby([spotify_payments['date'],spotify_payments['hour']]).mean().reset_index().rename(columns=lambda x: x + '_mean').rename(columns={"date_mean":"date","hour_mean":"hour"})
        spotify_collected_sum = spotify_payments['collected'].groupby([spotify_payments['date'],spotify_payments['hour']]).sum().reset_index().rename(columns=lambda x: x + '_sum').rename(columns={"date_sum":"date","hour_sum":"hour"})


        spotify_end = meaned_spotify_play
        spotify_end = pd.merge(spotify_end, max_spotify_play, how='left', on=['date','hour'])
        spotify_end = pd.merge(spotify_end, min_spotify_play, how='left', on=['date','hour'])
        spotify_end = pd.merge(spotify_end, std_spotify_play, how='left', on=['date','hour'])
        spotify_end = pd.merge(spotify_end, hourly_diff_spotify_play, how='left', on=['date','hour'])

        spotify_end.to_csv(str(key+"_music_variables.csv"))
        spotify_payments_hourly_mean.to_csv(str(key+"_payments_hourly_mean_variables.csv"))
        spotify_collected_sum.to_csv(str(key+"_collected_sum_variables.csv"))

        variable_prediction = 'average_price_per_item_mean' #modifier_share_mean
        spotify_end = pd.merge(spotify_payments_hourly_mean[[variable_prediction,'date','hour']],meaned_spotify_play, how='left', on=['date','hour'])
        spotify_end = pd.merge(spotify_end, max_spotify_play, how='left', on=['date','hour'])
        spotify_end = pd.merge(spotify_end, min_spotify_play, how='left', on=['date','hour'])
        spotify_end = pd.merge(spotify_end, std_spotify_play, how='left', on=['date','hour'])
        spotify_end = pd.merge(spotify_end, hourly_diff_spotify_play, how='left', on=['date','hour'])




        spotify_y_train = spotify_end[variable_prediction]
        spotify_X_train = spotify_end.drop([variable_prediction,'date','hour'], axis=1).replace(np.nan,-5).fillna(-5).replace(np.inf,-5).replace(-np.inf,-5)


        regr = linear_model.LinearRegression()
        regr.fit(spotify_X_train, spotify_y_train)
        logit = sm.OLS(spotify_y_train.as_matrix(), spotify_X_train.as_matrix())
        with open(str('./' + key + "_" + variable_prediction + '_linear_summery.csv'),'w') as fh:
            fh.write( logit.fit().summary().as_csv() )
            fh.write("\n")
            fh.write(str(list(np.array(spotify_X_train.columns))).replace("[","").replace("]","").replace(",","\n"))
        print(variable_prediction, "r2:" , logit.fit().rsquared)


        variable_prediction = 'modifier_share_mean' #modifier_share_mean
        spotify_end = pd.merge(spotify_payments_hourly_mean[[variable_prediction,'date','hour']],meaned_spotify_play, how='left', on=['date','hour'])
        spotify_end = pd.merge(spotify_end, max_spotify_play, how='left', on=['date','hour'])
        spotify_end = pd.merge(spotify_end, min_spotify_play, how='left', on=['date','hour'])
        spotify_end = pd.merge(spotify_end, std_spotify_play, how='left', on=['date','hour'])
        spotify_end = pd.merge(spotify_end, hourly_diff_spotify_play, how='left', on=['date','hour'])


        spotify_y_train = spotify_end[variable_prediction]
        spotify_X_train = spotify_end.drop([variable_prediction,'date','hour'], axis=1).replace(np.nan,-5).fillna(-5).replace(np.inf,-5).replace(-np.inf,-5)

        regr = linear_model.LinearRegression()
        regr.fit(spotify_X_train, spotify_y_train)
        logit = sm.OLS(spotify_y_train.as_matrix(), spotify_X_train.as_matrix())
        with open(str('./' + key + "_" + variable_prediction + '_linear_summery.csv'),'w') as fh:
            fh.write( logit.fit().summary().as_csv() )
            fh.write("\n")
            fh.write(str(list(np.array(spotify_X_train.columns))).replace("[","").replace("]","").replace(",","\n"))
        print(variable_prediction, "r2:" , logit.fit().rsquared)


        variable_prediction = 'collected_sum' #modifier_share_mean
        spotify_end = pd.merge(spotify_collected_sum[[variable_prediction,'date','hour']],meaned_spotify_play, how='left', on=['date','hour'])
        spotify_end = pd.merge(spotify_end, max_spotify_play, how='left', on=['date','hour'])
        spotify_end = pd.merge(spotify_end, min_spotify_play, how='left', on=['date','hour'])
        spotify_end = pd.merge(spotify_end, std_spotify_play, how='left', on=['date','hour'])
        spotify_end = pd.merge(spotify_end, hourly_diff_spotify_play, how='left', on=['date','hour'])


        spotify_y_train = spotify_end[variable_prediction]
        spotify_X_train = spotify_end.drop([variable_prediction,'date','hour'], axis=1).replace(np.nan,-5).fillna(-5).replace(np.inf,-5).replace(-np.inf,-5)

        regr = linear_model.LinearRegression()
        regr.fit(spotify_X_train, spotify_y_train)
        logit = sm.OLS(spotify_y_train.as_matrix(), spotify_X_train.as_matrix())
        with open(str('./' + key + "_" + variable_prediction + '_linear_summery.csv'),'w') as fh:
            fh.write( logit.fit().summary().as_csv() )
            fh.write("\n")
            fh.write(str(list(np.array(spotify_X_train.columns))).replace("[","").replace("]","").replace(",","\n"))
        print(variable_prediction, "r2:" , logit.fit().rsquared)




        spotify_end = pd.merge(spotify_collected_sum, spotify_payments_hourly_mean, how='left', on=['date','hour'])
        spotify_end = pd.merge(spotify_end, meaned_spotify_play, how='left', on=['date','hour'])
        spotify_end = pd.merge(spotify_end, max_spotify_play, how='left', on=['date','hour'])
        spotify_end = pd.merge(spotify_end, min_spotify_play, how='left', on=['date','hour'])
        spotify_end = pd.merge(spotify_end, std_spotify_play, how='left', on=['date','hour'])
        spotify_end = pd.merge(spotify_end, hourly_diff_spotify_play, how='left', on=['date','hour'])


        buisness_columns = ['collected_sum', 'regular_share_mean', 'regular_number_of_items_mean', 'modifier_number_of_items_mean', 'modifier_share_mean', 'weekday_mean', 'discount_percent_mean', 'discount_mean', 'credit_card_sales_mean', 'number_of_items_mean', 'average_price_per_item_mean']

        spotify_end.corr()[buisness_columns].drop(buisness_columns,axis=0).drop(['hour', 'time_chunk_mean'],axis=0).to_csv(str(key+"_correlation_coefficients.csv"))

        spotify_end.to_csv(str(key+"_spotify_calculated.csv"), index=False)
    except Exception:
        print("Something wrong when calculating", key)


    """

    spotify_end = pd.merge(spotify_payments_hourly_mean[['average_price_per_item_mean', 'modifier_share_mean' ,'date','hour']],meaned_spotify_play, how='left', on=['date','hour'])
    spotify_end = pd.merge(spotify_end, spotify_collected_sum, how='left', on=['date','hour'])
    spotify_end[spotify_end.isnull().any(axis=1)].dropna(axis=1).to_csv(str(key+"_spotify_end_no_music_payments_mean_hourly.csv"),index=None)


    spotify_end = pd.merge(spotify_payments_hourly_mean[['average_price_per_item_mean', 'modifier_share_mean' ,'date','hour']],meaned_spotify_play, how='left', on=['date','hour'])
    spotify_end = pd.merge(spotify_end, spotify_collected_sum, how='left', on=['date','hour'])
    spotify_end = spotify_end[~spotify_end.isnull().any(axis=1)].dropna(axis=1)
    spotify_end.to_csv(str(key+"_spotify_end_music_payments_mean_hourly.csv"),index=None)



    save_df = pd.merge(playlist_df, spotify_end, how='left', on=['date','hour'])


    save_df[save_df['context_externalUrls'] == "{\"spotify\":\"https://open.spotify.com/user/getpando/playlist/1uUFFR8cjsfo8G9Px4wXgM\"}"].fillna(0).to_csv("spotify_end_music_payments_mean_hourly_Exemplar-Juicery-PL1.csv",index=None)

    save_df[save_df['context_externalUrls'] == "{\"spotify\":\"https://open.spotify.com/user/getpando/playlist/7glirmmrdl7yaX6wdqPbYn\"}"].fillna(0).to_csv("spotify_end_music_payments_mean_hourly_Exemplar-Juicery-PL2.csv",index=None)

    save_df[save_df['context_externalUrls'] == "{\"spotify\":\"https://open.spotify.com/user/getpando/playlist/3YYuXtgkskxbOrMd43DmsA\"}"].fillna(0).to_csv("spotify_end_music_payments_mean_hourly_Exemplar-Juicery-PL3.csv",index=None)
    """