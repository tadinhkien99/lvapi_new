import io
from datetime import datetime
import time
import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from sklearn import preprocessing


def data_generator(content, column_number):
    data = []
    # for file in file_names:
    #     temp = pd.read_csv(open(file, 'r'), sep="\s+", header=None)
    temp = pd.read_csv(io.BytesIO(content), sep="\s+", header=None)
    num = np.int(np.floor(len(temp[0])/1024)) # As all columns have same number of entries
    data = data + list(temp[column_number][0:num*1024].values.reshape(num,32,32,1))
    data = np.asarray(data).reshape(-1,32,32,1)
    return data

def data_life_generator(content, fileName):
    columns = ["B1"]
    df = pd.read_csv(io.BytesIO(content), sep='\t', names=columns)
    rmsB1 = np.sqrt(np.mean(df.B1 ** 2))
    kurtosis1 = kurtosis(df.B1)

    dateString = fileName
    str_dt = dateString[:10].replace(".", "-") + " " + dateString[11:].replace(".", ":")

    datetime_object = datetime.strptime(str_dt, '%Y-%m-%d %H:%M:%S')
    timestamp = int(time.mktime(datetime_object.timetuple()))


    start_time = int(3)
    times_use = timestamp - start_time

    data = {"rms": rmsB1, "time": datetime_object.time(), "date_time_format": datetime_object,
            "day": datetime_object.date().day,
            "hour": datetime_object.time().hour, "date_time": timestamp,
            "date": datetime_object.date(), "kurtosis": kurtosis1, "year": datetime_object.date().year, "month": datetime_object.date().month,
            "day": datetime_object.date().day, "hour": datetime_object.time().hour,
            "minute": datetime_object.time().minute, "second": datetime_object.time().second,
            "time_use": times_use
            }

    # rms_list_all_events.append(data)
    rms_df = pd.DataFrame(data)
    # rms_df.sort_values(["date_time"], axis=0, ascending=[True], inplace=True)
    rms_df = rms_df.reset_index(drop=True)

    final_df = []

    min_max_scaler = preprocessing.MinMaxScaler()
    rms_df.time_use = min_max_scaler.fit_transform(rms_df['time_use'].values.reshape(-1, 1))
    rms_df.rms = min_max_scaler.fit_transform(rms_df['rms'].values.reshape(-1, 1))
    rms_df.kurtosis = min_max_scaler.fit_transform(rms_df['kurtosis'].values.reshape(-1, 1))
    time_use = rms_df.time_use

    kurt_rms_previous = pd.DataFrame({"kurt_fitted": rms_df.kurtosis, "rms_fitted": rms_df.rms,
                                      "date_time": time_use})
    current_k_r = kurt_rms_previous[1:]
    kurt_rms_previous.index = kurt_rms_previous.index + 1  # index-restting for joining on index
    kurt_rms_previous.columns = ['kurt_fitted_previous', 'rms_fitted_previous', "date_time_previous"]
    final_df = current_k_r.join(kurt_rms_previous)
    return final_df


