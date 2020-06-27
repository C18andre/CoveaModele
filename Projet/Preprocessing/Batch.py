# Imports externes
import pandas as pd


# Get a random sample from the dataframe
def getRandomSample(args) :
    data = pd.read_csv(args.path_data_csv)
    sample = data.sample(n = args.sample_size)
    sample.reset_index(inplace = True)
    del sample['index']
    sample_data = convertToFloat(sample)
    sample_data.to_csv(args.path_sample_csv,index = False)

# Convert to float
def convertToFloat(dataframe) :
    data = dataframe.copy()
    for column in data.columns :
        if data[column].dtypes == 'object' :
            for i in range(len(data)) :
                if type(data[column][i]) == str :
                        try :
                            data[column][i] = float(data[column][i].replace(',','.'))
                        except :
                            data[column][i] = 0.0
    return data
