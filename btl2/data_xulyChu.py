import pandas as pd
def floatToText(number):
    if number < 200:
        return 'low'
    elif number <250:
        return 'medium'
    else:
        return 'high'
df = pd.read_csv('btl2\cleaned_data2.csv')
df['Hardness'] = df['Hardness'].apply(floatToText)
df.to_csv('btl2\cleaned_data2.csv',index = False)