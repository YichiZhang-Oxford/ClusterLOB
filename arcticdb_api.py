"""Read only API to arctic db instances for OMI data. Please report any bugs to jan@robots.ox.ac.uk"""
from arcticdb import Arctic
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime 





""" ==================== METHODs ============================ """

def fn_str_s3_uri_ro(str_data_set):
    access_key = "ISvPoXxEfZq8R5NrhteF"
    secret_key = "q19I0b8lQvPEwAsJANs3lFUFyVDJpzjpzTTmZTb0"
    if str_data_set == 'pinnacle':
        str_arcticdb_name ='pinnacle'
        #access_key = "gFd3fcaSmeM1tyZiKnrF"
        #secret_key = "gisDfVDdioL309103m2MrJoU4AvUgj12cYr2oyjU"
        
        s3_uri = f"s3s://omi-rapid-graid.omirapid.oxford-man.ox.ac.uk:{str_arcticdb_name}?region=omi-eaglehouse&port=9000&access={access_key}&secret={secret_key}"


    elif str_data_set == 'lobster':
         str_arcticdb_name = 'lobster'
         s3_uri = f"s3s://omi-rapid-graid.omirapid.oxford-man.ox.ac.uk:{str_arcticdb_name}?region=omi-eaglehouse&port=9000&access={access_key}&secret={secret_key}"


    return s3_uri    #we return an a string to be used as Arctic(s3_uri) to create an arcticdb instance



#returns an arctic db instance for a specific data set (read_only):

def fn_adb_instance(str_data_set):
    
    return Arctic(fn_str_s3_uri_ro(str_data_set))    #we return an arctic db instance containing the chosen data set

#The next function fetches symbols from different libraries and collates them into one big pandas data frame:
def fn_get_data (arctic_instance,liblist, symblist, collist = None,timerange = None):

    dfs = []  # To store the individual DataFrames    
    
    le = len(liblist)

    for i in range(le):
        
        library_name = liblist[i]  # Replace with your library name
        library = arctic_instance[library_name]
        
        # Retrieve the symbols
        symbol_names = symblist[i]  # Replace with your symbol names
        if not (collist is None):
            column_names = collist[i]
        else:
            column_names = None
    
        
        for str_symbol_name in symbol_names:
            symbol = library.read(str_symbol_name,columns = column_names,date_range = timerange)
            df = pd.DataFrame(symbol.data).add_prefix(str_symbol_name+'_')  # Assuming Date and Value columns
            #df.set_index('Date', inplace=True)
            #df.rename(columns={'Value': symbol_name}, inplace=True)  # Rename the 'Value' column
            dfs.append(df)
        
    

    # Concatenate the DataFrames into a combined one which is returned
    return pd.concat(dfs, axis=1)




