The dataset folder is strictly for data source csv files

Do _not_ save additional csvs to this folder — it is not for resulting csvs. 

The only reason we are saving csvs that are taking up 20MB+ of space is to minimise on API calls. 
I repeat, do not save additional csvs. 

Every subset of data features/columns that you could ever want or need can be filtered from: 
- combinedData.csv
- discretizedData.csv

If you _do_ 
1) need additional columns accessible _everywhere_ or 
2) have a good reason that you cannot add the dataframe column within the flow of your code

you can add them to the preprocessing file or the discretisation file. 

Please consider the purpose and scope of what you're adding, if something similar already exists and you can merely update code to get your desired results and find a home for it accordingly. 

