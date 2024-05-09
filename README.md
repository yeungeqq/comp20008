1. Data Preprocessing
Ensure that the BX-Books.csv, BX-Ratings.csv and BX-Users.csv files live in the root project folder.  
To process the three datasets and create and export the combined dataframe, run:
```
python3 preprocessing.py
```

Once the preprocessing is complete, check that you have the file:
`datasets/combinedData.csv`

Then you need to discretise user age, year published and rating, run: 
```
python3 discretisation.py
```


2. Feature Selections
After performing data preprocessing program, make sure the below files are on the /datasets directory before executing the mi.py program:
BX-Books-processed.csv
BX-Ratings-processed.csv
BX-Users-processed.csv
discreizedData.csv

Execute the below command to perform feature selection:
python3 mi.py

3. Model Execution and Result Visualisation
After feature selections, make sure features.csv is on the main directory before executing the model program.

The model requires two command line arguments which specify the algorithm and if predicting new data or not.
python3 {model_program} {algorithm} {predict_new_data}

If executing the model of predicting actual rating, substitute model_program with classification_all_ratings.py; if predicting rating tiers, enter classification_rating_tiers.py:
python3 classification_all_rating.py {algorithm} {predict_new_data}
python3 classification_rating_tiers.py {algorithm} {predict_new_data}

If using decision tree as the algorithm, enter dt for algorithm; If using k-nearest neighbours as the algorithm, enter knn for algorithm:
python3 classification_all_rating.py dt {predict_new_data}
python3 classification_rating_tiers.py dt {predict_new_data}
python3 classification_all_rating.py knn {predict_new_data}
python3 classification_rating_tiers.py knn {predict_new_data}

If using the model to predict new data, enter y for predict_new_data. Otherwise, enter n.

An example usage is as below:
python3 classification_rating_tiers.py dt y

By executing the above command, it prompts users to enter user id and ISBN (The user and book have to be in the BX-Users-processed.csv and BX-Books-processed.csv files). After that, it generates the classification report which specify the performance of the model. It will also generate the confusion matrix and save it as an png file in the results folder. Finally, it will show the prediction of new data, given the user id and ISBN.

if the second command line argument is n, it will not have prompts and just create the performance report and graphs.



