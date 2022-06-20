# Zillow Clustering Project

### Project Goals:
### <center> __Log Error = log(Zestimate) - log(Sale Price)__ <center>

### Project Goals:

- Discover the key drivers of zestimate errors for single family properties.

- Analyze commonalities of zestimates that have overpredicted actual market price (positive value of log error)

- Analyze commonalities of zestimates that have underpredicted actual market price (negative value of log error)
    
- Empower zillow executives with insights which can influence business strategies

- Attempt to increase target variable (log error) predictions utilizing machine learning algorithms at the grassroots level.

- Set a clear and solid foundation for future exploration and modeling

- Thoroughly document the process and key findings.

### Hypotheses/Questions to Explore :

- Do different features drive decreased negative log error (underpredicting) verses increased positive(over predicting)?
    
    - Separate train dataframe into negative and positive (**note: this  is for exploratory purposes on train only. For predictive modeling, we do not know if log error will be positive or negative so I will only be splitting train data**)

- What kind of relationship does year built have to log error (if any)? I predict that older homes would be harder to predict, but I am curious to see if they'd be over predicted or under.

- Are there features that share some similarities and could be combined into a new feature to aid in predicting?

### Data Dictionary:

- parcelid: property identifier
- bedroomcnt: bedroom count of property
- bathroomcnt: bathroom count of property
- yearbuilt: property's year built
- fips: county of property
- calculatedfinishedsquarefeet: square footage of property
- lotsizesquarefeet: square footage of property's lot/land
- latitude: latitude coordinate of property
- longitude: longitude coordinate of property
- logerror : log of the difference between actual value and zestimate
- propertycountylandusecode: based on location


### Summary of Findings & Recommendations:
-  I have discovered some weak relationships between features (and specific combinations of features) to log error.
-  High positive log error (Zillow overpredicting the home price) & high negative log error (underprediction) have slightly different divers
-  **Higher over prediction risk:**
    - older homes
    - one bedroom homes
    - homes with one or four bathrooms
    - homes with assessed value on the very high or very low end
    - specific locations
- **Higher under prediction risk:**
    - older homes
    - specific locations
- **Best predictions (log error close to zero):**
    - homes built after 1970
    - homes in Orange county and other more specific locations
    - homes with 3 or 4 bedrooms
- I have attempted modeling to predict the target variable (log error) using classification algorithms (binning log error ranges in different categories). Specifically, I've built and trained 3 decision tree models and 3 random forest models on:
    - features from the original dataset the show at least a slight relationship to the target variable
    - manufactured features derived from clustering algorithms (KMEANS, DBSCAN)
    - manufactured features derived from clustering algorithms (KMEANS, DBSCAN) modeled on individual clusters (one of the five clusters is demonstrated in this report)
- Thus Far, I have not created a model that I can confidently recommend for utilization, however I did find that it is likely useful to model on subgroups (ie. the clusters demonstrated in this report) to enhance prediction accuracy.
- Moving forward, I'd like to focus on clustering and modeling with the goal of predicting overpredictions vs. underpredictions
- I strongly believe there are unknown features influencing log error and I am requesting further time and resources(additional data) to explore this notion.
- I recommend gathering additional data:
    - revisit features with too much missing data to be useful (ie. air conditioning type, heating type, basements) and attempting to gather missing information
    - obtain data that is not currently part of zillow's dataset:
    - population demographics
    - school districts
    - previous transaction history
    - proximity to the coast, schools, churches, airports, landmarks, jails, etc.
    - information on renovations

### Reproducing this project
Acquire and utilize credentials to access the Zillow database. Store credentials and database access in a env.py file and create a .gitignore which includes env.py to protect your credentials. Replicate zillow_final_report.ipynb, acquire.py, prepare.py, explore.py from this repository. 
