#### **Date: 27th-01-2021**
#### **Author : Bessam Mehenni**

### Adoption of solar energy in residential in Virginia state

Report is available and can be downloaded (“Report_solar_adopt_EN”)<br/><br/>

##  **1.	References**

[1] Data source: Stanford’s Deepsolar dataset, dec.2018
http://web.stanford.edu/group/deepsolar/home

[2] Python notebook / data analysis: 
https://www.kaggle.com/andromedasagan/implementation-of-solar-energy-in-the-us

##  **2.	Context**

The source of data is Stanford University’s DeepSolar project, a deep learning framework that analyzed satellite images to detect solar panels throughout the country. The data collected are the size, type (residential/non-residential) of the power systems distributed in the 48 states in the U.S. The associated socioeconomic data for these locations were recorded over several years.

My ambition in this work is to build a socio-economic analysis of the last mile to understand what are the profiles within a homogeneous group of households that is adopting solar energy. It focuses on the state of Virginia.

This work is based on a first chapter of data analysis [2] that highlights key trends and correlations in the deployment of solar power based on the full Deepsolar dataset.

##  **3.	Goal**
-	Explore the data, handle missing values.
-	Make visualizations to identify trends. Identify the characteristics of a homogeneous group of households that makes the majority of installed systems.
-	Model by ML the adoption of solar energy by households. we take the target variable **`Solar_panel_area_per_capita`** to illustrate the adoption of solar systems. Using ML to show the factors involved in explaining the adoption.


#Explanations about the way I proceed:

#Importing Deepsolar dataset restricted to Virginia state

#Creation of column 'employment_rate' and calculation of the employment rate

#Deleting 'employed' and 'unemployed' columns

![PIC1](/md_images/pic1.PNG)
 
##  **4.	Data exploration**

We will measure the **adoption of solar systems** through the variable `solar_panel_area_per_capita`. We will draw up a matrix of correlations of deemed and less deemed factors over the target.
Certain factors are deemed to be decisive in the choice for households to equip with solar systems. These factors become evident when the following observations are made:
-	households are likely to equip themselves with solar equipment where the solar resource is the most abundant.
-	households that can afford it financially are more likely to invest in solar systems.
-	incentives can facilitate access to solar systems, especially for households that initially lacked the capacity to afford them.
-	feed-in tariffs for grid electricity are decisive for acquiring solar systems.
Two missing factors that will unfortunately not be studied here, as they have not been collected in the present dataset:
-	I think that data about the **intensity of incentives** would have been useful, i.e. special retail electricity tariffs (in c$/kWh) or investment incentives for the purchase of solar systems. This can make the difference and allow households to move from a situation of non-accessor to potential accessor of a solar system.
-	the **ecological awareness** is a more personal factor that is likely to be influential. Maybe it could have been read in voting intentions for example.


#Explanations about the way I proceed:

#Variables are renamed : `incentive_count_residential` in `incentive_count_resid`, `incentive_residential_state_level` in `incentive_count_resid_state`



*Short description of the extracted dataset*

Dataset contains socio-economic and environmental data.
-	county - county name
-	average_household_income - average annual household income ($),ACS 2015 (5-Year Estimates)
-	daily_solar_radiation - daily solar radiation (kWh/m2/d),NASA Surface Meteorology and Solar Energy
-	incentive_count_resid - number of incentives for residential solar,www.dsireusa.org
-	avg_electricity_retail_rate - average residential retail electricity price over the past 5 years,EIA
-	incentive_count_resid_state - number of state-level incentives for residential solar,www.dsireusa.org
-	solar_panel_area_per_capita - solar panel area per capita (m2/capita),deepsolar
-	age_median - median age,ACS 2015 (5-Year Estimates)
-	number_of_years_of_education - number of years of education,ACS 2015 (5-Year Estimates)
-	employed,number of employed people,ACS 2015 (5-Year Estimates)
-	unemployed,number of unemployed people,ACS 2015 (5-Year Estimates)

##  **5.	Missing values**

In the dataset, there is a problem with very large floating-point numbers for which INF values are returned. To solve the problem, the max values are filtered and then discarded.

#Explanations about the way I proceed:

#Data cleaning: deletion of values Inf

#Printing a dataset describe

![PIC2](/md_images/pic2.PNG)
 
#Fill NaN with the column median value (except in`daily_solar_radiation` with the mean value)

#Delete rows without values in `Solar_panel_area_per_capita`.

##  **6.	Vizualisations**

#Visualization of the different behaviors between the main factors and the target in order to identify possible outliers
![PIC3](/md_images/pic3.PNG)

#Data cleaning: We see several outliers that can be deleted after 0.125
![PIC5](/md_images/pic5.PNG)
 
We take the target variable **`Solar_panel_area_per_capita`** to illustrate the **adoption of solar systems**.

#To not skew the analysis by zero-values, we keep a dataset with only rows `solar_panel_area_per_capita` > 0

#At this time, we ignore variables that are identical in value within the entire population (std = 0)

#Drop_elements = [`county`,`incentive_count_resid`,`incentive_count_resid_state`,`avg_electricity_retail_rate`]

Putting the factors face to face on graphs will allow us to see how what the locality has in household profiles contributes to adoption.

![PIC6](/md_images/pic6.PNG) 

According to the Solar_panel_area_per_capita histogram, the group of vast majority of installed solar systems is located where `Solar_panel_area_per_capita` is below **0.03 m2/capita**.


![PIC7](/md_images/pic7.PNG)

Let's look at the ranges of median age, income and education level of 60% of the records of this sample, between the 20% and 80% percentiles.

We note the very meaningful characteristics of the group of households whose 
`Solar_panel_area_per_capita` is below 0.03 m2/capita for the vast majority of installed solar systems. For **60% of the records in this sample**:
-	average household income is between **56000** and **128000$**
-	median age is between **33** and **44 years old**.
-	level of education is between **13** and **16 years of education**.

##  **7.	Correlational analysis**
![PIC8](/md_images/pic8.PNG)

**There is a very strong correlation between `average_household_income` and
`number_of_years_of_education`. We can easily explain this correlation. In general, education opens doors to higher-paying jobs.**

**`Age_median` is much less correlated with average_household_income.**

**It would have been interesting to see how these factors correlate with other factors that reflect the "green" mentality or the motivation to do savings.**

#What are the features without variance?

![PIC9](/md_images/pic9.PNG)

#We ignore variables without variance that will not bring to a modeling.

#Despite there is a strong correlation of the target with `number_of_years_of_education`, the model gives better prediction results with it, that's why we keep it. And it works better without `employment_rate`.

#retained = [`number_of_years_of_education`,`daily_solar_radiation`,`age_median`,`average_household_income`]


##  **8.	Forecasting solar adoption**

######  **8.1	Preliminary modeling**

#3D visualization between selected factors in a first view
![PIC10](/md_images/pic10.PNG)

#3D visualization between selected factors in a second view
![PIC11](/md_images/pic11.PNG)

#Preliminary modeling using a RandomForestRegressor

![PIC12](/md_images/pic12.PNG)
 
Good fitting on the training data. The test score could be much higher.

#Feature importance: ranking of the features in the explanation of the target

![PIC13](/md_images/pic13.PNG)
 
**A limited number of descriptives variables can acceptably explain the adoption of solar energy: the predominant are the `average_household_income` of the locality, `age_median` and `number_of_years_of_education` factors.**

**The level of education (`number_of_years_of_education`) factor may be influencing the adoption because of partly an intrinsec factor which is the level of income again (`average_household_income`). We have seen that these factors are highly correlated.**

**`Age_median` has also a contribution to the adoption.**
 
#We continue with the same descriptive variables for the fine-tuned model

#retained = [`average_household_income`,`age_median`,`daily_solar_radiation`,`number_of_years_of_education`]



######  **8.2	Fine-tuning the parameters of the model**

#GridSearch

![PIC14](/md_images/pic14.PNG)

Correct fitting on the training data, the test score is acceptable. The model shows some signs of overfitting, i.e. weaknesses in its generalizability.

#Feature importance : ranking of the features in the explanation of the target

![PIC15](/md_images/pic15.PNG)

![PIC16](/md_images/pic16.PNG)
 

######  **8.3	Conclusions**

**The fine-tuned model confirms that the `average_household_income` of the locality, 
`number_of_years_of_education`, `age_median` and `daily_solar_radiation` factors can acceptably explain the adoption of solar energy in Virginia state.**

**The `average_household_income` factor has the most important contribution.**

**The level of education (`number_of_years_of_education`) is also an important factor in which we can find intrinsically the level of income (`average_household_income`), as seen before in the correlation matrix.**

**Concerning the contribution of `age_median`, to explain this we should rather look at what advancement in age and career brings: perhaps the rationality of the choices of household members, the stability of its income and the ability to make investments in order to project savings in the upcoming years.**
