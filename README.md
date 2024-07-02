# voter-turnout
*Collaborators: Sarah Liang, George Weale, Ryan Lee, Ryan Yee*

The goal of this group project is to provide insights to potential political campaign moves for U.S. swing states supported by big data analysis of voter turnout in Georgia.


The following listed files include code and model building steps. The entire project is neatly reported and presented [here](https://liang-sarah.github.io/voter-turnout/project-report.pdf).
<br />

### Overview and Results
We developed a logistic regression model using GCP and PySpark to analyze voter turnout data, achieving an AUC of 0.612. From this information, we recommended targeting 18-35 and low-income demographics, according to model predictors, to increase voter turnout. Potential future work, can include refining the model, exploring interaction terms, and adding to campaign strategies.



<br />

#### Python packages used:
<table border = "0">
  <tr>
    <td>Pandas</td> <td>statsmodels</td> <td>matplotlib</td> <td>NumPy</td>
  </tr>
  <tr>
     <td>PySpark</td> <td>seaborn</td> <td>sklearn</td>
  </tr>
</table>
<br />
<br />

### `project-report.pdf`
Written report with abstract, project objectives, exploratory/preliminary analysis, further methodology and analysis, results, and conclusion.
<br />
<br />

### `main_code.ipynb`
This is a Jupyter notebook with code cells and commentary cells corresponding to the plots in the final pdf report.
<br />
<br />

### `data-visualizations-code.py`
Python script file with code for all plots conducted (report plots plus some).
<br />
<br />

### `logistic-regression-code.py`
Python script file with code for model building and updating using PySpark SQL and PySpark ML features.
