private_func_names_AIR_01_gpt35 = """
{head}
--- Interaction History: ---
{dialogue_hist}

--- Task: ---
You are provided with a custom function library called 'decision_company', which contains numerous functions provided by the user. From the User-AI interaction history under "---History:---", locate the line that begins with "from decision_company import read_csv_file," in the generated code between "'''". Your task is to analyze the underlying pattern of how to extract and import custom functions from 'decision_company'. 

To aid your analysis, start by generating a Pseudocode to replace the latest python code between "'''". Then, extracte important custom functions based on the generated pseudocode and User-AI interaction history. And you must also provide the confidence which is a number between 0 to 1 and the reason to choose that customized function as shown in the example below.

For example:
################### Example Starts: ###################
[USER (financial analyst)]: ... ... Finally, please create a head-to-head profit-loss matrix table for top 10 companies in the tech sector.

[YOU (AI assistant)]:
# Output:
---------------- Pseudocode Generation: ----------------
<pseudocode>
// PSEUDOCODE TO CREATE A HEAD-TO-HEAD PROFIT-LOSS MATRIX TABLE FOR TOP 10 COMPANIES IN THE TECH SECTOR

// Import necessary customized functions
FROM decision_company IMPORT read_csv_file, create_zeros_array, create_dataframe, check_elements_in_list, iterate_rows, access_dataframe_loc, update_dataframe_loc

// Load the financial data from a CSV file
LOAD DATA FROM 'Tech_Sector_Financials.csv'

// Define the top 10 company names (assuming this is already done)
DEFINE top_10_company_names

// Initialize a 10x10 matrix with zeros to hold the profit-loss records
INITIALIZE profit_loss_matrix WITH 10x10 ZEROS

// Filter financial records to include only those of top 10 companies in the last five years
FILTER tech_financials FOR records WHERE
    Company is in top_10_company_names
    AND record is in the last five years

// Iterate over the filtered financial records to update profit-loss records
FOR EACH record IN tech_financials
    DETERMINE profit AND loss FROM record
    INCREMENT the corresponding cell in profit_loss_matrix
    BASED ON profit AND loss values

// Output the matrix to check the results
DISPLAY profit_loss_matrix

// Save the profit-loss matrix to a file for future reference
SAVE profit_loss_matrix TO FILE 'profit_loss_matrix.pkl'
</pseudocode>

---------------- Customized Functions Extraction: ----------------
<FUNCTIONS>
1. function: "read_csv_file" | confidence: "0.9" | reason: "Need to load the financial data from a CSV file."
2. function: "create_zeros_array" | confidence: "0.8" | reason: "Need to initialize a 10x10 matrix with zeros to hold the profit-loss records"
 ... 
</FUNCTIONS>
################### Example Ends ###################

Please note that, you have to generate the correct pseudocode based on the last User-AI interaction. Also, the functions extracted between "<FUNCTIONS> [{{"function": "read_csv_file", "confidence": 0.9, reason: "..."}}, ..., ...]</FUNCTIONS>" should match the functions imported from "decision_company" in the pseudocode above between "<pseudocode> YOUR-PSEUDOCODE </pseudocode>".

[YOU (AI assistant)]:
# Output:
---------------- Pseudocode Generation: ----------------
<pseudocode>
"
"""

private_func_names_AIR_02_gpt35 = """
{head}
--- Interaction History: ---
{dialogue_hist}

--- Task: ---

Analyze the provided custom function library named 'decision_company', which is enclosed within the BEGIN and END markers.
Please extract and output the function names based on the content between the following markers:

------------------------------------- decision_company BEGIN: -------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from scipy import stats

{segment}
------------------------------------- decision_company END: -------------------------------------

The 'decision_company' is a custom function library provided to you. It contains various functions created by the user. You are asked to examine the User-AI interaction history and identify the line that starts with "from decision_company import read_csv_file," in the code generated under the "---BEGIN CODE TEMPLATE---". Your task is to understand the underlying pattern of how to extract and import custom functions from 'decision_company'. Below is an example of a function extract plan derived from the User-AI interaction history and the generated pseudocode.

################### Example Starts: ###################
{last_query}
# Output:
---------------- Pseudocode Generation: ----------------
<pseudocode>
{example}
################### Example Ends ###################

For this task, you need to extract useful customized functions from decision_company and fill into "<FUNCTIONS> [{{"function": "read_csv_file", "confidence": 0.9, "reason": "..."}}, ..., ...] </FUNCTIONS>" for the New Query with a code snippet. To do this, first generate pseudocode and then analyze it to determine which customized functions to extract. You can refer to the example above for guidance on how to analyze and extract from the user query and generated pseudocode. However, you need to analyze and come up with a new pseudocode instead of simply copying example. You can refer to customized functions provided in "decision_company" above, DO NOT imagine any custom function name not in "decision_company"!

NOTE: You must also provide the confidence which is a number between 0 to 1 and the reason to choose that customized function as shown in the example above.

--- New Query: ---
{new_query}

[YOU (AI assistant)]:
# Output:
---------------- Pseudocode Generation: ----------------
<pseudocode>
"""

segment_1 = """
def update_dict(original_dict, updates):  
    '''  
    Updates the original dictionary with the key-value pairs from the updates dictionary.  
  
    :param original_dict: The original dictionary to be updated.  
    :param updates: The dictionary containing updates to be applied.  
    '''  
    original_dict.update(updates)

def use_function(df, func, axis=0, raw=False, result_type=None, args=()):
    '''
    This function apply function to pandas Series.

    :param series: pandas Series
    :return: series that with function applied
    '''
    return df.apply(func, axis=axis, raw=raw, result_type=result_type, args=args)

def add_legend():
    '''
    This function add legned to plot.
    '''
    plt.legend()

def read_csv_file(file_path):
    '''
    This function reads a CSV file and returns a pandas DataFrame.

    :param file_path: file path of the CSV file
    :return: pandas DataFrame
    '''
    return pd.read_csv(file_path)

def create_dataframe(data, columns=None, index=None):
    '''
    This function creates a new DataFrame object from the given data.

    :param data: data to be used for creating the DataFrame
    :param columns: list of column names for the DataFrame (optional)
    :param index: list of index labels for the DataFrame (optional)
    :return: pandas DataFrame
    '''
    return pd.DataFrame(data, columns=columns, index=index)

def is_null(dataframe):
    '''
    This function is used to detect missing values in the input DataFrame and returns a boolean mask of the same shape.
    True indicates a missing value (NaN), and False indicates a non-missing value.

    :param dataframe: pandas DataFrame
    :return: DataFrame with boolean mask indicating missing values
    '''
    return dataframe.isnull()

def sum_up(dataframe):
    '''
    This function is used to sum the values along a specified axis (default is 0).
    In this case, it is used to count the number of missing values in each column of the DataFrame.

    :param dataframe: pandas DataFrame
    :return: Series with the sum of values along the specified axis
    '''
    return dataframe.sum()

def n_unique(dataframe, columns=None):
    '''
    This function is used to count the number of unique values in each column of the input DataFrame.
    If a list of columns is provided, it will only count unique values in those columns.

    :param dataframe: pandas DataFrame
    :param columns: list of column names to count unique values (default is None)
    :return: Series with the number of unique values in each column
    '''
    if columns:
        dataframe = dataframe[columns]
    return dataframe.nunique()

def count_unique_values(series, normalize=False, sort=True, ascending=False, bins=None, dropna=True):
    '''
    Return a pandas series containing the counts of unique values in the input series.

    :param series: The input pandas series to count unique values on.
    :param normalize: Whether to return relative frequencies of the unique values. Default is False.
    :param sort: Whether to sort the resulting counts by frequency. Default is True.
    :param ascending: Whether to sort the resulting counts in ascending order. Default is False.
    :param bins: If passed, will group the counts into the specified number of bins. Default is None.
    :param dropna: Whether to exclude missing values from the counts. Default is True.
    :return: A pandas series containing the counts of unique values in the input series.
    '''
    return series.value_counts(normalize=normalize, sort=sort, ascending=ascending, bins=bins, dropna=dropna)

def reset_index(dataframe, drop=True):
    '''
    This function is used to reset the index of a DataFrame.

    :param dataframe: pandas DataFrame
    :param drop: boolean, whether to drop the old index and not add it as a new column (default is True)
    :return: DataFrame with reset index
    '''
    return dataframe.reset_index(drop=drop)

def concatenate_objects(obj1, obj2, ignore_index=False, reset_index_flag=True):
    '''
    Concatenate two pandas objects along a particular axis and reset the index if specified.

    :param obj1: first pandas object (Series or DataFrame)
    :param obj2: second pandas object (Series or DataFrame)
    :param reset_index_flag: boolean, whether to reset the index or not (default: True)
    :return: concatenated pandas object (Series or DataFrame)
    '''
    concatenated = pd.concat([obj1, obj2], ignore_index=ignore_index)
    if reset_index_flag:
        concatenated = reset_index(concatenated, drop=True)
    return concatenated

def generate_summary_stat(dataframe):
    '''
    Generate various summary statistics of a DataFrame or Series.

    :param dataframe: pandas DataFrame or Series
    :return: pandas DataFrame with summary statistics
    '''
    return dataframe.describe()

def transform(series, mapping):
    '''
    Map values from one Series to another based on a function or a dictionary.

    :param series: pandas Series
    :param mapping: function or dictionary to map values
    :return: pandas Series with mapped values
    '''
    return series.map(mapping)

def create_subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, width_ratios=None, height_ratios=None, subplot_kw=None, gridspec_kw=None, **fig_kw):
    '''
    Create a new figure and a set of subplots.

    :return: tuple containing a figure and an axis
    '''
    return plt.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, squeeze=squeeze, width_ratios=width_ratios, height_ratios=height_ratios, subplot_kw=subplot_kw, gridspec_kw=gridspec_kw, **fig_kw)

def create_histogram_subplot(ax, data, bins, alpha):
    '''
    Create a histogram using the given axis, data, bins, and alpha (transparency).

    :param ax: axis object
    :param data: data for the histogram
    :param bins: number of bins for the histogram
    :param alpha: transparency value (0 to 1)
    '''
    ax.hist(data, bins=bins, alpha=alpha)

def create_scatter_plot(ax, x_data, y_data, alpha):
    '''
    Create a scatter plot using the given axis, x_data, y_data, and alpha (transparency).

    :param ax: axis object
    :param x_data: data for the x-axis
    :param y_data: data for the y-axis
    :param alpha: transparency value (0 to 1)
    '''
    ax.scatter(x_data, y_data, alpha=alpha)

def create_bar_chart(ax, data):
    '''
    Create a bar chart using the given axis and data.

    :param ax: axis object
    :param data: data for the bar chart
    '''
    data.plot.bar(ax=ax)

def set_plot_split_title(ax, title):
    '''
    Set the title of a plot.

    :param ax: axis object
    :param title: title for the plot
    '''
    ax.set_title(title)

def make_xlabel(ax, xlabel):
    '''
    Set the x-axis label of a plot.

    :param ax: axis object
    :param xlabel: label for the x-axis
    '''
    ax.set_xlabel(xlabel)

def make_ylabel(ax, ylabel):
    '''
    Set the y-axis label of a plot.

    :param ax: axis object
    :param ylabel: label for the y-axis
    '''
    ax.set_ylabel(ylabel)

def show_plots():
    '''
    Display the created plots.
    '''
    plt.show()

def df_copy(dataframe):
    '''
    Creates a new DataFrame with the same data as the original DataFrame.

    :param dataframe: original pandas DataFrame
    :return: new pandas DataFrame
    '''
    return dataframe.copy()

def dropna(dataframe, subset_columns):
    '''
    Removes rows with missing or invalid data in the specified subset of columns.

    :param dataframe: pandas DataFrame
    :param subset_columns: list of column names to check for missing or invalid data
    :return: pandas DataFrame with rows removed
    '''
    return df_copy(dataframe.dropna(subset=subset_columns))

def get_dummies(data, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False, dtype=None):
    '''
    Performs one-hot encoding on the specified columns.

    :param dataframe: pandas DataFrame
    :param columns: list of column names to be one-hot encoded
    :param prefix: string to append before the new column names
    :param prefix_sep: string to separate the prefix from the new column names
    :return: pandas DataFrame with one-hot encoded columns
    '''
    return pd.get_dummies(data, prefix=prefix, prefix_sep=prefix_sep, dummy_na=dummy_na, columns=columns, sparse=sparse, drop_first=drop_first, dtype=dtype)
"""

segment_2 = """
def create_barplot(data=None, x=None, y=None, hue=None, order=None, hue_order=None, estimator='mean', errorbar=('ci', 95), n_boot=1000, units=None, seed=None, orient=None, color=None, palette=None, saturation=0.75, fill=True, hue_norm=None, width=0.8, dodge='auto', gap=0, log_scale=None, native_scale=False, formatter=None, legend='auto', capsize=0, err_kws=None, ax=None, **kwargs):
    '''
    This function creates a bar plot using seaborn.

    :param data: DataFrame, array, or list of arrays, optional
    :param x, y, hue: names of variables in data or vector data, optional
    :param order, hue_order: lists of strings, optional
    :param estimator: statistical function to estimate within each categorical bin, optional
    :param errorbar: error bar to draw on bars, optional
    :param n_boot: number of bootstraps to compute the confidence interval, optional
    :param units: identifier of sampling units, optional
    :param seed: seed or random number generator, optional
    :param orient: orientation of the plot (vertical or horizontal), optional
    :param color: matplotlib color, optional
    :param palette: palette name, list, or dict, optional
    :param saturation: proportion of the original saturation to draw colors, optional
    :param fill: if True, fill the bars with the hue variable, optional
    :param hue_norm: tuple or matplotlib.colors.Normalize, optional
    :param width: width of a full element when not using hue nesting, or width of all the elements for one level of the major grouping variable, optional
    :param dodge: when hue nesting is used, whether elements should be shifted along the categorical axis, optional
    :param gap: size of the gap between bars, optional
    :param log_scale: boolean or number, or pair of boolean or number, optional
    :param native_scale: boolean, optional
    :param formatter: callable or string format, optional
    :param legend: boolean or 'auto', optional
    :param capsize: width of the caps on error bars, optional
    :param err_kws: keyword arguments for matplotlib.axes.Axes.errorbar(), optional
    :param ax: matplotlib Axes, optional
    :param kwargs: key, value pairings
    :return: matplotlib Axes
    '''
    return sns.barplot(data=data, x=x, y=y, hue=hue, order=order, hue_order=hue_order, estimator=estimator, errorbar=errorbar, n_boot=n_boot, units=units, seed=seed, orient=orient, color=color, palette=palette, saturation=saturation, fill=fill, hue_norm=hue_norm, width=width, dodge=dodge, gap=gap, log_scale=log_scale, native_scale=native_scale, formatter=formatter, legend=legend, capsize=capsize, err_kws=err_kws, ax=ax, **kwargs)

def access_dataframe_loc(dataframe, row_label, col_label):
    '''
    This function accesses a value in the DataFrame using row and column labels.

    :param dataframe: pandas DataFrame to access
    :param row_label: row label for the value to access
    :param col_label: column label for the value to access
    :return: value at the specified row and column
    '''
    return dataframe.loc[row_label, col_label]

def corr(dataframe):
    '''
    Creates a correlation matrix of the specified DataFrame columns.

    :param dataframe: pandas DataFrame
    :return: pandas DataFrame representing the correlation matrix
    '''
    return dataframe.corr()

def avg(dataframe):
    '''
    Calculates the average value of a dataset.

    :param dataframe: pandas DataFrame or Series
    :return: mean (average) value
    '''
    return dataframe.mean()

def f_oneway(*args):
    '''
    Performs a one-way ANOVA (Analysis of Variance) test to compare the means of multiple groups.

    :param *args: datasets (arrays, lists, or pandas Series) to be compared
    :return: F-statistic and p-value of the ANOVA test
    '''
    return stats.f_oneway(*args)

def sem(dataframe):
    '''
    Calculates the standard error of the mean (SEM) for a given dataset.

    :param dataframe: pandas DataFrame or Series
    :return: standard error of the mean (SEM)
    '''
    return stats.sem(dataframe)

def t_ppf(probability, degrees_of_freedom):
    '''
    Calculates the inverse of the Student's t-distribution cumulative distribution function (CDF)
    for a given probability and degrees of freedom.

    :param probability: probability to be used for the inverse CDF calculation
    :param degrees_of_freedom: degrees of freedom for the t-distribution
    :return: critical t-value
    '''
    return stats.t.ppf(probability, degrees_of_freedom)

def bind_dataframe(dataframe, column):
    '''
    Group the DataFrame by a specific column.
    :param dataframe: The input DataFrame to group
    :param column: The column to group by
    :return: A DataFrameGroupBy object
    '''
    return dataframe.groupby(column)

def aggregate_grouped_data(grouped_data, agg_dict):
    '''
    Apply aggregation functions on the grouped data.
    :param grouped_data: A DataFrameGroupBy object to apply the aggregation on
    :param agg_dict: A dictionary containing the aggregation functions for each column
    :return: A DataFrame with the aggregated data
    '''
    return grouped_data.agg(agg_dict).reset_index()

def make_bins(x, bins, right=True, labels=None, retbins=False, precision=3, include_lowest=False, duplicates='raise', ordered=True):
    '''
    Divide data into bins based on specified intervals and labels.
    :param data: The input data to be binned
    :param bins: A list of intervals to define the bins
    :param labels: A list of labels for each bin
    :return: A Series containing the binned data
    '''
    return pd.cut(x, bins, right=right, labels=labels, retbins=retbins, precision=precision, include_lowest=include_lowest, duplicates=duplicates, ordered=ordered)

def join_dataframes(df1,right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=None, indicator=False, validate=None):
    '''
    Merge two DataFrames based on common columns.
    :param df1: The first DataFrame to merge
    :param df2: The second DataFrame to merge
    :param left_on: Column or index level names to join on in the left DataFrame
    :param right_on: Column or index level names to join on in the right DataFrame
    :param how: Type of merge to be performed (default: 'inner')
    :return: A merged DataFrame
    '''
    return df1.merge(right, how=how, on=on, left_on=left_on, right_on=right_on, left_index=left_index, right_index=right_index, sort=sort, suffixes=suffixes, copy=copy, indicator=indicator, validate=validate)

def create_figure(figsize):
    '''
    This function creates a new figure with the specified figsize.
    :param figsize: A tuple of the width and height of the figure in inches
    :return: None
    '''
    plt.figure(figsize=figsize)

def create_barplot(x, y, hue, data):
    '''
    Create a bar plot based on the given data.
    :param x: The data to be used for the x-axis
    :param y: The data to be used for the y-axis
    :param hue: The variable to be used for distinguishing the bars in the plot
    :param data: The DataFrame containing the x, y, and hue data
    :return: An Axes object containing the bar plot
    '''
    return sns.barplot(x=x, y=y, hue=hue, data=data)

def set_plot_title(title):
    '''
    Set the title for the current plot.
    :param title: The title string
    :return: None
    '''
    plt.title(title)

def set_yaxis_label(label):
    '''
    Set the label for the y-axis.
    :param label: The label string
    :return: None
    '''
    plt.ylabel(label)

def anova_test(data1, data2, data3):
    '''
    Perform one-way ANOVA (analysis of variance) test on the given data.
    :param data1: The first dataset for the ANOVA test
    :param data2: The second dataset for the ANOVA test
    :param data3: The third dataset for the ANOVA test
    :return: An F_onewayResult object containing the results of the ANOVA test
    '''
    return stats.f_oneway(data1, data2, data3)

def positive_infinity():
    '''
    A infinity constant for computation.
    Return a positive infinity float value.
    :return: A float representing positive infinity
    '''
    return np.inf

def convert_to_datetime(input_data):
    '''
    This function converts the given input into datetime format.

    :param input_data: input data to be converted
    :return: datetime object
    '''
    return pd.to_datetime(input_data)

def col_copy(df, columns):
    '''
    This function creates a copy of the selected columns from a DataFrame.
    :param df: The input DataFrame to copy columns from
    :param columns: A list of column names to copy
    :return: A DataFrame containing the copied columns
    '''
    return df[columns].copy()

def extract_year(df, column):
    '''
    This function extracts the year from a specified datetime column in a DataFrame.
    :param df: The input DataFrame containing the datetime column
    :param column: The datetime column to extract the year from
    :return: A Series containing the extracted year values
    '''
    return df[column].dt.year

def aggregate_data(grouped_data, agg_dict):
    '''
    This function applies aggregation functions on a grouped DataFrame.
    :param grouped_data: A DataFrameGroupBy object to apply the aggregation on
    :param agg_dict: A dictionary containing the aggregation functions for each column
    :return: A DataFrame with the aggregated data
    '''
    return grouped_data.agg(agg_dict)

def draw_lineplot(x, y, hue=None, data=None, ci=None):
    '''
    This function creates a line plot based on the given data.
    :param x: The data to be used for the x-axis
    :param y: The data to be used for the y-axis
    :param hue: The variable to be used for distinguishing the lines in the plot
    :param data: The DataFrame containing the x, y, and hue data
    :param ci: The size of the confidence intervals for the lines
    :return: None
    '''
    sns.lineplot(x=x, y=y, hue=hue, data=data, errorbar=ci)

def set_plot_ylabel(label):
    '''
    This function sets the label for the y-axis of the current plot.
    :param label: The label string
    :return: None
    '''
    plt.ylabel(label)

def set_plot_xlabel(label):
    '''
    This function sets the label for the x-axis of the current plot.
    :param label: The label string
    :return: None
    '''
    plt.xlabel(label)
"""

segment_3 = """
def create_histogram(data=None, x=None, y=None, hue=None, weights=None, stat='count', bins='auto', binwidth=None, binrange=None, discrete=None, cumulative=False, common_bins=True, common_norm=True, multiple='layer', element='bars', fill=True, shrink=1, kde=False, kde_kws=None, line_kws=None, thresh=0, pthresh=None, pmax=None, cbar=False, cbar_ax=None, cbar_kws=None, palette=None, hue_order=None, hue_norm=None, color=None, log_scale=None, legend=True, ax=None, **kwargs):
    '''
    This function creates a histogram plot using seaborn.

    :param data: DataFrame, array, or list of arrays, optional
    :param x, y, hue: names of variables in data or vector data, optional
    :param weights: array of weights, optional
    :param stat: aggregate statistic to compute in each bin, optional
    :param bins: specification of hist bins, or None to use Freedman-Diaconis rule, optional
    :param binwidth: width of each bin, optional
    :param binrange: lowest and highest value of bins, optional
    :param discrete: if True, draw a bar at each unique value in x, instead of using a histogram, optional
    :param cumulative: if True, the histogram accumulates values, optional
    :param common_bins: if True, the same bins will be used across all histograms, optional
    :param common_norm: if True, all densities on the same axes are scaled by the same factor, optional
    :param multiple: approach to resolving multiple elements when semantic mapping creates subsets, optional
    :param element: approach to drawing the histogram, optional
    :param fill: if True, fill in the bars in the histogram, optional
    :param shrink: scale factor for shrinking the density curve in rugplot and kde, optional
    :param kde: if True, compute a kernel density estimate to smooth the distribution and show on the plot, optional
    :param kde_kws: keyword arguments for kdeplot(), optional
    :param line_kws: keyword arguments for plot(), optional
    :param thresh: threshold for small elements, optional
    :param pthresh: threshold for small elements as a proportion, optional
    :param pmax: maximum proportion of the figure area that a small element can occupy, optional
    :param cbar: if True, add a colorbar to the figure, optional
    :param cbar_ax: axes in which to draw the colorbar, optional
    :param cbar_kws: keyword arguments for colorbar(), optional
    :param palette: palette name, list, or dict, optional
    :param hue_order: order for the levels of the hue variable in the palette, optional
    :param hue_norm: normalization in data units for colormap applied to the hue variable, optional
    :param color: color for all elements, or seed for a gradient palette, optional
    :param log_scale: boolean or number, or pair of boolean or number, optional
    :param legend: if True, add a legend or legend elements, optional
    :param ax: matplotlib Axes, optional
    :param kwargs: key, value pairings
    :return: matplotlib Axes
    '''
    return sns.histplot(data=data, x=x, y=y, hue=hue, weights=weights, stat=stat, bins=bins, binwidth=binwidth, binrange=binrange, discrete=discrete, cumulative=cumulative, common_bins=common_bins, common_norm=common_norm, multiple=multiple, element=element, fill=fill, shrink=shrink, kde=kde, kde_kws=kde_kws, line_kws=line_kws, thresh=thresh, pthresh=pthresh, pmax=pmax, cbar=cbar, cbar_ax=cbar_ax, cbar_kws=cbar_kws, palette=palette, hue_order=hue_order, hue_norm=hue_norm, color=color, log_scale=log_scale, legend=legend, ax=ax, **kwargs)

def linear_regression(x, y):
    '''
    This function performs a linear regression on the given data.
    :param x: The independent variable data (x-axis)
    :param y: The dependent variable data (y-axis)
    :return: A LinregressResult object containing the results of the linear regression
    '''
    return stats.linregress(x, y)

def save_plot(filename, dpi=100, bbox_inches='tight'):
    '''
    This function saves the current plot as an image file.

    :param filename: The name of the image file to be saved
    :param dpi: The resolution of the image in dots per inch (default: 100)
    :param bbox_inches: The bounding box in inches (default: 'tight')
    :return: None
    '''
    plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)

def create_date_offset(years):
    '''
    This function creates a date offset object for the given duration (in years).

    :param years: duration in years
    :return: date offset object
    '''
    return pd.DateOffset(years=years)

def get_min_value(data):
    '''
    This function finds the minimum value in each group of the grouped data.

    :param data: grouped pandas DataFrame
    :return: DataFrame with minimum values
    '''
    return data.min()

def sort_by_values(data, column_name):
    '''
    This function sorts the data by the values in the specified column.

    :param data: pandas DataFrame
    :param column_name: column name to sort by
    :return: sorted DataFrame
    '''
    return data.sort_values(column_name)

def get_first_n_rows(data, n=5):
    '''
    This function returns the first 'n' rows of the data.

    :param data: pandas DataFrame
    :param n: number of rows to return
    :return: DataFrame with first 'n' rows
    '''
    return data.head(n)

def get_max(data):
    '''
    This function finds the maximum value in the given data.

    :param data: pandas DataFrame or Series
    :return: maximum value
    '''
    return data.max()

def filter_by_date(data, date_col, min_date):
    '''
    This function filters the DataFrame based on the minimum date value.

    :param data: pandas DataFrame
    :param date_col: date column name
    :param min_date: minimum date value
    :return: filtered DataFrame
    '''
    return data[data[date_col] >= min_date]

def check_elements_in_list(series, values_list):
    '''
    This function checks whether each element in the series is contained in the given list or not.
    It returns a boolean Series of the same shape as the original Series.

    :param series: pandas Series
    :param values_list: list of values to check for
    :return: boolean Series
    '''
    return series.isin(values_list)

def iterate_rows(dataframe):
    '''
    This function iterates over the rows of the DataFrame as (index, Series) pairs.

    :param dataframe: pandas DataFrame to be iterated
    :return: iterator yielding index and row data as pandas Series
    '''
    return dataframe.iterrows()

def update_dataframe_loc(dataframe, row_label, col_label, value):
    '''
    This function updates a value in the DataFrame using row and column labels.

    :param dataframe: pandas DataFrame to update
    :param row_label: row label for the value to update
    :param col_label: column label for the value to update
    :param value: new value to set at the specified row and column
    '''
    dataframe.loc[row_label, col_label] = value

def convert_to_np_array(df_or_series):
    '''
    This function returns a NumPy array representation of the given DataFrame or Series.

    :param df_or_series: The input DataFrame or Series
    :return: numpy ndarray
    '''
    return df_or_series.values

def convert_np_to_list(numpy_array):
    '''
    This function converts the given NumPy array to a Python list.

    :param numpy_array: The input NumPy array
    :return: list
    '''
    return numpy_array.tolist()

def to_list(data):
    '''
    This function converts the DataFrame values to a Python list.

    :param data: pandas DataFrame or Series
    :return: list of values
    '''
    return data.values.tolist()

def rename_columns(data, columns):
    '''
    This function renames the columns in the given DataFrame.

    :param data: pandas DataFrame
    :param columns: dictionary with old column names as keys and new column names as values
    :return: DataFrame with renamed columns
    '''
    return data.rename(columns=columns)

def fill_missing_values(series, value=None, method=None, axis=None, inplace=False, limit=None):
    '''
    This function is used to fill NaN (Not a Number) values with a specified value or method.

    :param series: pandas Series
    :param value: value to fill NaNs with (default is 0)
    :return: Series with NaNs filled
    '''
    return series.fillna(value=value, method=method, axis=axis, inplace=inplace, limit=limit)

def create_zeros_array(shape):
    '''
    This function creates a new array of the specified shape and type, filled with zeros.

    :param shape: tuple of integers defining the shape of the new array
    :return: numpy array filled with zeros
    '''
    return np.zeros(shape)

def assert_series(series, value, operation):
    '''
    Compares the input Series with the given value using the specified operation.

    :param series: The input Series to compare.
    :param value: The value to compare the Series with.
    :param operation: The comparison operation to use.
    :return: A boolean Series resulting from the comparison.
    '''
    if operation == 'equality':
        return series == value
    elif operation == 'inequality':
        return series != value
    else:
        raise ValueError("Invalid operation. Supported operations are '==' and '!='.")

def logical_and(series1, series2):
    '''
    Performs a logical AND operation between two boolean Series.

    :param series1: The first input boolean Series.
    :param series2: The second input boolean Series.
    :return: A boolean Series resulting from the logical AND operation.
    '''
    return series1 & series2

def logical_or(series1, series2):
    '''
    Performs a logical OR operation between two boolean Series.

    :param series1: The first input boolean Series.
    :param series2: The second input boolean Series.
    :return: A boolean Series resulting from the logical OR operation.
    '''
    return series1 | series2

def fetch_index(dataframe):
    '''
    This function returns the index of the given DataFrame.

    :param dataframe: pandas DataFrame
    :return: pandas Index object
    '''
    return dataframe.index

def is_a_null_df(dataframe):
    '''
    Check if a DataFrame or Series is empty.

    :param dataframe: pandas DataFrame or Series
    :return: boolean, True if empty, False otherwise
    '''
    return dataframe.empty
"""

segment_4 = """
def create_countplot(data=None, x=None, y=None, hue=None, order=None, hue_order=None, orient=None, color=None, palette=None, saturation=0.75, fill=True, width=0.8, dodge='auto', ax=None, **kwargs):
    '''
    This function creates a count plot using seaborn.

    :param data: DataFrame, array, or list of arrays, optional
    :param x, y, hue: names of variables in data or vector data, optional
    :param order, hue_order: lists of strings, optional
    :param orient: orientation of the plot (vertical or horizontal), optional
    :param color: matplotlib color, optional
    :param palette: palette name, list, or dict, optional
    :param saturation: proportion of the original saturation to draw colors, optional
    :param fill: if True, fill the bars with the hue variable, optional
    :param hue_norm: tuple or matplotlib.colors.Normalize, optional
    :param stat: aggregate statistic to compute in each bin, optional
    :param width: width of a full element when not using hue nesting, or width of all the elements for one level of the major grouping variable, optional
    :param dodge: when hue nesting is used, whether elements should be shifted along the categorical axis, optional
    :param gap: size of the gap between bars, optional
    :param log_scale: boolean or number, or pair of boolean or number, optional
    :param native_scale: boolean, optional
    :param formatter: callable or string format, optional
    :param legend: boolean or 'auto', optional
    :param ax: matplotlib Axes, optional
    :param kwargs: key, value pairings
    :return: matplotlib Axes
    '''
    return sns.countplot(data=data, x=x, y=y, hue=hue, order=order, hue_order=hue_order, orient=orient, color=color, palette=palette, saturation=saturation, fill=fill, width=width, dodge=dodge, ax=ax, **kwargs)

def set_current_ticks(ticks=None, labels=None, minor=False, **kwargs):
    '''
    Get or set the current tick locations and labels of the x-axis.

    :param ticks: An array-like object containing the positions of the ticks. If None, the current ticks are used.
    :param labels: A list of string labels to use as the tick labels. If None, the current labels are used.
    :param minor: If True, the ticks will be minor ticks. If False, the ticks will be major ticks.
    :param **kwargs: Additional keyword arguments to be passed to the plt.xticks() function.
    :return: A tuple containing the x-location of the new ticks and a list of <class 'matplotlib.text.Text'> instances.
    '''

    return plt.xticks(ticks=ticks, labels=labels, minor=minor, **kwargs)

def visit_by_index(dataframe, index):
    '''
    Access elements in a DataFrame or Series by index.

    :param dataframe: pandas DataFrame or Series
    :param index: integer, index of the element to access
    :return: element at the specified index
    '''
    return dataframe.iloc[index]

def fetch_df_size(dataframe):
    '''
    Return the size of the input DataFrame or DataFrameGroupBy object.

    :param dataframe: The input DataFrame or DataFrameGroupBy object.
    :return: A Series representing the size of the input object.
    '''
    if isinstance(dataframe, pd.core.groupby.generic.DataFrameGroupBy):
        return dataframe.size()
    else:
        return pd.Series(dataframe.size)

def create_multiindex_from_product(iterables, names):
    '''
    create a MultiIndex from the cartesian product of multiple iterables.

    Params:
    iterables: List of iterables to compute the cartesian product.
    names: List of names for the resulting MultiIndex levels.

    Returns:
    A MultiIndex object.
    '''
    return pd.MultiIndex.from_product(iterables, names=names)

def convert_multiindex_to_dataframe(multiindex, index=False):
    '''
    convert a MultiIndex to a DataFrame.

    Params:
    multiindex: A pandas MultiIndex object.
    index: Whether to include the index in the resulting DataFrame (default: False).

    Returns:
    A DataFrame object.
    '''
    return multiindex.to_frame(index=index)

def remove_labels(data, columns, axis=1):
    '''
    drop specified labels from rows or columns.

    Params:
    data: A pandas DataFrame or Series object.
    columns: Labels to drop.
    axis: Axis along which the labels will be dropped (default: 1).

    Returns:
    A DataFrame or Series with the specified labels dropped.
    '''
    return data.drop(columns=columns, axis=axis)

def draw_countplot(x, data):
    '''
    Shows the counts of observations in each categorical bin using bars.
    :param x: string, column name in the data
    :param data: DataFrame, input data
    '''
    sns.countplot(x=x, data=data)

def plot_kde(data, label, shade):
    '''
    Plots a kernel density estimate of the given data.
    :param data: array-like, input data
    :param label: string, label for the plot
    :param shade: bool, whether to shade the area under the curve
    '''
    sns.kdeplot(data, label=label, fill=shade)

def select_data_types(df, dtype_list):
    '''
    This function takes a pandas DataFrame and returns a DataFrame containing only the categorical columns.

    :param df: pandas DataFrame
    :return: pandas DataFrame with categorical columns
    '''
    return df.select_dtypes(include=dtype_list)

def get_columns(df):
    '''
    This function takes a pandas DataFrame and returns its columns.

    :param df: pandas DataFrame
    :return: columns (pandas Index object)
    '''
    return df.columns

def create_standard_scaler():
    '''
    This function creates a StandardScaler instance.

    :return: StandardScaler instance
    '''
    return StandardScaler()

def fit_transform_standard_scaler(scaler, data):
    '''
    This function fits the StandardScaler to the data and transforms the data using the fitted StandardScaler.

    :param scaler: StandardScaler instance
    :param data: Data to be standardized
    :return: Standardized data
    '''
    return scaler.fit_transform(data)

def create_label_encoder():
    '''
    This function creates a LabelEncoder instance.

    :return: LabelEncoder instance
    '''
    return LabelEncoder()

def fit_transform_label_encoder(le, data):
    '''
    This function fits the LabelEncoder to the data and transforms the data using the fitted LabelEncoder.

    :param le: LabelEncoder instance
    :param data: Data to be label encoded
    :return: Label encoded data
    '''
    return le.fit_transform(data)

def create_kmeans(n_clusters=8, init='k-means++', n_init='warn', max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='lloyd'):
    '''
    This function creates a KMeans instance with the specified number of clusters.

    :param cluster_num: Number of clusters
    :param random_state: Random state for the KMeans algorithm
    :return: KMeans instance
    '''
    return KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol, verbose=verbose, random_state=random_state, copy_x=copy_x, algorithm=algorithm)

def fit_predict_kmeans(kmeans, data):
    '''
    This function fits the KMeans instance to the data and predicts the cluster index for each sample.

    :param kmeans: KMeans instance
    :param data: Data to be clustered
    :return: Cluster index for each sample
    '''
    return kmeans.fit_predict(data)

def get_silhouette_score(X, labels, metric='euclidean', sample_size=None, random_state=None, **kwds):
    '''
    This function Compute the mean Silhouette Coefficient of all samples.

    :param X: An array of pairwise distances between samples, or a feature array.
    :param labels: Predicted labels for each sample.
    :param metric: The metric to use when calculating distance between instances in a feature array. If metric is a string, it must be one of the options allowed by pairwise_distances. If X is the distance array itself, use
    :param sample_size: The size of the sample to use when computing the Silhouette Coefficient on a random subset of the data.
    :param random_state: Determines random number generation for selecting a subset of samples.
    :return: Mean Silhouette Coefficient for all samples.
    '''
    return silhouette_score(X, labels, metric=metric, sample_size=sample_size, random_state=random_state, **kwds)

def plot(*args, scalex=True, scaley=True, data=None, **kwargs):
    '''
    Plot y versus x as lines and/or markers.

    :param x, y: The horizontal / vertical coordinates of the data points. x values are optional and default to range(len(y)).
    :param **kwargs: Line2D properties, optional
    :param scalex, scaley: These parameters determine if the view limits are adapted to the data limits. The values are passed on to autoscale_view.
    :return: list of Line2D A list of lines representing the plotted data.
    '''
    return plt.plot(*args, scalex=scalex, scaley=scaley, data=data, **kwargs)

def grid(visible=None, which='major', axis='both', **kwargs):
    '''
    Configure the grid lines.

    :param visible: Whether to show the grid lines. If any kwargs are supplied, it is assumed you want the grid on and visible will be set to True.
    :param **kwargs: Line2D properties: Define the line properties of the grid
    :param which: The grid lines to apply the changes on.
    :param axis: The axis to apply the changes on
    :return: list of Line2D A list of lines representing the plotted data.
    '''
    return plt.grid(visible=visible, which=which, axis=axis, **kwargs)

def col_assign_val(df, col, val):
    '''
    Assign a value to a specified column in a DataFrame.

    :param df: pandas.DataFrame The input DataFrame to be modified
    :param col: str The column name in the DataFrame to assign the value
    :param val: The value to be assigned to the specified column
    :return: pandas.Series The modified column with the assigned value
    '''
    df[col] = val

def extract_unique_values(series):
    '''
    This function extracts unique values from the given pandas Series.

    :param series: pandas Series
    :return: series of unique values
    '''
    return series.unique()

"""

segment_5 = """
def create_heatmap(data, vmin=None, vmax=None, cmap=None, center=None, robust=False, annot=None, fmt='.2g', annot_kws=None, linewidths=0, linecolor='white', cbar=True, cbar_kws=None, cbar_ax=None, square=False, xticklabels='auto', yticklabels='auto', mask=None, ax=None, **kwargs):
    '''
    This function creates a heatmap using seaborn's heatmap function.

    :param data: 2D dataset that can be coerced into an ndarray. If a Pandas DataFrame is provided, the index/column information will be used to label the columns and rows.
    :param vmin, vmax: Values to anchor the colormap, otherwise they are inferred from the data and other keyword arguments.
    :param cmap: The mapping from data values to color space.
    :param center: The value at which to center the colormap when plotting divergant data.
    :param robust: If True and vmin or vmax are absent, the colormap range is computed with robust quantiles instead of the extreme values.
    :param annot: If True, write the data value in each cell.
    :param fmt: String formatting code to use when adding annotations.
    :param annot_kws: Keyword arguments for ax.text when annot is True.
    :param linewidths: Width of the lines that will divide each cell.
    :param linecolor: Color of the lines that will divide each cell.
    :param cbar: Whether to draw a colorbar.
    :param cbar_kws: Keyword arguments for fig.colorbar.
    :param cbar_ax: None or Axes, optional. Axes in which to draw the colorbar, otherwise take space from the main Axes.
    :param square: If True, set the Axes aspect to “equal” so each cell will be square-shaped.
    :param xticklabels, yticklabels: If True, plot the column names of the dataframe. If False, don’t plot the column names. If list-like, plot these alternate labels as the xticklabels. If an integer, use the column names but plot only every n label. If “auto”, try to densely plot non-overlapping labels.
    :param mask: If passed, data will not be shown in cells where mask is True. Cells with missing values are automatically masked.
    :param ax: If provided, plot the heatmap on this axis.
    :param kwargs: Other keyword arguments are passed down to ax.pcolormesh.

    :return: Axes object with the heatmap.
    '''
    return sns.heatmap(data, vmin=vmin, vmax=vmax, cmap=cmap, center=center, robust=robust, annot=annot, fmt=fmt, annot_kws=annot_kws, linewidths=linewidths, linecolor=linecolor, cbar=cbar, cbar_kws=cbar_kws, cbar_ax=cbar_ax, square=square, xticklabels=xticklabels, yticklabels=yticklabels, mask=mask, ax=ax, **kwargs)

def search_where(data, condition, column):  
    '''  
    Filters the data based on the given condition and returns distinct values from the specified column.  
  
    :param data: The DataFrame to search in.  
    :param condition: The boolean Series representing the condition to filter by.  
    :param column: The column from which to return values.  
    :return: A list of values from the specified column after applying the condition.  
    ''' 
  
    # Return the values from the specified column  
    return data.where(condition, column)

def series_get_quantile(series, q=0.5, interpolation='linear'):

  '''
  Calculate the quantile value of a pandas series using linear interpolation.

  :param series: The pandas series to calculate the quantile value on.
  :param q: The quantile value to calculate. Default is 0.5.
  :param interpolation: The interpolation method to use. Default is 'linear'.
  :return: The quantile value of the series at the given quantile and interpolation.
  '''
  return series.quantile(q=q, interpolation=interpolation)

def series_value_counts(series, normalize=False, sort=True, ascending=False, bins=None, dropna=True):
  '''
  Return a pandas series containing the counts of unique values in the input series.

  :param series: The input pandas series to count unique values on.
  :param normalize: Whether to return relative frequencies of the unique values. Default is False.
  :param sort: Whether to sort the resulting counts by frequency. Default is True.
  :param ascending: Whether to sort the resulting counts in ascending order. Default is False.
  :param bins: If passed, will group the counts into the specified number of bins. Default is None.
  :param dropna: Whether to exclude missing values from the counts. Default is True.
  :return: A pandas series containing the counts of unique values in the input series.
  '''
  return series.value_counts(normalize=normalize, sort=sort, ascending=ascending, bins=bins, dropna=dropna)

def series_to_dict(series):
  '''
  Convert a pandas series into a dictionary.

  :param series: The pandas series to convert into a dictionary.
  :return: A dictionary with the index values of the series as the keys and the corresponding values of the series as the values.
  '''
  return series.to_dict()

def choose_data_types(df, dtype_list):
    '''
    This function takes a pandas DataFrame and returns a DataFrame containing only the categorical columns.

    :param df: pandas DataFrame
    :return: pandas DataFrame with categorical columns
    '''
    return df.select_dtypes(include=dtype_list)

def get_columns(df):
    '''
    This function takes a pandas DataFrame and returns its columns.

    :param df: pandas DataFrame
    :return: columns (pandas Index object)
    '''
    return df.columns

def standard_scaler_instance():
    '''
    This function creates a StandardScaler instance.

    :return: StandardScaler instance
    '''
    return StandardScaler()

def modify_data_standard_scaler(scaler, data):
    '''
    This function fits the StandardScaler to the data and transforms the data using the fitted StandardScaler.

    :param scaler: StandardScaler instance
    :param data: Data to be standardized
    :return: Standardized data
    '''
    return scaler.fit_transform(data)

def encoder_instance():
    '''
    This function creates a LabelEncoder instance.

    :return: LabelEncoder instance
    '''
    return LabelEncoder()

def encode_column(le, data):
    '''
    This function fits the LabelEncoder to the data and transforms the data using the fitted LabelEncoder.

    :param le: LabelEncoder instance
    :param data: Data to be label encoded
    :return: Label encoded data
    '''
    return le.fit_transform(data)

def kmeans_instance(n_clusters=8, init='k-means++', n_init='warn', max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='lloyd'):
    '''
    This function creates a KMeans instance with the specified number of clusters.

    :param cluster_num: Number of clusters
    :param random_state: Random state for the KMeans algorithm
    :return: KMeans instance
    '''
    return KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol, verbose=verbose, random_state=random_state, copy_x=copy_x, algorithm=algorithm)

def predict_data_kmeans(kmeans, data):
    '''
    This function fits the KMeans instance to the data and predicts the cluster index for each sample.

    :param kmeans: KMeans instance
    :param data: Data to be clustered
    :return: Cluster index for each sample
    '''
    return kmeans.fit_predict(data)

def get_coefficient_silhouette(X, labels, metric='euclidean', sample_size=None, random_state=None, **kwds):
    '''
    This function Compute the mean Silhouette Coefficient of all samples.

    :param X: An array of pairwise distances between samples, or a feature array.
    :param labels: Predicted labels for each sample.
    :param metric: The metric to use when calculating distance between instances in a feature array. If metric is a string, it must be one of the options allowed by pairwise_distances. If X is the distance array itself, use
    :param sample_size: The size of the sample to use when computing the Silhouette Coefficient on a random subset of the data.
    :param random_state: Determines random number generation for selecting a subset of samples.
    :return: Mean Silhouette Coefficient for all samples.
    '''
    return silhouette_score(X, labels, metric=metric, sample_size=sample_size, random_state=random_state, **kwds)

def get_figure(*args, scalex=True, scaley=True, data=None, **kwargs):
    '''
    Plot y versus x as lines and/or markers.

    :param x, y: The horizontal / vertical coordinates of the data points. x values are optional and default to range(len(y)).
    :param **kwargs: Line2D properties, optional
    :param scalex, scaley: These parameters determine if the view limits are adapted to the data limits. The values are passed on to autoscale_view.
    :return: list of Line2D A list of lines representing the plotted data.
    '''
    plt.plot(*args, scalex=scalex, scaley=scaley, data=data, **kwargs)

def configure_gridlines(visible=None, which='major', axis='both', **kwargs):
    '''
    Configure the grid lines.

    :param visible: Whether to show the grid lines. If any kwargs are supplied, it is assumed you want the grid on and visible will be set to True.
    :param **kwargs: Line2D properties: Define the line properties of the grid
    :param which: The grid lines to apply the changes on.
    :param axis: The axis to apply the changes on
    :return: list of Line2D A list of lines representing the plotted data.
    '''
    return plt.grid(visible=visible, which=which, axis=axis, **kwargs)

def col_assign_val(df, col, val):
    '''
    Assign a value to a specified column in a DataFrame.

    :param df: pandas.DataFrame The input DataFrame to be modified
    :param col: str The column name in the DataFrame to assign the value
    :param val: The value to be assigned to the specified column
    :return: pandas.Series The modified column with the assigned value
    '''
    df[col] = val

def extract_unique_values(series):
    '''
    This function extracts unique values from the given pandas Series.

    :param series: pandas Series
    :return: series of unique values
    '''
    return series.unique()

def calc_quantile_val(series, q=0.5, interpolation='linear'):

  '''
  Calculate the quantile value of a pandas series using linear interpolation.

  :param series: The pandas series to calculate the quantile value on.
  :param q: The quantile value to calculate. Default is 0.5.
  :param interpolation: The interpolation method to use. Default is 'linear'.
  :return: The quantile value of the series at the given quantile and interpolation.
  '''
  return series.quantile(q=q, interpolation=interpolation)

def transform_to_dictionary(series):
  '''
  Convert a pandas series into a dictionary.

  :param series: The pandas series to convert into a dictionary.
  :return: A dictionary with the index values of the series as the keys and the corresponding values of the series as the values.
  '''
  return series.to_dict()

def calculate_quantile(series, percentile):
    '''
    This function calculates the quantile for a pandas Series.

    :param series: pandas Series
    :param percentile: percentile value (0.75 for 75th percentile)
    :return: quantile value
    '''
    return series.quantile(percentile)

def filter_by_condition(dataframe, condition):
    '''
    This function filters a pandas DataFrame based on a given condition.

    :param dataframe: pandas DataFrame
    :param condition: boolean condition to filter the DataFrame
    :return: filtered pandas DataFrame
    '''
    return dataframe[condition]

"""

segment_6 = """
def create_condition(series, value):
    '''
    This function creates a boolean condition for filtering a pandas DataFrame.

    :param series: pandas Series
    :param value: value to compare with the Series
    :return: boolean condition
    '''
    return series > value

def convert_to_tuples(dataframe):
    '''
    This function converts a pandas DataFrame to a list of tuples.

    :param dataframe: pandas DataFrame
    :return: list of tuples
    '''
    return dataframe.itertuples(index=True, name=None)

def filter_by_value(dataframe, column, value):
    '''
    This function filters a pandas DataFrame based on a specific value in a column.

    :param dataframe: pandas DataFrame
    :param column: column name in the DataFrame
    :param value: value to filter by in the column
    :return: filtered pandas DataFrame
    '''
    return dataframe[dataframe[column] == value]

def convert_to_list(index):
    '''
    This function converts a pandas Index to a list.

    :param index: pandas Index
    :return: list
    '''
    return list(index)

def calculate_median(dataframe, column):
    '''
    This function calculates the median for a specific column in a pandas DataFrame.

    :param dataframe: pandas DataFrame
    :param column: column name in the DataFrame
    :return: median value
    '''
    return dataframe[column].median()

def count_rows(dataframe):
    '''
    This function counts the number of rows in a pandas DataFrame.

    :param dataframe: pandas DataFrame
    :return: number of rows
    '''
    return dataframe.shape[0]

def locate_mode(series):
    '''
    This function calculates the mode of a series.

    :param series: input series
    :return: mode of the series
    '''
    return series.mode()

def get_n_row(df, n):
    '''
    This function returns the first row of a DataFrame.

    :param df: input DataFrame
    :return: first row of the DataFrame
    '''
    return df.iloc[n]

def set_layout(pad=1.08, h_pad=None, w_pad=None, rect=None):
    '''
    Adjust the padding between and around subplots.

    :param pad: Padding between the figure edge and the edges of subplots, as a fraction of the font size.
    :param h_pad, w_pad: Padding (height/width) between edges of adjacent subplots, as a fraction of the font size.
    :param rect: A rectangle in normalized figure coordinates into which the whole subplots area (including labels) will fit.
    :return: None
    '''
    plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad, rect=rect)

def get_fig_from_df(data, *args, **kwargs):
    '''
    This function plots a bar chart.

    :param data: The object for which the method is called.
    :return: None
    '''
    data.plot(*args, **kwargs)

def pivot_a_level(data, level=-1, fill_value=None):
    '''
    Pivot a level of the (necessarily hierarchical) index labels.

    :param level: Level(s) of index to unstack, can pass level name.
    :param fill_value: Replace NaN with this value if the unstack produces missing values.
    :param sort: Sort the level(s) in the resulting MultiIndex columns.
    :return: DataFrame after unstacking
    '''
    return data.unstack(level=level, fill_value=fill_value)

def cast_to_a_dtype(data, dtype, copy=None, errors='raise'):
    '''
    Cast a pandas object to a specified dtype dtype.

    :param dtype: Use a str, numpy.dtype, pandas.ExtensionDtype or Python type to cast entire pandas object to the same type. Alternatively, use a mapping, where col is a column label and dtype is a numpy.dtype or Python type to cast one or more of the DataFrame’s columns to column-specific types.
    :param copy: Return a copy when copy=True (be very careful setting copy=False as changes to values then may propagate to other pandas objects).
    :param errors: Control raising of exceptions on invalid data for provided dtype.

    :return: None
    '''
    return data.astype(dtype, copy=copy, errors=errors)

def categoricalIndex(data):
    '''
    Accessor object for categorical properties of the Series values.

    :return: None
    '''
    return data.cat

def  categorical_codes(cat):
    '''
    The category codes of this categorical index.

    :return: None
    '''
    return cat.codes

def scatter_fig_instance(data, x, y, ax):
    '''
    This function creates a scatterplot using seaborn.

    :param data: pandas DataFrame
    :param x: column name for the x-axis
    :param y: column name for the y-axis
    :param ax: axis object to plot the scatterplot on
    '''
    sns.scatterplot(data=data, x=x, y=y, ax=ax)

def divide_dataset(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None):
    '''
    Split arrays or matrices into random train and test subsets.

    :param *arrays: Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.
    :param test_size: If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. If None, the value is set to the complement of the train size. If train_size is also None, it will be set to 0.25.
    :param train_size: If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split. If int, represents the absolute number of train samples. If None, the value is automatically set to the complement of the test size.
    :param random_state: Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.
    :param shuffle: Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.
    :param stratify: If not None, data is split in a stratified fashion, using this as the class labels.

    :return: List containing train-test split of inputs.
    '''
    return train_test_split(*arrays, test_size=test_size, train_size=train_size, random_state=random_state, shuffle=shuffle, stratify=stratify)

def create_LR_instance(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None):
    '''
    This function initializes a Logistic Regression.

    :return: initialized Logistic Regression
    '''
    return LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs, l1_ratio=l1_ratio)

def classifier_training(classifier, X_train, y_train):
    '''
    This function trains a classifier on the given training data.

    :param classifier: classifier object
    :param X_train: training data features
    :param y_train: training data labels
    :return: trained classifier
    '''
    classifier.fit(X_train, y_train)
    return classifier

def classifier_predictions(classifier, X_test):
    '''
    This function makes predictions using a trained classifier.

    :param classifier: trained classifier object
    :param X_test: testing data features
    :return: predicted labels
    '''
    return classifier.predict(X_test)

def calculate_conf_mat(y_true, y_pred):
    '''
    This function calculates the confusion matrix.

    :param y_true: true labels
    :param y_pred: predicted labels
    :return: confusion matrix
    '''
    return confusion_matrix(y_true, y_pred)

def calc_acc(y_true, y_pred):
    '''
    This function calculates the accuracy score.

    :param y_true: true labels
    :param y_pred: predicted labels
    :return: accuracy score
    '''
    return accuracy_score(y_true, y_pred)
"""

pre_prompt_func_names_user_gpt35 = """
{dialogue_hist}

Analyze the provided custom function library named 'decision_company', which is enclosed within the BEGIN and END markers.
Please extract and output the function names based on the content between the following markers:

------------------------------------- decision_company BEGIN: -------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from scipy import stats

{segment}
------------------------------------- decision_company END: -------------------------------------

What are the top twenty functions in the decision_company library above that can be used to answer my latest query with code snippet? Please provide the list of function names in a specific format. If there are no relevant functions, please indicate this appropriately. The output format should be as follows:

<FUNCTIONS>
1. <FUNC>read_csv_file</FUNC>: This function is used to read a CSV file and return a pandas DataFrame. (confidence: 0.9, reason: It is the main function used to load the dataset)
2. ...
 ... ...
</FUNCTIONS>

Please include the confidence level, which is a number between 0 and 1, and the reason for choosing each customized function, as shown in the example above.

"""

pre_prompt_func_names_AI_gpt35 = """
[YOU (AI assistant)]:
# Output:
<FUNCTIONS> 
1. <FUNC>
"""

private_func_names_AIR_01 = """
{head}
--- Interaction History: ---
{dialogue_hist}

--- Task: ---
You are provided with a custom function library called 'decision_company', which contains numerous functions provided by the user. From the User-AI interaction history under "---History:---", locate the line that begins with "from decision_company import read_csv_file," in the generated code between "'''". Your task is to analyze the underlying pattern of how to extract and import custom functions from 'decision_company'. 

To aid your analysis, start by generating a Pseudocode to replace the latest python code between "'''". Then, extracte important custom functions based on the generated pseudocode and User-AI interaction history. 

For example:
################### Example Starts: ###################
[USER (financial analyst)]: ... ... Finally, please create a head-to-head profit-loss matrix table for top 10 companies in the tech sector.

[YOU (AI assistant)]:
# Output:
---------------- Pseudocode Generation: ----------------
<pseudocode>
// PSEUDOCODE TO CREATE A HEAD-TO-HEAD PROFIT-LOSS MATRIX TABLE FOR TOP 10 COMPANIES IN THE TECH SECTOR

// Import necessary customized functions
FROM decision_company IMPORT read_csv_file, create_zeros_array, create_dataframe, check_elements_in_list, iterate_rows, access_dataframe_loc, update_dataframe_loc

// Load the financial data from a CSV file
LOAD DATA FROM 'Tech_Sector_Financials.csv'

// Define the top 10 company names (assuming this is already done)
DEFINE top_10_company_names

// Initialize a 10x10 matrix with zeros to hold the profit-loss records
INITIALIZE profit_loss_matrix WITH 10x10 ZEROS

// Filter financial records to include only those of top 10 companies in the last five years
FILTER tech_financials FOR records WHERE
    Company is in top_10_company_names
    AND record is in the last five years

// Iterate over the filtered financial records to update profit-loss records
FOR EACH record IN tech_financials
    DETERMINE profit AND loss FROM record
    INCREMENT the corresponding cell in profit_loss_matrix
    BASED ON profit AND loss values

// Output the matrix to check the results
DISPLAY profit_loss_matrix

// Save the profit-loss matrix to a file for future reference
SAVE profit_loss_matrix TO FILE 'profit_loss_matrix.pkl'
</pseudocode>

---------------- Customized Functions Extraction: ----------------
<FUNCTIONS>read_csv_file,create_zeros_array,create_dataframe,check_elements_in_list,iterate_rows,access_dataframe_loc,update_dataframe_loc</FUNCTIONS>
################### Example Ends ###################

Please note that, you have to generate the correct pseudocode based on the last User-AI interaction. Also, the functions extracted between "<FUNCTIONS> EXTRACTED-FUNCTIONS</FUNCTIONS>" should match the functions imported from "decision_company" in the pseudocode above between "<pseudocode> YOUR-PSEUDOCODE </pseudocode>".

[YOU (AI assistant)]:
# Output:
---------------- Pseudocode Generation: ----------------
<pseudocode>
"
"""

private_func_names_AIR_02 = """
{head}
--- Interaction History: ---
{dialogue_hist}

--- Task: ---

Analyze the provided custom function library named 'decision_company', which is enclosed within the BEGIN and END markers.
Please extract and output the function names based on the content between the following markers:

------------------------------------- decision_company BEGIN: -------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from scipy import stats

def read_csv_file(file_path):
    '''
    This function reads a CSV file and returns a pandas DataFrame.

    :param file_path: file path of the CSV file
    :return: pandas DataFrame
    '''
    return pd.read_csv(file_path)

def create_dataframe(data, columns=None, index=None):
    '''
    This function creates a new DataFrame object from the given data.

    :param data: data to be used for creating the DataFrame
    :param columns: list of column names for the DataFrame (optional)
    :param index: list of index labels for the DataFrame (optional)
    :return: pandas DataFrame
    '''
    return pd.DataFrame(data, columns=columns, index=index)

def is_null(dataframe):
    '''
    This function is used to detect missing values in the input DataFrame and returns a boolean mask of the same shape.
    True indicates a missing value (NaN), and False indicates a non-missing value.

    :param dataframe: pandas DataFrame
    :return: DataFrame with boolean mask indicating missing values
    '''
    return dataframe.isnull()

def sum_up(dataframe):
    '''
    This function is used to sum the values along a specified axis (default is 0).
    In this case, it is used to count the number of missing values in each column of the DataFrame.

    :param dataframe: pandas DataFrame
    :return: Series with the sum of values along the specified axis
    '''
    return dataframe.sum()

def n_unique(dataframe, columns=None):
    '''
    This function is used to count the number of unique values in each column of the input DataFrame.
    If a list of columns is provided, it will only count unique values in those columns.

    :param dataframe: pandas DataFrame
    :param columns: list of column names to count unique values (default is None)
    :return: Series with the number of unique values in each column
    '''
    if columns:
        dataframe = dataframe[columns]
    return dataframe.nunique()

def count_unique_values(series, normalize=False, sort=True, ascending=False, bins=None, dropna=True):
    '''
    Return a pandas series containing the counts of unique values in the input series.

    :param series: The input pandas series to count unique values on.
    :param normalize: Whether to return relative frequencies of the unique values. Default is False.
    :param sort: Whether to sort the resulting counts by frequency. Default is True.
    :param ascending: Whether to sort the resulting counts in ascending order. Default is False.
    :param bins: If passed, will group the counts into the specified number of bins. Default is None.
    :param dropna: Whether to exclude missing values from the counts. Default is True.
    :return: A pandas series containing the counts of unique values in the input series.
    '''
    return series.value_counts(normalize=normalize, sort=sort, ascending=ascending, bins=bins, dropna=dropna)

def reset_index(dataframe, drop=True):
    '''
    This function is used to reset the index of a DataFrame.

    :param dataframe: pandas DataFrame
    :param drop: boolean, whether to drop the old index and not add it as a new column (default is True)
    :return: DataFrame with reset index
    '''
    return dataframe.reset_index(drop=drop)

def concatenate_objects(obj1, obj2, ignore_index=False, reset_index_flag=True):
    '''
    Concatenate two pandas objects along a particular axis and reset the index if specified.

    :param obj1: first pandas object (Series or DataFrame)
    :param obj2: second pandas object (Series or DataFrame)
    :param reset_index_flag: boolean, whether to reset the index or not (default: True)
    :return: concatenated pandas object (Series or DataFrame)
    '''
    concatenated = pd.concat([obj1, obj2], ignore_index=ignore_index)
    if reset_index_flag:
        concatenated = reset_index(concatenated, drop=True)
    return concatenated

def generate_summary_stat(dataframe):
    '''
    Generate various summary statistics of a DataFrame or Series.

    :param dataframe: pandas DataFrame or Series
    :return: pandas DataFrame with summary statistics
    '''
    return dataframe.describe()

def transform(series, mapping):
    '''
    Map values from one Series to another based on a function or a dictionary.

    :param series: pandas Series
    :param mapping: function or dictionary to map values
    :return: pandas Series with mapped values
    '''
    return series.map(mapping)

def create_subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, width_ratios=None, height_ratios=None, subplot_kw=None, gridspec_kw=None, **fig_kw):
    '''
    Create a new figure and a set of subplots.

    :return: tuple containing a figure and an axis
    '''
    return plt.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, squeeze=squeeze, width_ratios=width_ratios, height_ratios=height_ratios, subplot_kw=subplot_kw, gridspec_kw=gridspec_kw, **fig_kw)

def create_histogram_subplot(ax, data, bins, alpha):
    '''
    Create a histogram using the given axis, data, bins, and alpha (transparency).

    :param ax: axis object
    :param data: data for the histogram
    :param bins: number of bins for the histogram
    :param alpha: transparency value (0 to 1)
    '''
    ax.hist(data, bins=bins, alpha=alpha)

def create_scatter_plot(ax, x_data, y_data, alpha):
    '''
    Create a scatter plot using the given axis, x_data, y_data, and alpha (transparency).

    :param ax: axis object
    :param x_data: data for the x-axis
    :param y_data: data for the y-axis
    :param alpha: transparency value (0 to 1)
    '''
    ax.scatter(x_data, y_data, alpha=alpha)

def create_bar_chart(ax, data):
    '''
    Create a bar chart using the given axis and data.

    :param ax: axis object
    :param data: data for the bar chart
    '''
    data.plot.bar(ax=ax)

def set_plot_split_title(ax, title):
    '''
    Set the title of a plot.

    :param ax: axis object
    :param title: title for the plot
    '''
    ax.set_title(title)

def make_xlabel(ax, xlabel):
    '''
    Set the x-axis label of a plot.

    :param ax: axis object
    :param xlabel: label for the x-axis
    '''
    ax.set_xlabel(xlabel)

def make_ylabel(ax, ylabel):
    '''
    Set the y-axis label of a plot.

    :param ax: axis object
    :param ylabel: label for the y-axis
    '''
    ax.set_ylabel(ylabel)

def show_plots():
    '''
    Display the created plots.
    '''
    plt.show()

def df_copy(dataframe):
    '''
    Creates a new DataFrame with the same data as the original DataFrame.

    :param dataframe: original pandas DataFrame
    :return: new pandas DataFrame
    '''
    return dataframe.copy()

def dropna(dataframe, subset_columns):
    '''
    Removes rows with missing or invalid data in the specified subset of columns.

    :param dataframe: pandas DataFrame
    :param subset_columns: list of column names to check for missing or invalid data
    :return: pandas DataFrame with rows removed
    '''
    return df_copy(dataframe.dropna(subset=subset_columns))

def get_dummies(data, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False, dtype=None):
    '''
    Performs one-hot encoding on the specified columns.

    :param dataframe: pandas DataFrame
    :param columns: list of column names to be one-hot encoded
    :param prefix: string to append before the new column names
    :param prefix_sep: string to separate the prefix from the new column names
    :return: pandas DataFrame with one-hot encoded columns
    '''
    return pd.get_dummies(data, prefix=prefix, prefix_sep=prefix_sep, dummy_na=dummy_na, columns=columns, sparse=sparse, drop_first=drop_first, dtype=dtype)

def access_dataframe_loc(dataframe, row_label, col_label):
    '''
    This function accesses a value in the DataFrame using row and column labels.

    :param dataframe: pandas DataFrame to access
    :param row_label: row label for the value to access
    :param col_label: column label for the value to access
    :return: value at the specified row and column
    '''
    return dataframe.loc[row_label, col_label]

def corr(dataframe):
    '''
    Creates a correlation matrix of the specified DataFrame columns.

    :param dataframe: pandas DataFrame
    :return: pandas DataFrame representing the correlation matrix
    '''
    return dataframe.corr()

def avg(dataframe):
    '''
    Calculates the average value of a dataset.

    :param dataframe: pandas DataFrame or Series
    :return: mean (average) value
    '''
    return dataframe.mean()

def f_oneway(*args):
    '''
    Performs a one-way ANOVA (Analysis of Variance) test to compare the means of multiple groups.

    :param *args: datasets (arrays, lists, or pandas Series) to be compared
    :return: F-statistic and p-value of the ANOVA test
    '''
    return stats.f_oneway(*args)

def sem(dataframe):
    '''
    Calculates the standard error of the mean (SEM) for a given dataset.

    :param dataframe: pandas DataFrame or Series
    :return: standard error of the mean (SEM)
    '''
    return stats.sem(dataframe)

def t_ppf(probability, degrees_of_freedom):
    '''
    Calculates the inverse of the Student's t-distribution cumulative distribution function (CDF)
    for a given probability and degrees of freedom.

    :param probability: probability to be used for the inverse CDF calculation
    :param degrees_of_freedom: degrees of freedom for the t-distribution
    :return: critical t-value
    '''
    return stats.t.ppf(probability, degrees_of_freedom)

def bind_dataframe(dataframe, column):
    '''
    Group the DataFrame by a specific column.
    :param dataframe: The input DataFrame to group
    :param column: The column to group by
    :return: A DataFrameGroupBy object
    '''
    return dataframe.groupby(column)

def aggregate_grouped_data(grouped_data, agg_dict):
    '''
    Apply aggregation functions on the grouped data.
    :param grouped_data: A DataFrameGroupBy object to apply the aggregation on
    :param agg_dict: A dictionary containing the aggregation functions for each column
    :return: A DataFrame with the aggregated data
    '''
    return grouped_data.agg(agg_dict).reset_index()

def make_bins(x, bins, right=True, labels=None, retbins=False, precision=3, include_lowest=False, duplicates='raise', ordered=True):
    '''
    Divide data into bins based on specified intervals and labels.
    :param data: The input data to be binned
    :param bins: A list of intervals to define the bins
    :param labels: A list of labels for each bin
    :return: A Series containing the binned data
    '''
    return pd.cut(x, bins, right=right, labels=labels, retbins=retbins, precision=precision, include_lowest=include_lowest, duplicates=duplicates, ordered=ordered)

def join_dataframes(df1,right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=None, indicator=False, validate=None):
    '''
    Merge two DataFrames based on common columns.
    :param df1: The first DataFrame to merge
    :param df2: The second DataFrame to merge
    :param left_on: Column or index level names to join on in the left DataFrame
    :param right_on: Column or index level names to join on in the right DataFrame
    :param how: Type of merge to be performed (default: 'inner')
    :return: A merged DataFrame
    '''
    return df1.merge(right, how=how, on=on, left_on=left_on, right_on=right_on, left_index=left_index, right_index=right_index, sort=sort, suffixes=suffixes, copy=copy, indicator=indicator, validate=validate)

def create_figure(figsize):
    '''
    This function creates a new figure with the specified figsize.
    :param figsize: A tuple of the width and height of the figure in inches
    :return: None
    '''
    plt.figure(figsize=figsize)

def create_barplot(x, y, hue, data):
    '''
    Create a bar plot based on the given data.
    :param x: The data to be used for the x-axis
    :param y: The data to be used for the y-axis
    :param hue: The variable to be used for distinguishing the bars in the plot
    :param data: The DataFrame containing the x, y, and hue data
    :return: An Axes object containing the bar plot
    '''
    return sns.barplot(x=x, y=y, hue=hue, data=data)

def set_plot_title(title):
    '''
    Set the title for the current plot.
    :param title: The title string
    :return: None
    '''
    plt.title(title)

def set_yaxis_label(label):
    '''
    Set the label for the y-axis.
    :param label: The label string
    :return: None
    '''
    plt.ylabel(label)

def anova_test(data1, data2, data3):
    '''
    Perform one-way ANOVA (analysis of variance) test on the given data.
    :param data1: The first dataset for the ANOVA test
    :param data2: The second dataset for the ANOVA test
    :param data3: The third dataset for the ANOVA test
    :return: An F_onewayResult object containing the results of the ANOVA test
    '''
    return stats.f_oneway(data1, data2, data3)

def positive_infinity():
    '''
    A infinity constant for computation.
    Return a positive infinity float value.
    :return: A float representing positive infinity
    '''
    return np.inf

def convert_to_datetime(input_data):
    '''
    This function converts the given input into datetime format.

    :param input_data: input data to be converted
    :return: datetime object
    '''
    return pd.to_datetime(input_data)

def col_copy(df, columns):
    '''
    This function creates a copy of the selected columns from a DataFrame.
    :param df: The input DataFrame to copy columns from
    :param columns: A list of column names to copy
    :return: A DataFrame containing the copied columns
    '''
    return df[columns].copy()

def extract_year(df, column):
    '''
    This function extracts the year from a specified datetime column in a DataFrame.
    :param df: The input DataFrame containing the datetime column
    :param column: The datetime column to extract the year from
    :return: A Series containing the extracted year values
    '''
    return df[column].dt.year

def aggregate_data(grouped_data, agg_dict):
    '''
    This function applies aggregation functions on a grouped DataFrame.
    :param grouped_data: A DataFrameGroupBy object to apply the aggregation on
    :param agg_dict: A dictionary containing the aggregation functions for each column
    :return: A DataFrame with the aggregated data
    '''
    return grouped_data.agg(agg_dict)

def draw_lineplot(x, y, hue=None, data=None, ci=None):
    '''
    This function creates a line plot based on the given data.
    :param x: The data to be used for the x-axis
    :param y: The data to be used for the y-axis
    :param hue: The variable to be used for distinguishing the lines in the plot
    :param data: The DataFrame containing the x, y, and hue data
    :param ci: The size of the confidence intervals for the lines
    :return: None
    '''
    sns.lineplot(x=x, y=y, hue=hue, data=data, errorbar=ci)

def set_plot_ylabel(label):
    '''
    This function sets the label for the y-axis of the current plot.
    :param label: The label string
    :return: None
    '''
    plt.ylabel(label)

def set_plot_xlabel(label):
    '''
    This function sets the label for the x-axis of the current plot.
    :param label: The label string
    :return: None
    '''
    plt.xlabel(label)

def linear_regression(x, y):
    '''
    This function performs a linear regression on the given data.
    :param x: The independent variable data (x-axis)
    :param y: The dependent variable data (y-axis)
    :return: A LinregressResult object containing the results of the linear regression
    '''
    return stats.linregress(x, y)

def save_plot(filename, dpi=100, bbox_inches='tight'):
    '''
    This function saves the current plot as an image file.

    :param filename: The name of the image file to be saved
    :param dpi: The resolution of the image in dots per inch (default: 100)
    :param bbox_inches: The bounding box in inches (default: 'tight')
    :return: None
    '''
    plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)

def create_date_offset(years):
    '''
    This function creates a date offset object for the given duration (in years).

    :param years: duration in years
    :return: date offset object
    '''
    return pd.DateOffset(years=years)

def get_min_value(data):
    '''
    This function finds the minimum value in each group of the grouped data.

    :param data: grouped pandas DataFrame
    :return: DataFrame with minimum values
    '''
    return data.min()

def sort_by_values(data, column_name):
    '''
    This function sorts the data by the values in the specified column.

    :param data: pandas DataFrame
    :param column_name: column name to sort by
    :return: sorted DataFrame
    '''
    return data.sort_values(column_name)

def get_first_n_rows(data, n=5):
    '''
    This function returns the first 'n' rows of the data.

    :param data: pandas DataFrame
    :param n: number of rows to return
    :return: DataFrame with first 'n' rows
    '''
    return data.head(n)

def get_max(data):
    '''
    This function finds the maximum value in the given data.

    :param data: pandas DataFrame or Series
    :return: maximum value
    '''
    return data.max()

def filter_by_date(data, date_col, min_date):
    '''
    This function filters the DataFrame based on the minimum date value.

    :param data: pandas DataFrame
    :param date_col: date column name
    :param min_date: minimum date value
    :return: filtered DataFrame
    '''
    return data[data[date_col] >= min_date]

def check_elements_in_list(series, values_list):
    '''
    This function checks whether each element in the series is contained in the given list or not.
    It returns a boolean Series of the same shape as the original Series.

    :param series: pandas Series
    :param values_list: list of values to check for
    :return: boolean Series
    '''
    return series.isin(values_list)

def iterate_rows(dataframe):
    '''
    This function iterates over the rows of the DataFrame as (index, Series) pairs.

    :param dataframe: pandas DataFrame to be iterated
    :return: iterator yielding index and row data as pandas Series
    '''
    return dataframe.iterrows()

def update_dataframe_loc(dataframe, row_label, col_label, value):
    '''
    This function updates a value in the DataFrame using row and column labels.

    :param dataframe: pandas DataFrame to update
    :param row_label: row label for the value to update
    :param col_label: column label for the value to update
    :param value: new value to set at the specified row and column
    '''
    dataframe.loc[row_label, col_label] = value

def convert_to_np_array(df_or_series):
    '''
    This function returns a NumPy array representation of the given DataFrame or Series.

    :param df_or_series: The input DataFrame or Series
    :return: numpy ndarray
    '''
    return df_or_series.values

def convert_np_to_list(numpy_array):
    '''
    This function converts the given NumPy array to a Python list.

    :param numpy_array: The input NumPy array
    :return: list
    '''
    return numpy_array.tolist()

def to_list(data):
    '''
    This function converts the DataFrame values to a Python list.

    :param data: pandas DataFrame or Series
    :return: list of values
    '''
    return data.values.tolist()

def rename_columns(data, columns):
    '''
    This function renames the columns in the given DataFrame.

    :param data: pandas DataFrame
    :param columns: dictionary with old column names as keys and new column names as values
    :return: DataFrame with renamed columns
    '''
    return data.rename(columns=columns)

def fill_missing_values(series, value=None, method=None, axis=None, inplace=False, limit=None):
    '''
    This function is used to fill NaN (Not a Number) values with a specified value or method.

    :param series: pandas Series
    :param value: value to fill NaNs with (default is 0)
    :return: Series with NaNs filled
    '''
    return series.fillna(value=value, method=method, axis=axis, inplace=inplace, limit=limit)

def create_zeros_array(shape):
    '''
    This function creates a new array of the specified shape and type, filled with zeros.

    :param shape: tuple of integers defining the shape of the new array
    :return: numpy array filled with zeros
    '''
    return np.zeros(shape)

def assert_series(series, value, operation):
    '''
    Compares the input Series with the given value using the specified operation.

    :param series: The input Series to compare.
    :param value: The value to compare the Series with.
    :param operation: The comparison operation to use.
    :return: A boolean Series resulting from the comparison.
    '''
    if operation == 'equality':
        return series == value
    elif operation == 'inequality':
        return series != value
    else:
        raise ValueError("Invalid operation. Supported operations are '==' and '!='.")

def logical_and(series1, series2):
    '''
    Performs a logical AND operation between two boolean Series.

    :param series1: The first input boolean Series.
    :param series2: The second input boolean Series.
    :return: A boolean Series resulting from the logical AND operation.
    '''
    return series1 & series2

def logical_or(series1, series2):
    '''
    Performs a logical OR operation between two boolean Series.

    :param series1: The first input boolean Series.
    :param series2: The second input boolean Series.
    :return: A boolean Series resulting from the logical OR operation.
    '''
    return series1 | series2

def fetch_index(dataframe):
    '''
    This function returns the index of the given DataFrame.

    :param dataframe: pandas DataFrame
    :return: pandas Index object
    '''
    return dataframe.index

def is_a_null_df(dataframe):
    '''
    Check if a DataFrame or Series is empty.

    :param dataframe: pandas DataFrame or Series
    :return: boolean, True if empty, False otherwise
    '''
    return dataframe.empty

def visit_by_index(dataframe, index):
    '''
    Access elements in a DataFrame or Series by index.

    :param dataframe: pandas DataFrame or Series
    :param index: integer, index of the element to access
    :return: element at the specified index
    '''
    return dataframe.iloc[index]

def fetch_df_size(dataframe):
    '''
    Return the size of the input DataFrame or DataFrameGroupBy object.

    :param dataframe: The input DataFrame or DataFrameGroupBy object.
    :return: A Series representing the size of the input object.
    '''
    if isinstance(dataframe, pd.core.groupby.generic.DataFrameGroupBy):
        return dataframe.size()
    else:
        return pd.Series(dataframe.size)

def create_multiindex_from_product(iterables, names):
    '''
    create a MultiIndex from the cartesian product of multiple iterables.

    Params:
    iterables: List of iterables to compute the cartesian product.
    names: List of names for the resulting MultiIndex levels.

    Returns:
    A MultiIndex object.
    '''
    return pd.MultiIndex.from_product(iterables, names=names)

def convert_multiindex_to_dataframe(multiindex, index=False):
    '''
    convert a MultiIndex to a DataFrame.

    Params:
    multiindex: A pandas MultiIndex object.
    index: Whether to include the index in the resulting DataFrame (default: False).

    Returns:
    A DataFrame object.
    '''
    return multiindex.to_frame(index=index)

def remove_labels(data, columns, axis=1):
    '''
    drop specified labels from rows or columns.

    Params:
    data: A pandas DataFrame or Series object.
    columns: Labels to drop.
    axis: Axis along which the labels will be dropped (default: 1).

    Returns:
    A DataFrame or Series with the specified labels dropped.
    '''
    return data.drop(columns=columns, axis=axis)

def draw_countplot(x, data):
    '''
    Shows the counts of observations in each categorical bin using bars.
    :param x: string, column name in the data
    :param data: DataFrame, input data
    '''
    sns.countplot(x=x, data=data)

def plot_kde(data, label, shade):
    '''
    Plots a kernel density estimate of the given data.
    :param data: array-like, input data
    :param label: string, label for the plot
    :param shade: bool, whether to shade the area under the curve
    '''
    sns.kdeplot(data, label=label, fill=shade)

def select_data_types(df, dtype_list):
    '''
    This function takes a pandas DataFrame and returns a DataFrame containing only the categorical columns.

    :param df: pandas DataFrame
    :return: pandas DataFrame with categorical columns
    '''
    return df.select_dtypes(include=dtype_list)

def get_columns(df):
    '''
    This function takes a pandas DataFrame and returns its columns.

    :param df: pandas DataFrame
    :return: columns (pandas Index object)
    '''
    return df.columns

def create_standard_scaler():
    '''
    This function creates a StandardScaler instance.

    :return: StandardScaler instance
    '''
    return StandardScaler()

def fit_transform_standard_scaler(scaler, data):
    '''
    This function fits the StandardScaler to the data and transforms the data using the fitted StandardScaler.

    :param scaler: StandardScaler instance
    :param data: Data to be standardized
    :return: Standardized data
    '''
    return scaler.fit_transform(data)

def create_label_encoder():
    '''
    This function creates a LabelEncoder instance.

    :return: LabelEncoder instance
    '''
    return LabelEncoder()

def fit_transform_label_encoder(le, data):
    '''
    This function fits the LabelEncoder to the data and transforms the data using the fitted LabelEncoder.

    :param le: LabelEncoder instance
    :param data: Data to be label encoded
    :return: Label encoded data
    '''
    return le.fit_transform(data)

def create_kmeans(n_clusters=8, init='k-means++', n_init='warn', max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='lloyd'):
    '''
    This function creates a KMeans instance with the specified number of clusters.

    :param cluster_num: Number of clusters
    :param random_state: Random state for the KMeans algorithm
    :return: KMeans instance
    '''
    return KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol, verbose=verbose, random_state=random_state, copy_x=copy_x, algorithm=algorithm)

def fit_predict_kmeans(kmeans, data):
    '''
    This function fits the KMeans instance to the data and predicts the cluster index for each sample.

    :param kmeans: KMeans instance
    :param data: Data to be clustered
    :return: Cluster index for each sample
    '''
    return kmeans.fit_predict(data)

def get_silhouette_score(X, labels, metric='euclidean', sample_size=None, random_state=None, **kwds):
    '''
    This function Compute the mean Silhouette Coefficient of all samples.

    :param X: An array of pairwise distances between samples, or a feature array.
    :param labels: Predicted labels for each sample.
    :param metric: The metric to use when calculating distance between instances in a feature array. If metric is a string, it must be one of the options allowed by pairwise_distances. If X is the distance array itself, use
    :param sample_size: The size of the sample to use when computing the Silhouette Coefficient on a random subset of the data.
    :param random_state: Determines random number generation for selecting a subset of samples.
    :return: Mean Silhouette Coefficient for all samples.
    '''
    return silhouette_score(X, labels, metric=metric, sample_size=sample_size, random_state=random_state, **kwds)

def plot(*args, scalex=True, scaley=True, data=None, **kwargs):
    '''
    Plot y versus x as lines and/or markers.

    :param x, y: The horizontal / vertical coordinates of the data points. x values are optional and default to range(len(y)).
    :param **kwargs: Line2D properties, optional
    :param scalex, scaley: These parameters determine if the view limits are adapted to the data limits. The values are passed on to autoscale_view.
    :return: list of Line2D A list of lines representing the plotted data.
    '''
    return plt.plot(*args, scalex=scalex, scaley=scaley, data=data, **kwargs)

def grid(visible=None, which='major', axis='both', **kwargs):
    '''
    Configure the grid lines.

    :param visible: Whether to show the grid lines. If any kwargs are supplied, it is assumed you want the grid on and visible will be set to True.
    :param **kwargs: Line2D properties: Define the line properties of the grid
    :param which: The grid lines to apply the changes on.
    :param axis: The axis to apply the changes on
    :return: list of Line2D A list of lines representing the plotted data.
    '''
    return plt.grid(visible=visible, which=which, axis=axis, **kwargs)

def col_assign_val(df, col, val):
    '''
    Assign a value to a specified column in a DataFrame.

    :param df: pandas.DataFrame The input DataFrame to be modified
    :param col: str The column name in the DataFrame to assign the value
    :param val: The value to be assigned to the specified column
    :return: pandas.Series The modified column with the assigned value
    '''
    df[col] = val

def extract_unique_values(series):
    '''
    This function extracts unique values from the given pandas Series.

    :param series: pandas Series
    :return: series of unique values
    '''
    return series.unique()

def series_get_quantile(series, q=0.5, interpolation='linear'):

  '''
  Calculate the quantile value of a pandas series using linear interpolation.

  :param series: The pandas series to calculate the quantile value on.
  :param q: The quantile value to calculate. Default is 0.5.
  :param interpolation: The interpolation method to use. Default is 'linear'.
  :return: The quantile value of the series at the given quantile and interpolation.
  '''
  return series.quantile(q=q, interpolation=interpolation)

def series_value_counts(series, normalize=False, sort=True, ascending=False, bins=None, dropna=True):
  '''
  Return a pandas series containing the counts of unique values in the input series.

  :param series: The input pandas series to count unique values on.
  :param normalize: Whether to return relative frequencies of the unique values. Default is False.
  :param sort: Whether to sort the resulting counts by frequency. Default is True.
  :param ascending: Whether to sort the resulting counts in ascending order. Default is False.
  :param bins: If passed, will group the counts into the specified number of bins. Default is None.
  :param dropna: Whether to exclude missing values from the counts. Default is True.
  :return: A pandas series containing the counts of unique values in the input series.
  '''
  return series.value_counts(normalize=normalize, sort=sort, ascending=ascending, bins=bins, dropna=dropna)

def series_to_dict(series):
  '''
  Convert a pandas series into a dictionary.

  :param series: The pandas series to convert into a dictionary.
  :return: A dictionary with the index values of the series as the keys and the corresponding values of the series as the values.
  '''
  return series.to_dict()

def choose_data_types(df, dtype_list):
    '''
    This function takes a pandas DataFrame and returns a DataFrame containing only the categorical columns.

    :param df: pandas DataFrame
    :return: pandas DataFrame with categorical columns
    '''
    return df.select_dtypes(include=dtype_list)

def get_columns(df):
    '''
    This function takes a pandas DataFrame and returns its columns.

    :param df: pandas DataFrame
    :return: columns (pandas Index object)
    '''
    return df.columns

def standard_scaler_instance():
    '''
    This function creates a StandardScaler instance.

    :return: StandardScaler instance
    '''
    return StandardScaler()

def modify_data_standard_scaler(scaler, data):
    '''
    This function fits the StandardScaler to the data and transforms the data using the fitted StandardScaler.

    :param scaler: StandardScaler instance
    :param data: Data to be standardized
    :return: Standardized data
    '''
    return scaler.fit_transform(data)

def encoder_instance():
    '''
    This function creates a LabelEncoder instance.

    :return: LabelEncoder instance
    '''
    return LabelEncoder()

def encode_column(le, data):
    '''
    This function fits the LabelEncoder to the data and transforms the data using the fitted LabelEncoder.

    :param le: LabelEncoder instance
    :param data: Data to be label encoded
    :return: Label encoded data
    '''
    return le.fit_transform(data)

def kmeans_instance(n_clusters=8, init='k-means++', n_init='warn', max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='lloyd'):
    '''
    This function creates a KMeans instance with the specified number of clusters.

    :param cluster_num: Number of clusters
    :param random_state: Random state for the KMeans algorithm
    :return: KMeans instance
    '''
    return KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol, verbose=verbose, random_state=random_state, copy_x=copy_x, algorithm=algorithm)

def predict_data_kmeans(kmeans, data):
    '''
    This function fits the KMeans instance to the data and predicts the cluster index for each sample.

    :param kmeans: KMeans instance
    :param data: Data to be clustered
    :return: Cluster index for each sample
    '''
    return kmeans.fit_predict(data)

def get_coefficient_silhouette(X, labels, metric='euclidean', sample_size=None, random_state=None, **kwds):
    '''
    This function Compute the mean Silhouette Coefficient of all samples.

    :param X: An array of pairwise distances between samples, or a feature array.
    :param labels: Predicted labels for each sample.
    :param metric: The metric to use when calculating distance between instances in a feature array. If metric is a string, it must be one of the options allowed by pairwise_distances. If X is the distance array itself, use
    :param sample_size: The size of the sample to use when computing the Silhouette Coefficient on a random subset of the data.
    :param random_state: Determines random number generation for selecting a subset of samples.
    :return: Mean Silhouette Coefficient for all samples.
    '''
    return silhouette_score(X, labels, metric=metric, sample_size=sample_size, random_state=random_state, **kwds)

def get_figure(*args, scalex=True, scaley=True, data=None, **kwargs):
    '''
    Plot y versus x as lines and/or markers.

    :param x, y: The horizontal / vertical coordinates of the data points. x values are optional and default to range(len(y)).
    :param **kwargs: Line2D properties, optional
    :param scalex, scaley: These parameters determine if the view limits are adapted to the data limits. The values are passed on to autoscale_view.
    :return: list of Line2D A list of lines representing the plotted data.
    '''
    plt.plot(*args, scalex=scalex, scaley=scaley, data=data, **kwargs)

def configure_gridlines(visible=None, which='major', axis='both', **kwargs):
    '''
    Configure the grid lines.

    :param visible: Whether to show the grid lines. If any kwargs are supplied, it is assumed you want the grid on and visible will be set to True.
    :param **kwargs: Line2D properties: Define the line properties of the grid
    :param which: The grid lines to apply the changes on.
    :param axis: The axis to apply the changes on
    :return: list of Line2D A list of lines representing the plotted data.
    '''
    return plt.grid(visible=visible, which=which, axis=axis, **kwargs)

def col_assign_val(df, col, val):
    '''
    Assign a value to a specified column in a DataFrame.

    :param df: pandas.DataFrame The input DataFrame to be modified
    :param col: str The column name in the DataFrame to assign the value
    :param val: The value to be assigned to the specified column
    :return: pandas.Series The modified column with the assigned value
    '''
    df[col] = val

def extract_unique_values(series):
    '''
    This function extracts unique values from the given pandas Series.

    :param series: pandas Series
    :return: series of unique values
    '''
    return series.unique()

def calc_quantile_val(series, q=0.5, interpolation='linear'):

  '''
  Calculate the quantile value of a pandas series using linear interpolation.

  :param series: The pandas series to calculate the quantile value on.
  :param q: The quantile value to calculate. Default is 0.5.
  :param interpolation: The interpolation method to use. Default is 'linear'.
  :return: The quantile value of the series at the given quantile and interpolation.
  '''
  return series.quantile(q=q, interpolation=interpolation)

def transform_to_dictionary(series):
  '''
  Convert a pandas series into a dictionary.

  :param series: The pandas series to convert into a dictionary.
  :return: A dictionary with the index values of the series as the keys and the corresponding values of the series as the values.
  '''
  return series.to_dict()

def calculate_quantile(series, percentile):
    '''
    This function calculates the quantile for a pandas Series.

    :param series: pandas Series
    :param percentile: percentile value (0.75 for 75th percentile)
    :return: quantile value
    '''
    return series.quantile(percentile)

def filter_by_condition(dataframe, condition):
    '''
    This function filters a pandas DataFrame based on a given condition.

    :param dataframe: pandas DataFrame
    :param condition: boolean condition to filter the DataFrame
    :return: filtered pandas DataFrame
    '''
    return dataframe[condition]

def create_condition(series, value):
    '''
    This function creates a boolean condition for filtering a pandas DataFrame.

    :param series: pandas Series
    :param value: value to compare with the Series
    :return: boolean condition
    '''
    return series > value

def convert_to_tuples(dataframe):
    '''
    This function converts a pandas DataFrame to a list of tuples.

    :param dataframe: pandas DataFrame
    :return: list of tuples
    '''
    return dataframe.itertuples(index=True, name=None)

def filter_by_value(dataframe, column, value):
    '''
    This function filters a pandas DataFrame based on a specific value in a column.

    :param dataframe: pandas DataFrame
    :param column: column name in the DataFrame
    :param value: value to filter by in the column
    :return: filtered pandas DataFrame
    '''
    return dataframe[dataframe[column] == value]

def convert_to_list(index):
    '''
    This function converts a pandas Index to a list.

    :param index: pandas Index
    :return: list
    '''
    return list(index)

def calculate_median(dataframe, column):
    '''
    This function calculates the median for a specific column in a pandas DataFrame.

    :param dataframe: pandas DataFrame
    :param column: column name in the DataFrame
    :return: median value
    '''
    return dataframe[column].median()

def count_rows(dataframe):
    '''
    This function counts the number of rows in a pandas DataFrame.

    :param dataframe: pandas DataFrame
    :return: number of rows
    '''
    return dataframe.shape[0]

def locate_mode(series):
    '''
    This function calculates the mode of a series.

    :param series: input series
    :return: mode of the series
    '''
    return series.mode()

def get_n_row(df, n):
    '''
    This function returns the first row of a DataFrame.

    :param df: input DataFrame
    :return: first row of the DataFrame
    '''
    return df.iloc[n]

def set_layout(pad=1.08, h_pad=None, w_pad=None, rect=None):
    '''
    Adjust the padding between and around subplots.

    :param pad: Padding between the figure edge and the edges of subplots, as a fraction of the font size.
    :param h_pad, w_pad: Padding (height/width) between edges of adjacent subplots, as a fraction of the font size.
    :param rect: A rectangle in normalized figure coordinates into which the whole subplots area (including labels) will fit.
    :return: None
    '''
    plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad, rect=rect)

def get_fig_from_df(data, *args, **kwargs):
    '''
    This function plots a bar chart.

    :param data: The object for which the method is called.
    :return: None
    '''
    data.plot(*args, **kwargs)

def pivot_a_level(data, level=-1, fill_value=None):
    '''
    Pivot a level of the (necessarily hierarchical) index labels.

    :param level: Level(s) of index to unstack, can pass level name.
    :param fill_value: Replace NaN with this value if the unstack produces missing values.
    :param sort: Sort the level(s) in the resulting MultiIndex columns.
    :return: DataFrame after unstacking
    '''
    return data.unstack(level=level, fill_value=fill_value)

def cast_to_a_dtype(data, dtype, copy=None, errors='raise'):
    '''
    Cast a pandas object to a specified dtype dtype.

    :param dtype: Use a str, numpy.dtype, pandas.ExtensionDtype or Python type to cast entire pandas object to the same type. Alternatively, use a mapping, where col is a column label and dtype is a numpy.dtype or Python type to cast one or more of the DataFrame’s columns to column-specific types.
    :param copy: Return a copy when copy=True (be very careful setting copy=False as changes to values then may propagate to other pandas objects).
    :param errors: Control raising of exceptions on invalid data for provided dtype.

    :return: None
    '''
    return data.astype(dtype, copy=copy, errors=errors)

def categoricalIndex(data):
    '''
    Accessor object for categorical properties of the Series values.

    :return: None
    '''
    return data.cat

def  categorical_codes(cat):
    '''
    The category codes of this categorical index.

    :return: None
    '''
    return cat.codes

def scatter_fig_instance(data, x, y, ax):
    '''
    This function creates a scatterplot using seaborn.

    :param data: pandas DataFrame
    :param x: column name for the x-axis
    :param y: column name for the y-axis
    :param ax: axis object to plot the scatterplot on
    '''
    sns.scatterplot(data=data, x=x, y=y, ax=ax)

def divide_dataset(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None):
    '''
    Split arrays or matrices into random train and test subsets.

    :param *arrays: Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.
    :param test_size: If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. If None, the value is set to the complement of the train size. If train_size is also None, it will be set to 0.25.
    :param train_size: If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split. If int, represents the absolute number of train samples. If None, the value is automatically set to the complement of the test size.
    :param random_state: Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.
    :param shuffle: Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.
    :param stratify: If not None, data is split in a stratified fashion, using this as the class labels.

    :return: List containing train-test split of inputs.
    '''
    return train_test_split(*arrays, test_size=test_size, train_size=train_size, random_state=random_state, shuffle=shuffle, stratify=stratify)

def create_LR_instance(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None):
    '''
    This function initializes a Logistic Regression.

    :return: initialized Logistic Regression
    '''
    return LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs, l1_ratio=l1_ratio)

def classifier_training(classifier, X_train, y_train):
    '''
    This function trains a classifier on the given training data.

    :param classifier: classifier object
    :param X_train: training data features
    :param y_train: training data labels
    :return: trained classifier
    '''
    classifier.fit(X_train, y_train)
    return classifier

def classifier_predictions(classifier, X_test):
    '''
    This function makes predictions using a trained classifier.

    :param classifier: trained classifier object
    :param X_test: testing data features
    :return: predicted labels
    '''
    return classifier.predict(X_test)

def calculate_conf_mat(y_true, y_pred):
    '''
    This function calculates the confusion matrix.

    :param y_true: true labels
    :param y_pred: predicted labels
    :return: confusion matrix
    '''
    return confusion_matrix(y_true, y_pred)

def calc_acc(y_true, y_pred):
    '''
    This function calculates the accuracy score.

    :param y_true: true labels
    :param y_pred: predicted labels
    :return: accuracy score
    '''
    return accuracy_score(y_true, y_pred)

def create_barplot(data=None, x=None, y=None, hue=None, order=None, hue_order=None, estimator='mean', errorbar=('ci', 95), n_boot=1000, units=None, seed=None, orient=None, color=None, palette=None, saturation=0.75, fill=True, hue_norm=None, width=0.8, dodge='auto', gap=0, log_scale=None, native_scale=False, formatter=None, legend='auto', capsize=0, err_kws=None, ax=None, **kwargs):
    '''
    This function creates a bar plot using seaborn.

    :param data: DataFrame, array, or list of arrays, optional
    :param x, y, hue: names of variables in data or vector data, optional
    :param order, hue_order: lists of strings, optional
    :param estimator: statistical function to estimate within each categorical bin, optional
    :param errorbar: error bar to draw on bars, optional
    :param n_boot: number of bootstraps to compute the confidence interval, optional
    :param units: identifier of sampling units, optional
    :param seed: seed or random number generator, optional
    :param orient: orientation of the plot (vertical or horizontal), optional
    :param color: matplotlib color, optional
    :param palette: palette name, list, or dict, optional
    :param saturation: proportion of the original saturation to draw colors, optional
    :param fill: if True, fill the bars with the hue variable, optional
    :param hue_norm: tuple or matplotlib.colors.Normalize, optional
    :param width: width of a full element when not using hue nesting, or width of all the elements for one level of the major grouping variable, optional
    :param dodge: when hue nesting is used, whether elements should be shifted along the categorical axis, optional
    :param gap: size of the gap between bars, optional
    :param log_scale: boolean or number, or pair of boolean or number, optional
    :param native_scale: boolean, optional
    :param formatter: callable or string format, optional
    :param legend: boolean or 'auto', optional
    :param capsize: width of the caps on error bars, optional
    :param err_kws: keyword arguments for matplotlib.axes.Axes.errorbar(), optional
    :param ax: matplotlib Axes, optional
    :param kwargs: key, value pairings
    :return: matplotlib Axes
    '''
    return sns.barplot(data=data, x=x, y=y, hue=hue, order=order, hue_order=hue_order, estimator=estimator, errorbar=errorbar, n_boot=n_boot, units=units, seed=seed, orient=orient, color=color, palette=palette, saturation=saturation, fill=fill, hue_norm=hue_norm, width=width, dodge=dodge, gap=gap, log_scale=log_scale, native_scale=native_scale, formatter=formatter, legend=legend, capsize=capsize, err_kws=err_kws, ax=ax, **kwargs)

def create_histogram(data=None, x=None, y=None, hue=None, weights=None, stat='count', bins='auto', binwidth=None, binrange=None, discrete=None, cumulative=False, common_bins=True, common_norm=True, multiple='layer', element='bars', fill=True, shrink=1, kde=False, kde_kws=None, line_kws=None, thresh=0, pthresh=None, pmax=None, cbar=False, cbar_ax=None, cbar_kws=None, palette=None, hue_order=None, hue_norm=None, color=None, log_scale=None, legend=True, ax=None, **kwargs):
    '''
    This function creates a histogram plot using seaborn.

    :param data: DataFrame, array, or list of arrays, optional
    :param x, y, hue: names of variables in data or vector data, optional
    :param weights: array of weights, optional
    :param stat: aggregate statistic to compute in each bin, optional
    :param bins: specification of hist bins, or None to use Freedman-Diaconis rule, optional
    :param binwidth: width of each bin, optional
    :param binrange: lowest and highest value of bins, optional
    :param discrete: if True, draw a bar at each unique value in x, instead of using a histogram, optional
    :param cumulative: if True, the histogram accumulates values, optional
    :param common_bins: if True, the same bins will be used across all histograms, optional
    :param common_norm: if True, all densities on the same axes are scaled by the same factor, optional
    :param multiple: approach to resolving multiple elements when semantic mapping creates subsets, optional
    :param element: approach to drawing the histogram, optional
    :param fill: if True, fill in the bars in the histogram, optional
    :param shrink: scale factor for shrinking the density curve in rugplot and kde, optional
    :param kde: if True, compute a kernel density estimate to smooth the distribution and show on the plot, optional
    :param kde_kws: keyword arguments for kdeplot(), optional
    :param line_kws: keyword arguments for plot(), optional
    :param thresh: threshold for small elements, optional
    :param pthresh: threshold for small elements as a proportion, optional
    :param pmax: maximum proportion of the figure area that a small element can occupy, optional
    :param cbar: if True, add a colorbar to the figure, optional
    :param cbar_ax: axes in which to draw the colorbar, optional
    :param cbar_kws: keyword arguments for colorbar(), optional
    :param palette: palette name, list, or dict, optional
    :param hue_order: order for the levels of the hue variable in the palette, optional
    :param hue_norm: normalization in data units for colormap applied to the hue variable, optional
    :param color: color for all elements, or seed for a gradient palette, optional
    :param log_scale: boolean or number, or pair of boolean or number, optional
    :param legend: if True, add a legend or legend elements, optional
    :param ax: matplotlib Axes, optional
    :param kwargs: key, value pairings
    :return: matplotlib Axes
    '''
    return sns.histplot(data=data, x=x, y=y, hue=hue, weights=weights, stat=stat, bins=bins, binwidth=binwidth, binrange=binrange, discrete=discrete, cumulative=cumulative, common_bins=common_bins, common_norm=common_norm, multiple=multiple, element=element, fill=fill, shrink=shrink, kde=kde, kde_kws=kde_kws, line_kws=line_kws, thresh=thresh, pthresh=pthresh, pmax=pmax, cbar=cbar, cbar_ax=cbar_ax, cbar_kws=cbar_kws, palette=palette, hue_order=hue_order, hue_norm=hue_norm, color=color, log_scale=log_scale, legend=legend, ax=ax, **kwargs)

def create_countplot(data=None, x=None, y=None, hue=None, order=None, hue_order=None, orient=None, color=None, palette=None, saturation=0.75, fill=True, width=0.8, dodge='auto', ax=None, **kwargs):
    '''
    This function creates a count plot using seaborn.

    :param data: DataFrame, array, or list of arrays, optional
    :param x, y, hue: names of variables in data or vector data, optional
    :param order, hue_order: lists of strings, optional
    :param orient: orientation of the plot (vertical or horizontal), optional
    :param color: matplotlib color, optional
    :param palette: palette name, list, or dict, optional
    :param saturation: proportion of the original saturation to draw colors, optional
    :param fill: if True, fill the bars with the hue variable, optional
    :param hue_norm: tuple or matplotlib.colors.Normalize, optional
    :param stat: aggregate statistic to compute in each bin, optional
    :param width: width of a full element when not using hue nesting, or width of all the elements for one level of the major grouping variable, optional
    :param dodge: when hue nesting is used, whether elements should be shifted along the categorical axis, optional
    :param gap: size of the gap between bars, optional
    :param log_scale: boolean or number, or pair of boolean or number, optional
    :param native_scale: boolean, optional
    :param formatter: callable or string format, optional
    :param legend: boolean or 'auto', optional
    :param ax: matplotlib Axes, optional
    :param kwargs: key, value pairings
    :return: matplotlib Axes
    '''
    return sns.countplot(data=data, x=x, y=y, hue=hue, order=order, hue_order=hue_order, orient=orient, color=color, palette=palette, saturation=saturation, fill=fill, width=width, dodge=dodge, ax=ax, **kwargs)

def set_current_ticks(ticks=None, labels=None, minor=False, **kwargs):
    '''
    Get or set the current tick locations and labels of the x-axis.

    :param ticks: An array-like object containing the positions of the ticks. If None, the current ticks are used.
    :param labels: A list of string labels to use as the tick labels. If None, the current labels are used.
    :param minor: If True, the ticks will be minor ticks. If False, the ticks will be major ticks.
    :param **kwargs: Additional keyword arguments to be passed to the plt.xticks() function.
    :return: A tuple containing the x-location of the new ticks and a list of <class 'matplotlib.text.Text'> instances.
    '''

    return plt.xticks(ticks=ticks, labels=labels, minor=minor, **kwargs)

def create_heatmap(data, vmin=None, vmax=None, cmap=None, center=None, robust=False, annot=None, fmt='.2g', annot_kws=None, linewidths=0, linecolor='white', cbar=True, cbar_kws=None, cbar_ax=None, square=False, xticklabels='auto', yticklabels='auto', mask=None, ax=None, **kwargs):
    '''
    This function creates a heatmap using seaborn's heatmap function.

    :param data: 2D dataset that can be coerced into an ndarray. If a Pandas DataFrame is provided, the index/column information will be used to label the columns and rows.
    :param vmin, vmax: Values to anchor the colormap, otherwise they are inferred from the data and other keyword arguments.
    :param cmap: The mapping from data values to color space.
    :param center: The value at which to center the colormap when plotting divergant data.
    :param robust: If True and vmin or vmax are absent, the colormap range is computed with robust quantiles instead of the extreme values.
    :param annot: If True, write the data value in each cell.
    :param fmt: String formatting code to use when adding annotations.
    :param annot_kws: Keyword arguments for ax.text when annot is True.
    :param linewidths: Width of the lines that will divide each cell.
    :param linecolor: Color of the lines that will divide each cell.
    :param cbar: Whether to draw a colorbar.
    :param cbar_kws: Keyword arguments for fig.colorbar.
    :param cbar_ax: None or Axes, optional. Axes in which to draw the colorbar, otherwise take space from the main Axes.
    :param square: If True, set the Axes aspect to “equal” so each cell will be square-shaped.
    :param xticklabels, yticklabels: If True, plot the column names of the dataframe. If False, don’t plot the column names. If list-like, plot these alternate labels as the xticklabels. If an integer, use the column names but plot only every n label. If “auto”, try to densely plot non-overlapping labels.
    :param mask: If passed, data will not be shown in cells where mask is True. Cells with missing values are automatically masked.
    :param ax: If provided, plot the heatmap on this axis.
    :param kwargs: Other keyword arguments are passed down to ax.pcolormesh.

    :return: Axes object with the heatmap.
    '''
    return sns.heatmap(data, vmin=vmin, vmax=vmax, cmap=cmap, center=center, robust=robust, annot=annot, fmt=fmt, annot_kws=annot_kws, linewidths=linewidths, linecolor=linecolor, cbar=cbar, cbar_kws=cbar_kws, cbar_ax=cbar_ax, square=square, xticklabels=xticklabels, yticklabels=yticklabels, mask=mask, ax=ax, **kwargs)

def search_where(data, condition, column):  
    '''  
    Filters the data based on the given condition and returns distinct values from the specified column.  
  
    :param data: The DataFrame to search in.  
    :param condition: The boolean Series representing the condition to filter by.  
    :param column: The column from which to return values.  
    :return: A list of values from the specified column after applying the condition.  
    ''' 
  
    # Return the values from the specified column  
    return data.where(condition, column)

def update_dict(original_dict, updates):  
    '''  
    Updates the original dictionary with the key-value pairs from the updates dictionary.  
  
    :param original_dict: The original dictionary to be updated.  
    :param updates: The dictionary containing updates to be applied.  
    '''  
    original_dict.update(updates)

def use_function(df, func, axis=0, raw=False, result_type=None, args=()):
    '''
    This function apply function to pandas Series.

    :param series: pandas Series
    :return: series that with function applied
    '''
    return df.apply(func, axis=axis, raw=raw, result_type=result_type, args=args)

def add_legend():
    '''
    This function add legned to plot.
    '''
    plt.legend()
------------------------------------- decision_company END: -------------------------------------

The 'decision_company' is a custom function library provided to you. It contains various functions created by the user. You are asked to examine the User-AI interaction history and identify the line that starts with "from decision_company import read_csv_file," in the code generated under the "---BEGIN CODE TEMPLATE---". Your task is to understand the underlying pattern of how to extract and import custom functions from 'decision_company'. Below is an example of a function extract plan derived from the User-AI interaction history and the generated pseudocode.

################### Example Starts: ###################
{last_query}
# Output:
---------------- Pseudocode Generation: ----------------
<pseudocode>
{example}
################### Example Ends ###################

For this task, you need to extract useful customized functions from decision_company and fill into "<FUNCTIONS>...</FUNCTIONS>" for the New Query with a code snippet. To do this, first generate pseudocode and then analyze it to determine which customized functions to extract. You can refer to the example above for guidance on how to analyze and extract from the user query and generated pseudocode. However, you need to analyze and come up with a new pseudocode instead of simply copying example. You can refer to customized functions provided in "decision_company" above, DO NOT imagine any custom function name not in "decision_company"!

NOTE: Rank the extracted functions with their importance in descending order. For example, first function should be more important than function2, function3 ...

--- New Query: ---
{new_query}

[YOU (AI assistant)]:
# Output:
---------------- Pseudocode Generation: ----------------
<pseudocode>
"""

pre_prompt_func_names_user = """
{dialogue_hist}

Analyze the provided custom function library named 'decision_company', which is enclosed within the BEGIN and END markers.
Please extract and output the function names based on the content between the following markers:

------------------------------------- decision_company BEGIN: -------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from scipy import stats

def read_csv_file(file_path):
    '''
    This function reads a CSV file and returns a pandas DataFrame.

    :param file_path: file path of the CSV file
    :return: pandas DataFrame
    '''
    return pd.read_csv(file_path)

def create_dataframe(data, columns=None, index=None):
    '''
    This function creates a new DataFrame object from the given data.

    :param data: data to be used for creating the DataFrame
    :param columns: list of column names for the DataFrame (optional)
    :param index: list of index labels for the DataFrame (optional)
    :return: pandas DataFrame
    '''
    return pd.DataFrame(data, columns=columns, index=index)

def is_null(dataframe):
    '''
    This function is used to detect missing values in the input DataFrame and returns a boolean mask of the same shape.
    True indicates a missing value (NaN), and False indicates a non-missing value.

    :param dataframe: pandas DataFrame
    :return: DataFrame with boolean mask indicating missing values
    '''
    return dataframe.isnull()

def sum_up(dataframe):
    '''
    This function is used to sum the values along a specified axis (default is 0).
    In this case, it is used to count the number of missing values in each column of the DataFrame.

    :param dataframe: pandas DataFrame
    :return: Series with the sum of values along the specified axis
    '''
    return dataframe.sum()

def n_unique(dataframe, columns=None):
    '''
    This function is used to count the number of unique values in each column of the input DataFrame.
    If a list of columns is provided, it will only count unique values in those columns.

    :param dataframe: pandas DataFrame
    :param columns: list of column names to count unique values (default is None)
    :return: Series with the number of unique values in each column
    '''
    if columns:
        dataframe = dataframe[columns]
    return dataframe.nunique()

def count_unique_values(series, normalize=False, sort=True, ascending=False, bins=None, dropna=True):
    '''
    Return a pandas series containing the counts of unique values in the input series.

    :param series: The input pandas series to count unique values on.
    :param normalize: Whether to return relative frequencies of the unique values. Default is False.
    :param sort: Whether to sort the resulting counts by frequency. Default is True.
    :param ascending: Whether to sort the resulting counts in ascending order. Default is False.
    :param bins: If passed, will group the counts into the specified number of bins. Default is None.
    :param dropna: Whether to exclude missing values from the counts. Default is True.
    :return: A pandas series containing the counts of unique values in the input series.
    '''
    return series.value_counts(normalize=normalize, sort=sort, ascending=ascending, bins=bins, dropna=dropna)

def reset_index(dataframe, drop=True):
    '''
    This function is used to reset the index of a DataFrame.

    :param dataframe: pandas DataFrame
    :param drop: boolean, whether to drop the old index and not add it as a new column (default is True)
    :return: DataFrame with reset index
    '''
    return dataframe.reset_index(drop=drop)

def concatenate_objects(obj1, obj2, ignore_index=False, reset_index_flag=True):
    '''
    Concatenate two pandas objects along a particular axis and reset the index if specified.

    :param obj1: first pandas object (Series or DataFrame)
    :param obj2: second pandas object (Series or DataFrame)
    :param reset_index_flag: boolean, whether to reset the index or not (default: True)
    :return: concatenated pandas object (Series or DataFrame)
    '''
    concatenated = pd.concat([obj1, obj2], ignore_index=ignore_index)
    if reset_index_flag:
        concatenated = reset_index(concatenated, drop=True)
    return concatenated

def generate_summary_stat(dataframe):
    '''
    Generate various summary statistics of a DataFrame or Series.

    :param dataframe: pandas DataFrame or Series
    :return: pandas DataFrame with summary statistics
    '''
    return dataframe.describe()

def transform(series, mapping):
    '''
    Map values from one Series to another based on a function or a dictionary.

    :param series: pandas Series
    :param mapping: function or dictionary to map values
    :return: pandas Series with mapped values
    '''
    return series.map(mapping)

def create_subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, width_ratios=None, height_ratios=None, subplot_kw=None, gridspec_kw=None, **fig_kw):
    '''
    Create a new figure and a set of subplots.

    :return: tuple containing a figure and an axis
    '''
    return plt.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, squeeze=squeeze, width_ratios=width_ratios, height_ratios=height_ratios, subplot_kw=subplot_kw, gridspec_kw=gridspec_kw, **fig_kw)

def create_histogram_subplot(ax, data, bins, alpha):
    '''
    Create a histogram using the given axis, data, bins, and alpha (transparency).

    :param ax: axis object
    :param data: data for the histogram
    :param bins: number of bins for the histogram
    :param alpha: transparency value (0 to 1)
    '''
    ax.hist(data, bins=bins, alpha=alpha)

def create_scatter_plot(ax, x_data, y_data, alpha):
    '''
    Create a scatter plot using the given axis, x_data, y_data, and alpha (transparency).

    :param ax: axis object
    :param x_data: data for the x-axis
    :param y_data: data for the y-axis
    :param alpha: transparency value (0 to 1)
    '''
    ax.scatter(x_data, y_data, alpha=alpha)

def create_bar_chart(ax, data):
    '''
    Create a bar chart using the given axis and data.

    :param ax: axis object
    :param data: data for the bar chart
    '''
    data.plot.bar(ax=ax)

def set_plot_split_title(ax, title):
    '''
    Set the title of a plot.

    :param ax: axis object
    :param title: title for the plot
    '''
    ax.set_title(title)

def make_xlabel(ax, xlabel):
    '''
    Set the x-axis label of a plot.

    :param ax: axis object
    :param xlabel: label for the x-axis
    '''
    ax.set_xlabel(xlabel)

def make_ylabel(ax, ylabel):
    '''
    Set the y-axis label of a plot.

    :param ax: axis object
    :param ylabel: label for the y-axis
    '''
    ax.set_ylabel(ylabel)

def show_plots():
    '''
    Display the created plots.
    '''
    plt.show()

def df_copy(dataframe):
    '''
    Creates a new DataFrame with the same data as the original DataFrame.

    :param dataframe: original pandas DataFrame
    :return: new pandas DataFrame
    '''
    return dataframe.copy()

def dropna(dataframe, subset_columns):
    '''
    Removes rows with missing or invalid data in the specified subset of columns.

    :param dataframe: pandas DataFrame
    :param subset_columns: list of column names to check for missing or invalid data
    :return: pandas DataFrame with rows removed
    '''
    return df_copy(dataframe.dropna(subset=subset_columns))

def get_dummies(data, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False, dtype=None):
    '''
    Performs one-hot encoding on the specified columns.

    :param dataframe: pandas DataFrame
    :param columns: list of column names to be one-hot encoded
    :param prefix: string to append before the new column names
    :param prefix_sep: string to separate the prefix from the new column names
    :return: pandas DataFrame with one-hot encoded columns
    '''
    return pd.get_dummies(data, prefix=prefix, prefix_sep=prefix_sep, dummy_na=dummy_na, columns=columns, sparse=sparse, drop_first=drop_first, dtype=dtype)

def access_dataframe_loc(dataframe, row_label, col_label):
    '''
    This function accesses a value in the DataFrame using row and column labels.

    :param dataframe: pandas DataFrame to access
    :param row_label: row label for the value to access
    :param col_label: column label for the value to access
    :return: value at the specified row and column
    '''
    return dataframe.loc[row_label, col_label]

def corr(dataframe):
    '''
    Creates a correlation matrix of the specified DataFrame columns.

    :param dataframe: pandas DataFrame
    :return: pandas DataFrame representing the correlation matrix
    '''
    return dataframe.corr()

def avg(dataframe):
    '''
    Calculates the average value of a dataset.

    :param dataframe: pandas DataFrame or Series
    :return: mean (average) value
    '''
    return dataframe.mean()

def f_oneway(*args):
    '''
    Performs a one-way ANOVA (Analysis of Variance) test to compare the means of multiple groups.

    :param *args: datasets (arrays, lists, or pandas Series) to be compared
    :return: F-statistic and p-value of the ANOVA test
    '''
    return stats.f_oneway(*args)

def sem(dataframe):
    '''
    Calculates the standard error of the mean (SEM) for a given dataset.

    :param dataframe: pandas DataFrame or Series
    :return: standard error of the mean (SEM)
    '''
    return stats.sem(dataframe)

def t_ppf(probability, degrees_of_freedom):
    '''
    Calculates the inverse of the Student's t-distribution cumulative distribution function (CDF)
    for a given probability and degrees of freedom.

    :param probability: probability to be used for the inverse CDF calculation
    :param degrees_of_freedom: degrees of freedom for the t-distribution
    :return: critical t-value
    '''
    return stats.t.ppf(probability, degrees_of_freedom)

def bind_dataframe(dataframe, column):
    '''
    Group the DataFrame by a specific column.
    :param dataframe: The input DataFrame to group
    :param column: The column to group by
    :return: A DataFrameGroupBy object
    '''
    return dataframe.groupby(column)

def aggregate_grouped_data(grouped_data, agg_dict):
    '''
    Apply aggregation functions on the grouped data.
    :param grouped_data: A DataFrameGroupBy object to apply the aggregation on
    :param agg_dict: A dictionary containing the aggregation functions for each column
    :return: A DataFrame with the aggregated data
    '''
    return grouped_data.agg(agg_dict).reset_index()

def make_bins(x, bins, right=True, labels=None, retbins=False, precision=3, include_lowest=False, duplicates='raise', ordered=True):
    '''
    Divide data into bins based on specified intervals and labels.
    :param data: The input data to be binned
    :param bins: A list of intervals to define the bins
    :param labels: A list of labels for each bin
    :return: A Series containing the binned data
    '''
    return pd.cut(x, bins, right=right, labels=labels, retbins=retbins, precision=precision, include_lowest=include_lowest, duplicates=duplicates, ordered=ordered)

def join_dataframes(df1,right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=None, indicator=False, validate=None):
    '''
    Merge two DataFrames based on common columns.
    :param df1: The first DataFrame to merge
    :param df2: The second DataFrame to merge
    :param left_on: Column or index level names to join on in the left DataFrame
    :param right_on: Column or index level names to join on in the right DataFrame
    :param how: Type of merge to be performed (default: 'inner')
    :return: A merged DataFrame
    '''
    return df1.merge(right, how=how, on=on, left_on=left_on, right_on=right_on, left_index=left_index, right_index=right_index, sort=sort, suffixes=suffixes, copy=copy, indicator=indicator, validate=validate)

def create_figure(figsize):
    '''
    This function creates a new figure with the specified figsize.
    :param figsize: A tuple of the width and height of the figure in inches
    :return: None
    '''
    plt.figure(figsize=figsize)

def create_barplot(x, y, hue, data):
    '''
    Create a bar plot based on the given data.
    :param x: The data to be used for the x-axis
    :param y: The data to be used for the y-axis
    :param hue: The variable to be used for distinguishing the bars in the plot
    :param data: The DataFrame containing the x, y, and hue data
    :return: An Axes object containing the bar plot
    '''
    return sns.barplot(x=x, y=y, hue=hue, data=data)

def set_plot_title(title):
    '''
    Set the title for the current plot.
    :param title: The title string
    :return: None
    '''
    plt.title(title)

def set_yaxis_label(label):
    '''
    Set the label for the y-axis.
    :param label: The label string
    :return: None
    '''
    plt.ylabel(label)

def anova_test(data1, data2, data3):
    '''
    Perform one-way ANOVA (analysis of variance) test on the given data.
    :param data1: The first dataset for the ANOVA test
    :param data2: The second dataset for the ANOVA test
    :param data3: The third dataset for the ANOVA test
    :return: An F_onewayResult object containing the results of the ANOVA test
    '''
    return stats.f_oneway(data1, data2, data3)

def positive_infinity():
    '''
    A infinity constant for computation.
    Return a positive infinity float value.
    :return: A float representing positive infinity
    '''
    return np.inf

def convert_to_datetime(input_data):
    '''
    This function converts the given input into datetime format.

    :param input_data: input data to be converted
    :return: datetime object
    '''
    return pd.to_datetime(input_data)

def col_copy(df, columns):
    '''
    This function creates a copy of the selected columns from a DataFrame.
    :param df: The input DataFrame to copy columns from
    :param columns: A list of column names to copy
    :return: A DataFrame containing the copied columns
    '''
    return df[columns].copy()

def extract_year(df, column):
    '''
    This function extracts the year from a specified datetime column in a DataFrame.
    :param df: The input DataFrame containing the datetime column
    :param column: The datetime column to extract the year from
    :return: A Series containing the extracted year values
    '''
    return df[column].dt.year

def aggregate_data(grouped_data, agg_dict):
    '''
    This function applies aggregation functions on a grouped DataFrame.
    :param grouped_data: A DataFrameGroupBy object to apply the aggregation on
    :param agg_dict: A dictionary containing the aggregation functions for each column
    :return: A DataFrame with the aggregated data
    '''
    return grouped_data.agg(agg_dict)

def draw_lineplot(x, y, hue=None, data=None, ci=None):
    '''
    This function creates a line plot based on the given data.
    :param x: The data to be used for the x-axis
    :param y: The data to be used for the y-axis
    :param hue: The variable to be used for distinguishing the lines in the plot
    :param data: The DataFrame containing the x, y, and hue data
    :param ci: The size of the confidence intervals for the lines
    :return: None
    '''
    sns.lineplot(x=x, y=y, hue=hue, data=data, errorbar=ci)

def set_plot_ylabel(label):
    '''
    This function sets the label for the y-axis of the current plot.
    :param label: The label string
    :return: None
    '''
    plt.ylabel(label)

def set_plot_xlabel(label):
    '''
    This function sets the label for the x-axis of the current plot.
    :param label: The label string
    :return: None
    '''
    plt.xlabel(label)

def linear_regression(x, y):
    '''
    This function performs a linear regression on the given data.
    :param x: The independent variable data (x-axis)
    :param y: The dependent variable data (y-axis)
    :return: A LinregressResult object containing the results of the linear regression
    '''
    return stats.linregress(x, y)

def save_plot(filename, dpi=100, bbox_inches='tight'):
    '''
    This function saves the current plot as an image file.

    :param filename: The name of the image file to be saved
    :param dpi: The resolution of the image in dots per inch (default: 100)
    :param bbox_inches: The bounding box in inches (default: 'tight')
    :return: None
    '''
    plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)

def create_date_offset(years):
    '''
    This function creates a date offset object for the given duration (in years).

    :param years: duration in years
    :return: date offset object
    '''
    return pd.DateOffset(years=years)

def get_min_value(data):
    '''
    This function finds the minimum value in each group of the grouped data.

    :param data: grouped pandas DataFrame
    :return: DataFrame with minimum values
    '''
    return data.min()

def sort_by_values(data, column_name):
    '''
    This function sorts the data by the values in the specified column.

    :param data: pandas DataFrame
    :param column_name: column name to sort by
    :return: sorted DataFrame
    '''
    return data.sort_values(column_name)

def get_first_n_rows(data, n=5):
    '''
    This function returns the first 'n' rows of the data.

    :param data: pandas DataFrame
    :param n: number of rows to return
    :return: DataFrame with first 'n' rows
    '''
    return data.head(n)

def get_max(data):
    '''
    This function finds the maximum value in the given data.

    :param data: pandas DataFrame or Series
    :return: maximum value
    '''
    return data.max()

def filter_by_date(data, date_col, min_date):
    '''
    This function filters the DataFrame based on the minimum date value.

    :param data: pandas DataFrame
    :param date_col: date column name
    :param min_date: minimum date value
    :return: filtered DataFrame
    '''
    return data[data[date_col] >= min_date]

def check_elements_in_list(series, values_list):
    '''
    This function checks whether each element in the series is contained in the given list or not.
    It returns a boolean Series of the same shape as the original Series.

    :param series: pandas Series
    :param values_list: list of values to check for
    :return: boolean Series
    '''
    return series.isin(values_list)

def iterate_rows(dataframe):
    '''
    This function iterates over the rows of the DataFrame as (index, Series) pairs.

    :param dataframe: pandas DataFrame to be iterated
    :return: iterator yielding index and row data as pandas Series
    '''
    return dataframe.iterrows()

def update_dataframe_loc(dataframe, row_label, col_label, value):
    '''
    This function updates a value in the DataFrame using row and column labels.

    :param dataframe: pandas DataFrame to update
    :param row_label: row label for the value to update
    :param col_label: column label for the value to update
    :param value: new value to set at the specified row and column
    '''
    dataframe.loc[row_label, col_label] = value

def convert_to_np_array(df_or_series):
    '''
    This function returns a NumPy array representation of the given DataFrame or Series.

    :param df_or_series: The input DataFrame or Series
    :return: numpy ndarray
    '''
    return df_or_series.values

def convert_np_to_list(numpy_array):
    '''
    This function converts the given NumPy array to a Python list.

    :param numpy_array: The input NumPy array
    :return: list
    '''
    return numpy_array.tolist()

def to_list(data):
    '''
    This function converts the DataFrame values to a Python list.

    :param data: pandas DataFrame or Series
    :return: list of values
    '''
    return data.values.tolist()

def rename_columns(data, columns):
    '''
    This function renames the columns in the given DataFrame.

    :param data: pandas DataFrame
    :param columns: dictionary with old column names as keys and new column names as values
    :return: DataFrame with renamed columns
    '''
    return data.rename(columns=columns)

def fill_missing_values(series, value=None, method=None, axis=None, inplace=False, limit=None):
    '''
    This function is used to fill NaN (Not a Number) values with a specified value or method.

    :param series: pandas Series
    :param value: value to fill NaNs with (default is 0)
    :return: Series with NaNs filled
    '''
    return series.fillna(value=value, method=method, axis=axis, inplace=inplace, limit=limit)

def create_zeros_array(shape):
    '''
    This function creates a new array of the specified shape and type, filled with zeros.

    :param shape: tuple of integers defining the shape of the new array
    :return: numpy array filled with zeros
    '''
    return np.zeros(shape)

def assert_series(series, value, operation):
    '''
    Compares the input Series with the given value using the specified operation.

    :param series: The input Series to compare.
    :param value: The value to compare the Series with.
    :param operation: The comparison operation to use.
    :return: A boolean Series resulting from the comparison.
    '''
    if operation == 'equality':
        return series == value
    elif operation == 'inequality':
        return series != value
    else:
        raise ValueError("Invalid operation. Supported operations are '==' and '!='.")

def logical_and(series1, series2):
    '''
    Performs a logical AND operation between two boolean Series.

    :param series1: The first input boolean Series.
    :param series2: The second input boolean Series.
    :return: A boolean Series resulting from the logical AND operation.
    '''
    return series1 & series2

def logical_or(series1, series2):
    '''
    Performs a logical OR operation between two boolean Series.

    :param series1: The first input boolean Series.
    :param series2: The second input boolean Series.
    :return: A boolean Series resulting from the logical OR operation.
    '''
    return series1 | series2

def fetch_index(dataframe):
    '''
    This function returns the index of the given DataFrame.

    :param dataframe: pandas DataFrame
    :return: pandas Index object
    '''
    return dataframe.index

def is_a_null_df(dataframe):
    '''
    Check if a DataFrame or Series is empty.

    :param dataframe: pandas DataFrame or Series
    :return: boolean, True if empty, False otherwise
    '''
    return dataframe.empty

def visit_by_index(dataframe, index):
    '''
    Access elements in a DataFrame or Series by index.

    :param dataframe: pandas DataFrame or Series
    :param index: integer, index of the element to access
    :return: element at the specified index
    '''
    return dataframe.iloc[index]

def fetch_df_size(dataframe):
    '''
    Return the size of the input DataFrame or DataFrameGroupBy object.

    :param dataframe: The input DataFrame or DataFrameGroupBy object.
    :return: A Series representing the size of the input object.
    '''
    if isinstance(dataframe, pd.core.groupby.generic.DataFrameGroupBy):
        return dataframe.size()
    else:
        return pd.Series(dataframe.size)

def create_multiindex_from_product(iterables, names):
    '''
    create a MultiIndex from the cartesian product of multiple iterables.

    Params:
    iterables: List of iterables to compute the cartesian product.
    names: List of names for the resulting MultiIndex levels.

    Returns:
    A MultiIndex object.
    '''
    return pd.MultiIndex.from_product(iterables, names=names)

def convert_multiindex_to_dataframe(multiindex, index=False):
    '''
    convert a MultiIndex to a DataFrame.

    Params:
    multiindex: A pandas MultiIndex object.
    index: Whether to include the index in the resulting DataFrame (default: False).

    Returns:
    A DataFrame object.
    '''
    return multiindex.to_frame(index=index)

def remove_labels(data, columns, axis=1):
    '''
    drop specified labels from rows or columns.

    Params:
    data: A pandas DataFrame or Series object.
    columns: Labels to drop.
    axis: Axis along which the labels will be dropped (default: 1).

    Returns:
    A DataFrame or Series with the specified labels dropped.
    '''
    return data.drop(columns=columns, axis=axis)

def draw_countplot(x, data):
    '''
    Shows the counts of observations in each categorical bin using bars.
    :param x: string, column name in the data
    :param data: DataFrame, input data
    '''
    sns.countplot(x=x, data=data)

def plot_kde(data, label, shade):
    '''
    Plots a kernel density estimate of the given data.
    :param data: array-like, input data
    :param label: string, label for the plot
    :param shade: bool, whether to shade the area under the curve
    '''
    sns.kdeplot(data, label=label, fill=shade)

def select_data_types(df, dtype_list):
    '''
    This function takes a pandas DataFrame and returns a DataFrame containing only the categorical columns.

    :param df: pandas DataFrame
    :return: pandas DataFrame with categorical columns
    '''
    return df.select_dtypes(include=dtype_list)

def get_columns(df):
    '''
    This function takes a pandas DataFrame and returns its columns.

    :param df: pandas DataFrame
    :return: columns (pandas Index object)
    '''
    return df.columns

def create_standard_scaler():
    '''
    This function creates a StandardScaler instance.

    :return: StandardScaler instance
    '''
    return StandardScaler()

def fit_transform_standard_scaler(scaler, data):
    '''
    This function fits the StandardScaler to the data and transforms the data using the fitted StandardScaler.

    :param scaler: StandardScaler instance
    :param data: Data to be standardized
    :return: Standardized data
    '''
    return scaler.fit_transform(data)

def create_label_encoder():
    '''
    This function creates a LabelEncoder instance.

    :return: LabelEncoder instance
    '''
    return LabelEncoder()

def fit_transform_label_encoder(le, data):
    '''
    This function fits the LabelEncoder to the data and transforms the data using the fitted LabelEncoder.

    :param le: LabelEncoder instance
    :param data: Data to be label encoded
    :return: Label encoded data
    '''
    return le.fit_transform(data)

def create_kmeans(n_clusters=8, init='k-means++', n_init='warn', max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='lloyd'):
    '''
    This function creates a KMeans instance with the specified number of clusters.

    :param cluster_num: Number of clusters
    :param random_state: Random state for the KMeans algorithm
    :return: KMeans instance
    '''
    return KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol, verbose=verbose, random_state=random_state, copy_x=copy_x, algorithm=algorithm)

def fit_predict_kmeans(kmeans, data):
    '''
    This function fits the KMeans instance to the data and predicts the cluster index for each sample.

    :param kmeans: KMeans instance
    :param data: Data to be clustered
    :return: Cluster index for each sample
    '''
    return kmeans.fit_predict(data)

def get_silhouette_score(X, labels, metric='euclidean', sample_size=None, random_state=None, **kwds):
    '''
    This function Compute the mean Silhouette Coefficient of all samples.

    :param X: An array of pairwise distances between samples, or a feature array.
    :param labels: Predicted labels for each sample.
    :param metric: The metric to use when calculating distance between instances in a feature array. If metric is a string, it must be one of the options allowed by pairwise_distances. If X is the distance array itself, use
    :param sample_size: The size of the sample to use when computing the Silhouette Coefficient on a random subset of the data.
    :param random_state: Determines random number generation for selecting a subset of samples.
    :return: Mean Silhouette Coefficient for all samples.
    '''
    return silhouette_score(X, labels, metric=metric, sample_size=sample_size, random_state=random_state, **kwds)

def plot(*args, scalex=True, scaley=True, data=None, **kwargs):
    '''
    Plot y versus x as lines and/or markers.

    :param x, y: The horizontal / vertical coordinates of the data points. x values are optional and default to range(len(y)).
    :param **kwargs: Line2D properties, optional
    :param scalex, scaley: These parameters determine if the view limits are adapted to the data limits. The values are passed on to autoscale_view.
    :return: list of Line2D A list of lines representing the plotted data.
    '''
    return plt.plot(*args, scalex=scalex, scaley=scaley, data=data, **kwargs)

def grid(visible=None, which='major', axis='both', **kwargs):
    '''
    Configure the grid lines.

    :param visible: Whether to show the grid lines. If any kwargs are supplied, it is assumed you want the grid on and visible will be set to True.
    :param **kwargs: Line2D properties: Define the line properties of the grid
    :param which: The grid lines to apply the changes on.
    :param axis: The axis to apply the changes on
    :return: list of Line2D A list of lines representing the plotted data.
    '''
    return plt.grid(visible=visible, which=which, axis=axis, **kwargs)

def col_assign_val(df, col, val):
    '''
    Assign a value to a specified column in a DataFrame.

    :param df: pandas.DataFrame The input DataFrame to be modified
    :param col: str The column name in the DataFrame to assign the value
    :param val: The value to be assigned to the specified column
    :return: pandas.Series The modified column with the assigned value
    '''
    df[col] = val

def extract_unique_values(series):
    '''
    This function extracts unique values from the given pandas Series.

    :param series: pandas Series
    :return: series of unique values
    '''
    return series.unique()

def series_get_quantile(series, q=0.5, interpolation='linear'):

  '''
  Calculate the quantile value of a pandas series using linear interpolation.

  :param series: The pandas series to calculate the quantile value on.
  :param q: The quantile value to calculate. Default is 0.5.
  :param interpolation: The interpolation method to use. Default is 'linear'.
  :return: The quantile value of the series at the given quantile and interpolation.
  '''
  return series.quantile(q=q, interpolation=interpolation)

def series_value_counts(series, normalize=False, sort=True, ascending=False, bins=None, dropna=True):
    '''
    Return a pandas series containing the counts of unique values in the input series.

    :param series: The input pandas series to count unique values on.
    :param normalize: Whether to return relative frequencies of the unique values. Default is False.
    :param sort: Whether to sort the resulting counts by frequency. Default is True.
    :param ascending: Whether to sort the resulting counts in ascending order. Default is False.
    :param bins: If passed, will group the counts into the specified number of bins. Default is None.
    :param dropna: Whether to exclude missing values from the counts. Default is True.
    :return: A pandas series containing the counts of unique values in the input series.
    '''
    return series.value_counts(normalize=normalize, sort=sort, ascending=ascending, bins=bins, dropna=dropna)

def series_to_dict(series):
    '''
    Convert a pandas series into a dictionary.

    :param series: The pandas series to convert into a dictionary.
    :return: A dictionary with the index values of the series as the keys and the corresponding values of the series as the values.
    '''
    return series.to_dict()

def choose_data_types(df, dtype_list):
    '''
    This function takes a pandas DataFrame and returns a DataFrame containing only the categorical columns.

    :param df: pandas DataFrame
    :return: pandas DataFrame with categorical columns
    '''
    return df.select_dtypes(include=dtype_list)

def get_columns(df):
    '''
    This function takes a pandas DataFrame and returns its columns.

    :param df: pandas DataFrame
    :return: columns (pandas Index object)
    '''
    return df.columns

def standard_scaler_instance():
    '''
    This function creates a StandardScaler instance.

    :return: StandardScaler instance
    '''
    return StandardScaler()

def modify_data_standard_scaler(scaler, data):
    '''
    This function fits the StandardScaler to the data and transforms the data using the fitted StandardScaler.

    :param scaler: StandardScaler instance
    :param data: Data to be standardized
    :return: Standardized data
    '''
    return scaler.fit_transform(data)

def encoder_instance():
    '''
    This function creates a LabelEncoder instance.

    :return: LabelEncoder instance
    '''
    return LabelEncoder()

def encode_column(le, data):
    '''
    This function fits the LabelEncoder to the data and transforms the data using the fitted LabelEncoder.

    :param le: LabelEncoder instance
    :param data: Data to be label encoded
    :return: Label encoded data
    '''
    return le.fit_transform(data)

def kmeans_instance(n_clusters=8, init='k-means++', n_init='warn', max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='lloyd'):
    '''
    This function creates a KMeans instance with the specified number of clusters.

    :param cluster_num: Number of clusters
    :param random_state: Random state for the KMeans algorithm
    :return: KMeans instance
    '''
    return KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol, verbose=verbose, random_state=random_state, copy_x=copy_x, algorithm=algorithm)

def predict_data_kmeans(kmeans, data):
    '''
    This function fits the KMeans instance to the data and predicts the cluster index for each sample.

    :param kmeans: KMeans instance
    :param data: Data to be clustered
    :return: Cluster index for each sample
    '''
    return kmeans.fit_predict(data)

def get_coefficient_silhouette(X, labels, metric='euclidean', sample_size=None, random_state=None, **kwds):
    '''
    This function Compute the mean Silhouette Coefficient of all samples.

    :param X: An array of pairwise distances between samples, or a feature array.
    :param labels: Predicted labels for each sample.
    :param metric: The metric to use when calculating distance between instances in a feature array. If metric is a string, it must be one of the options allowed by pairwise_distances. If X is the distance array itself, use
    :param sample_size: The size of the sample to use when computing the Silhouette Coefficient on a random subset of the data.
    :param random_state: Determines random number generation for selecting a subset of samples.
    :return: Mean Silhouette Coefficient for all samples.
    '''
    return silhouette_score(X, labels, metric=metric, sample_size=sample_size, random_state=random_state, **kwds)

def get_figure(*args, scalex=True, scaley=True, data=None, **kwargs):
    '''
    Plot y versus x as lines and/or markers.

    :param x, y: The horizontal / vertical coordinates of the data points. x values are optional and default to range(len(y)).
    :param **kwargs: Line2D properties, optional
    :param scalex, scaley: These parameters determine if the view limits are adapted to the data limits. The values are passed on to autoscale_view.
    :return: list of Line2D A list of lines representing the plotted data.
    '''
    plt.plot(*args, scalex=scalex, scaley=scaley, data=data, **kwargs)

def configure_gridlines(visible=None, which='major', axis='both', **kwargs):
    '''
    Configure the grid lines.

    :param visible: Whether to show the grid lines. If any kwargs are supplied, it is assumed you want the grid on and visible will be set to True.
    :param **kwargs: Line2D properties: Define the line properties of the grid
    :param which: The grid lines to apply the changes on.
    :param axis: The axis to apply the changes on
    :return: list of Line2D A list of lines representing the plotted data.
    '''
    return plt.grid(visible=visible, which=which, axis=axis, **kwargs)

def col_assign_val(df, col, val):
    '''
    Assign a value to a specified column in a DataFrame.

    :param df: pandas.DataFrame The input DataFrame to be modified
    :param col: str The column name in the DataFrame to assign the value
    :param val: The value to be assigned to the specified column
    :return: pandas.Series The modified column with the assigned value
    '''
    df[col] = val

def extract_unique_values(series):
    '''
    This function extracts unique values from the given pandas Series.

    :param series: pandas Series
    :return: series of unique values
    '''
    return series.unique()

def calc_quantile_val(series, q=0.5, interpolation='linear'):

    '''
    Calculate the quantile value of a pandas series using linear interpolation.

    :param series: The pandas series to calculate the quantile value on.
    :param q: The quantile value to calculate. Default is 0.5.
    :param interpolation: The interpolation method to use. Default is 'linear'.
    :return: The quantile value of the series at the given quantile and interpolation.
    '''
    return series.quantile(q=q, interpolation=interpolation)

def transform_to_dictionary(series):
    '''
    Convert a pandas series into a dictionary.

    :param series: The pandas series to convert into a dictionary.
    :return: A dictionary with the index values of the series as the keys and the corresponding values of the series as the values.
    '''
    return series.to_dict()

def calculate_quantile(series, percentile):
    '''
    This function calculates the quantile for a pandas Series.

    :param series: pandas Series
    :param percentile: percentile value (0.75 for 75th percentile)
    :return: quantile value
    '''
    return series.quantile(percentile)

def filter_by_condition(dataframe, condition):
    '''
    This function filters a pandas DataFrame based on a given condition.

    :param dataframe: pandas DataFrame
    :param condition: boolean condition to filter the DataFrame
    :return: filtered pandas DataFrame
    '''
    return dataframe[condition]

def create_condition(series, value):
    '''
    This function creates a boolean condition for filtering a pandas DataFrame.

    :param series: pandas Series
    :param value: value to compare with the Series
    :return: boolean condition
    '''
    return series > value

def convert_to_tuples(dataframe):
    '''
    This function converts a pandas DataFrame to a list of tuples.

    :param dataframe: pandas DataFrame
    :return: list of tuples
    '''
    return dataframe.itertuples(index=True, name=None)

def filter_by_value(dataframe, column, value):
    '''
    This function filters a pandas DataFrame based on a specific value in a column.

    :param dataframe: pandas DataFrame
    :param column: column name in the DataFrame
    :param value: value to filter by in the column
    :return: filtered pandas DataFrame
    '''
    return dataframe[dataframe[column] == value]

def convert_to_list(index):
    '''
    This function converts a pandas Index to a list.

    :param index: pandas Index
    :return: list
    '''
    return list(index)

def calculate_median(dataframe, column):
    '''
    This function calculates the median for a specific column in a pandas DataFrame.

    :param dataframe: pandas DataFrame
    :param column: column name in the DataFrame
    :return: median value
    '''
    return dataframe[column].median()

def count_rows(dataframe):
    '''
    This function counts the number of rows in a pandas DataFrame.

    :param dataframe: pandas DataFrame
    :return: number of rows
    '''
    return dataframe.shape[0]

def locate_mode(series):
    '''
    This function calculates the mode of a series.

    :param series: input series
    :return: mode of the series
    '''
    return series.mode()

def get_n_row(df, n):
    '''
    This function returns the first row of a DataFrame.

    :param df: input DataFrame
    :return: first row of the DataFrame
    '''
    return df.iloc[n]

def set_layout(pad=1.08, h_pad=None, w_pad=None, rect=None):
    '''
    Adjust the padding between and around subplots.

    :param pad: Padding between the figure edge and the edges of subplots, as a fraction of the font size.
    :param h_pad, w_pad: Padding (height/width) between edges of adjacent subplots, as a fraction of the font size.
    :param rect: A rectangle in normalized figure coordinates into which the whole subplots area (including labels) will fit.
    :return: None
    '''
    plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad, rect=rect)

def get_fig_from_df(data, *args, **kwargs):
    '''
    This function plots a bar chart.

    :param data: The object for which the method is called.
    :return: None
    '''
    data.plot(*args, **kwargs)

def pivot_a_level(data, level=-1, fill_value=None):
    '''
    Pivot a level of the (necessarily hierarchical) index labels.

    :param level: Level(s) of index to unstack, can pass level name.
    :param fill_value: Replace NaN with this value if the unstack produces missing values.
    :param sort: Sort the level(s) in the resulting MultiIndex columns.
    :return: DataFrame after unstacking
    '''
    return data.unstack(level=level, fill_value=fill_value)

def cast_to_a_dtype(data, dtype, copy=None, errors='raise'):
    '''
    Cast a pandas object to a specified dtype dtype.

    :param dtype: Use a str, numpy.dtype, pandas.ExtensionDtype or Python type to cast entire pandas object to the same type. Alternatively, use a mapping, where col is a column label and dtype is a numpy.dtype or Python type to cast one or more of the DataFrame’s columns to column-specific types.
    :param copy: Return a copy when copy=True (be very careful setting copy=False as changes to values then may propagate to other pandas objects).
    :param errors: Control raising of exceptions on invalid data for provided dtype.

    :return: None
    '''
    return data.astype(dtype, copy=copy, errors=errors)

def categoricalIndex(data):
    '''
    Accessor object for categorical properties of the Series values.

    :return: None
    '''
    return data.cat

def  categorical_codes(cat):
    '''
    The category codes of this categorical index.

    :return: None
    '''
    return cat.codes

def scatter_fig_instance(data, x, y, ax):
    '''
    This function creates a scatterplot using seaborn.

    :param data: pandas DataFrame
    :param x: column name for the x-axis
    :param y: column name for the y-axis
    :param ax: axis object to plot the scatterplot on
    '''
    sns.scatterplot(data=data, x=x, y=y, ax=ax)

def divide_dataset(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None):
    '''
    Split arrays or matrices into random train and test subsets.

    :param *arrays: Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.
    :param test_size: If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. If None, the value is set to the complement of the train size. If train_size is also None, it will be set to 0.25.
    :param train_size: If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split. If int, represents the absolute number of train samples. If None, the value is automatically set to the complement of the test size.
    :param random_state: Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.
    :param shuffle: Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.
    :param stratify: If not None, data is split in a stratified fashion, using this as the class labels.

    :return: List containing train-test split of inputs.
    '''
    return train_test_split(*arrays, test_size=test_size, train_size=train_size, random_state=random_state, shuffle=shuffle, stratify=stratify)

def create_LR_instance(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None):
    '''
    This function initializes a Logistic Regression.

    :return: initialized Logistic Regression
    '''
    return LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs, l1_ratio=l1_ratio)

def classifier_training(classifier, X_train, y_train):
    '''
    This function trains a classifier on the given training data.

    :param classifier: classifier object
    :param X_train: training data features
    :param y_train: training data labels
    :return: trained classifier
    '''
    classifier.fit(X_train, y_train)
    return classifier

def classifier_predictions(classifier, X_test):
    '''
    This function makes predictions using a trained classifier.

    :param classifier: trained classifier object
    :param X_test: testing data features
    :return: predicted labels
    '''
    return classifier.predict(X_test)

def calculate_conf_mat(y_true, y_pred):
    '''
    This function calculates the confusion matrix.

    :param y_true: true labels
    :param y_pred: predicted labels
    :return: confusion matrix
    '''
    return confusion_matrix(y_true, y_pred)

def calc_acc(y_true, y_pred):
    '''
    This function calculates the accuracy score.

    :param y_true: true labels
    :param y_pred: predicted labels
    :return: accuracy score
    '''
    return accuracy_score(y_true, y_pred)

def create_barplot(data=None, x=None, y=None, hue=None, order=None, hue_order=None, estimator='mean', errorbar=('ci', 95), n_boot=1000, units=None, seed=None, orient=None, color=None, palette=None, saturation=0.75, fill=True, hue_norm=None, width=0.8, dodge='auto', gap=0, log_scale=None, native_scale=False, formatter=None, legend='auto', capsize=0, err_kws=None, ax=None, **kwargs):
    '''
    This function creates a bar plot using seaborn.

    :param data: DataFrame, array, or list of arrays, optional
    :param x, y, hue: names of variables in data or vector data, optional
    :param order, hue_order: lists of strings, optional
    :param estimator: statistical function to estimate within each categorical bin, optional
    :param errorbar: error bar to draw on bars, optional
    :param n_boot: number of bootstraps to compute the confidence interval, optional
    :param units: identifier of sampling units, optional
    :param seed: seed or random number generator, optional
    :param orient: orientation of the plot (vertical or horizontal), optional
    :param color: matplotlib color, optional
    :param palette: palette name, list, or dict, optional
    :param saturation: proportion of the original saturation to draw colors, optional
    :param fill: if True, fill the bars with the hue variable, optional
    :param hue_norm: tuple or matplotlib.colors.Normalize, optional
    :param width: width of a full element when not using hue nesting, or width of all the elements for one level of the major grouping variable, optional
    :param dodge: when hue nesting is used, whether elements should be shifted along the categorical axis, optional
    :param gap: size of the gap between bars, optional
    :param log_scale: boolean or number, or pair of boolean or number, optional
    :param native_scale: boolean, optional
    :param formatter: callable or string format, optional
    :param legend: boolean or 'auto', optional
    :param capsize: width of the caps on error bars, optional
    :param err_kws: keyword arguments for matplotlib.axes.Axes.errorbar(), optional
    :param ax: matplotlib Axes, optional
    :param kwargs: key, value pairings
    :return: matplotlib Axes
    '''
    return sns.barplot(data=data, x=x, y=y, hue=hue, order=order, hue_order=hue_order, estimator=estimator, errorbar=errorbar, n_boot=n_boot, units=units, seed=seed, orient=orient, color=color, palette=palette, saturation=saturation, fill=fill, hue_norm=hue_norm, width=width, dodge=dodge, gap=gap, log_scale=log_scale, native_scale=native_scale, formatter=formatter, legend=legend, capsize=capsize, err_kws=err_kws, ax=ax, **kwargs)

def create_histogram(data=None, x=None, y=None, hue=None, weights=None, stat='count', bins='auto', binwidth=None, binrange=None, discrete=None, cumulative=False, common_bins=True, common_norm=True, multiple='layer', element='bars', fill=True, shrink=1, kde=False, kde_kws=None, line_kws=None, thresh=0, pthresh=None, pmax=None, cbar=False, cbar_ax=None, cbar_kws=None, palette=None, hue_order=None, hue_norm=None, color=None, log_scale=None, legend=True, ax=None, **kwargs):
    '''
    This function creates a histogram plot using seaborn.

    :param data: DataFrame, array, or list of arrays, optional
    :param x, y, hue: names of variables in data or vector data, optional
    :param weights: array of weights, optional
    :param stat: aggregate statistic to compute in each bin, optional
    :param bins: specification of hist bins, or None to use Freedman-Diaconis rule, optional
    :param binwidth: width of each bin, optional
    :param binrange: lowest and highest value of bins, optional
    :param discrete: if True, draw a bar at each unique value in x, instead of using a histogram, optional
    :param cumulative: if True, the histogram accumulates values, optional
    :param common_bins: if True, the same bins will be used across all histograms, optional
    :param common_norm: if True, all densities on the same axes are scaled by the same factor, optional
    :param multiple: approach to resolving multiple elements when semantic mapping creates subsets, optional
    :param element: approach to drawing the histogram, optional
    :param fill: if True, fill in the bars in the histogram, optional
    :param shrink: scale factor for shrinking the density curve in rugplot and kde, optional
    :param kde: if True, compute a kernel density estimate to smooth the distribution and show on the plot, optional
    :param kde_kws: keyword arguments for kdeplot(), optional
    :param line_kws: keyword arguments for plot(), optional
    :param thresh: threshold for small elements, optional
    :param pthresh: threshold for small elements as a proportion, optional
    :param pmax: maximum proportion of the figure area that a small element can occupy, optional
    :param cbar: if True, add a colorbar to the figure, optional
    :param cbar_ax: axes in which to draw the colorbar, optional
    :param cbar_kws: keyword arguments for colorbar(), optional
    :param palette: palette name, list, or dict, optional
    :param hue_order: order for the levels of the hue variable in the palette, optional
    :param hue_norm: normalization in data units for colormap applied to the hue variable, optional
    :param color: color for all elements, or seed for a gradient palette, optional
    :param log_scale: boolean or number, or pair of boolean or number, optional
    :param legend: if True, add a legend or legend elements, optional
    :param ax: matplotlib Axes, optional
    :param kwargs: key, value pairings
    :return: matplotlib Axes
    '''
    return sns.histplot(data=data, x=x, y=y, hue=hue, weights=weights, stat=stat, bins=bins, binwidth=binwidth, binrange=binrange, discrete=discrete, cumulative=cumulative, common_bins=common_bins, common_norm=common_norm, multiple=multiple, element=element, fill=fill, shrink=shrink, kde=kde, kde_kws=kde_kws, line_kws=line_kws, thresh=thresh, pthresh=pthresh, pmax=pmax, cbar=cbar, cbar_ax=cbar_ax, cbar_kws=cbar_kws, palette=palette, hue_order=hue_order, hue_norm=hue_norm, color=color, log_scale=log_scale, legend=legend, ax=ax, **kwargs)

def create_countplot(data=None, x=None, y=None, hue=None, order=None, hue_order=None, orient=None, color=None, palette=None, saturation=0.75, fill=True, width=0.8, dodge='auto', ax=None, **kwargs):
    '''
    This function creates a count plot using seaborn.

    :param data: DataFrame, array, or list of arrays, optional
    :param x, y, hue: names of variables in data or vector data, optional
    :param order, hue_order: lists of strings, optional
    :param orient: orientation of the plot (vertical or horizontal), optional
    :param color: matplotlib color, optional
    :param palette: palette name, list, or dict, optional
    :param saturation: proportion of the original saturation to draw colors, optional
    :param fill: if True, fill the bars with the hue variable, optional
    :param hue_norm: tuple or matplotlib.colors.Normalize, optional
    :param stat: aggregate statistic to compute in each bin, optional
    :param width: width of a full element when not using hue nesting, or width of all the elements for one level of the major grouping variable, optional
    :param dodge: when hue nesting is used, whether elements should be shifted along the categorical axis, optional
    :param gap: size of the gap between bars, optional
    :param log_scale: boolean or number, or pair of boolean or number, optional
    :param native_scale: boolean, optional
    :param formatter: callable or string format, optional
    :param legend: boolean or 'auto', optional
    :param ax: matplotlib Axes, optional
    :param kwargs: key, value pairings
    :return: matplotlib Axes
    '''
    return sns.countplot(data=data, x=x, y=y, hue=hue, order=order, hue_order=hue_order, orient=orient, color=color, palette=palette, saturation=saturation, fill=fill, width=width, dodge=dodge, ax=ax, **kwargs)

def set_current_ticks(ticks=None, labels=None, minor=False, **kwargs):
    '''
    Get or set the current tick locations and labels of the x-axis.

    :param ticks: An array-like object containing the positions of the ticks. If None, the current ticks are used.
    :param labels: A list of string labels to use as the tick labels. If None, the current labels are used.
    :param minor: If True, the ticks will be minor ticks. If False, the ticks will be major ticks.
    :param **kwargs: Additional keyword arguments to be passed to the plt.xticks() function.
    :return: A tuple containing the x-location of the new ticks and a list of <class 'matplotlib.text.Text'> instances.
    '''

    return plt.xticks(ticks=ticks, labels=labels, minor=minor, **kwargs)

def create_heatmap(data, vmin=None, vmax=None, cmap=None, center=None, robust=False, annot=None, fmt='.2g', annot_kws=None, linewidths=0, linecolor='white', cbar=True, cbar_kws=None, cbar_ax=None, square=False, xticklabels='auto', yticklabels='auto', mask=None, ax=None, **kwargs):
    '''
    This function creates a heatmap using seaborn's heatmap function.

    :param data: 2D dataset that can be coerced into an ndarray. If a Pandas DataFrame is provided, the index/column information will be used to label the columns and rows.
    :param vmin, vmax: Values to anchor the colormap, otherwise they are inferred from the data and other keyword arguments.
    :param cmap: The mapping from data values to color space.
    :param center: The value at which to center the colormap when plotting divergant data.
    :param robust: If True and vmin or vmax are absent, the colormap range is computed with robust quantiles instead of the extreme values.
    :param annot: If True, write the data value in each cell.
    :param fmt: String formatting code to use when adding annotations.
    :param annot_kws: Keyword arguments for ax.text when annot is True.
    :param linewidths: Width of the lines that will divide each cell.
    :param linecolor: Color of the lines that will divide each cell.
    :param cbar: Whether to draw a colorbar.
    :param cbar_kws: Keyword arguments for fig.colorbar.
    :param cbar_ax: None or Axes, optional. Axes in which to draw the colorbar, otherwise take space from the main Axes.
    :param square: If True, set the Axes aspect to “equal” so each cell will be square-shaped.
    :param xticklabels, yticklabels: If True, plot the column names of the dataframe. If False, don’t plot the column names. If list-like, plot these alternate labels as the xticklabels. If an integer, use the column names but plot only every n label. If “auto”, try to densely plot non-overlapping labels.
    :param mask: If passed, data will not be shown in cells where mask is True. Cells with missing values are automatically masked.
    :param ax: If provided, plot the heatmap on this axis.
    :param kwargs: Other keyword arguments are passed down to ax.pcolormesh.

    :return: Axes object with the heatmap.
    '''
    return sns.heatmap(data, vmin=vmin, vmax=vmax, cmap=cmap, center=center, robust=robust, annot=annot, fmt=fmt, annot_kws=annot_kws, linewidths=linewidths, linecolor=linecolor, cbar=cbar, cbar_kws=cbar_kws, cbar_ax=cbar_ax, square=square, xticklabels=xticklabels, yticklabels=yticklabels, mask=mask, ax=ax, **kwargs)

def search_where(data, condition, column):  
    '''  
    Filters the data based on the given condition and returns distinct values from the specified column.  
  
    :param data: The DataFrame to search in.  
    :param condition: The boolean Series representing the condition to filter by.  
    :param column: The column from which to return values.  
    :return: A list of values from the specified column after applying the condition.  
    ''' 
  
    # Return the values from the specified column  
    return data.where(condition, column)

def update_dict(original_dict, updates):  
    '''  
    Updates the original dictionary with the key-value pairs from the updates dictionary.  
  
    :param original_dict: The original dictionary to be updated.  
    :param updates: The dictionary containing updates to be applied.  
    '''  
    original_dict.update(updates)

def use_function(df, func, axis=0, raw=False, result_type=None, args=()):
    '''
    This function apply function to pandas Series.

    :param series: pandas Series
    :return: series that with function applied
    '''
    return df.apply(func, axis=axis, raw=raw, result_type=result_type, args=args)

def add_legend():
    '''
    This function add legned to plot.
    '''
    plt.legend()
------------------------------------- decision_company END: -------------------------------------

The user requests the most twenty relevant and usable functions contained within decision_company library to answer my latest query. Present the list of function names in a specific format. If no functions are found, indicate this appropriately. The expected output format is either:

<FUNCTIONS>function1,function2,...,function20</FUNCTIONS>

or, if no functions are identified:

<FUNCTIONS>None</FUNCTIONS>

NOTE: Rank the extracted functions with their importance in descending order. For example, function1 should be more important than function2, function3 ...

"""
pre_prompt_func_names_AI = """
[YOU (AI assistant)]:
# Output:
<FUNCTIONS>read_csv_file,
"""

ATP_normal_pseudo_learn_gpt4 = """There is the data: atp_tennis containing the following columns: ['Tournament', 'Date', 'Series', 'Court', 'Surface', 'Round', 'Best of', 'Player_1', 'Player_2', 'Winner', 'Rank_1', 'Rank_2', 'Pts_1', 'Pts_2', 'Odd_1', 'Odd_2', 'score'].  

Considering contents from the dataset and requirements from user. Please note DO NOT CHANGE FILE AND VARIABLE NAMES THAT I HAVE SET!
Interactions begin:
--- Interaction History: ---
{dialogue_hist}

--- New Query: ---
{new_query}

################### Example Starts: ###################
{pseudo_example}
################### Example Ends ###################

As a data science professor, your mission is to guide your students in creating Python code to tackle data science problems, using the last successful interaction between users and AI as a reference. You need to firstly analyze the Python code used in that specific interaction. Your task is to develop a pseudocode that will guide students to write correct python code to answer the "New Query". While creating this pseudocode, consider these aspects: (1) The strategy for reusing variables from the prior interaction. Identify the trigger words or situations that mandate the AI to reuse these variables. (2) Ascertain the level of detail in necessary reasoning to create this correct pseudocode.    
    
Please give me a STANDARD pseudocode format with proper hierarchy, neither bullet points nor python format!    

# Your pseudocode:
"""

ATP_normal_pseudo_infer_gpt4 = """There is the data: atp_tennis containing the following columns: ['Tournament', 'Date', 'Series', 'Court', 'Surface', 'Round', 'Best of', 'Player_1', 'Player_2', 'Winner', 'Rank_1', 'Rank_2', 'Pts_1', 'Pts_2', 'Odd_1', 'Odd_2', 'score'].  

Considering contents from the dataset and requirements from user. Please note DO NOT CHANGE FILE AND VARIABLE NAMES THAT I HAVE SET!
Interactions begin:
--- Interaction History: ---
{dialogue_hist}

--- New Query: ---
{new_query}

As a data science professor, your mission is to guide your students in creating Python code to tackle data science problems, using the successful example above. This involves analyzing the pseudocode and Python code used in that example interaction. Your task is to firstly develop a pseudocode for a programme that will guide students to write correct python code to answer the "New Query". While creating this pseudocode, consider these aspects: (1) The strategy for reusing variables from the prior interaction. Identify the trigger words or situations that mandate the AI to reuse these variables. (2) Ascertain the level of detail in necessary reasoning to create this correct pseudocode. And finally provide the Python code to fill the <code> and </code> in the "New Query". Note: there might be more than one places need to fill.

[PROFESSOR]:
# Step 1: pseudocode to fill <code> and </code>:
<code>
"""

header_ATP = """There is the data: atp_tennis containing the following columns: ['Tournament', 'Date', 'Series', 'Court', 'Surface', 'Round', 'Best of', 'Player_1', 'Player_2', 'Winner', 'Rank_1', 'Rank_2', 'Pts_1', 'Pts_2', 'Odd_1', 'Odd_2', 'score'].  

Considering contents from the dataset and requirements from user. Please note DO NOT CHANGE FILE AND VARIABLE NAMES THAT I HAVE SET!
Interactions begin:
"""


prompt_cot_normal = """
Analyze the example below first and solve the code generation task after the example based on what you learned from example. Let's think step by step!

------------------------------ Example Start: ------------------------------
[USER (data scientist)]: Please generate the code to determine ...  

[YOU (AI assistant)]: Sure, let's think step by step to generate the code!
################### Your Answer Starts Here: ################### 
Step 1:
Thought 1: First import all packages needed and load the dataset.
Code 1:
'''
import pandas as pd  
import ...
... ...

YOUR-CODE-HERE
'''

Step 2:
Thought 2: Calculate ...
Code 2:
'''
YOUR-CODE-HERE
'''

... ...

Step 4:
Thought 4: Print the ...
Code 4:
'''
YOUR-CODE-HERE
'''
################### Your Answer Ends Here: ################### 
------------------------------ Example End ------------------------------ 

{dialogue_hist}

NOTE: Please follow my example between "Example Start" and "Example End" above to solve the code generation task.

[YOU (AI assistant)]: Sure, let's think step by step to generate the code!
################### Your Answer Starts Here: ################### 
Step 1:
Thought 1: First import all packages needed and load the dataset.
Code 1:
'''
"""

credit_df = """
checking_status | duration | credit_history | purpose | credit_amount | savings_status | employment | installment_commitment | personal_status | other_parties | residence_since | property_magnitude | age | other_payment_plans | housing | existing_credits | job | num_dependents | own_telephone | foreign_worker | class
<0 | 6.0 | critical/other existing credit | radio/tv | 1169.0 | no known savings | >=7 | 4.0 | male single | none | 4.0 | real estate | 67.0 | none | own | 2.0 | skilled | 1.0 | yes | yes | good
0<=X<200 | 48.0 | existing paid | radio/tv | 5951.0 | <100 | 1<=X<4 | 2.0 | female div/dep/mar | none | 2.0 | real estate | 22.0 | none | own | 1.0 | skilled | 1.0 | none | yes | bad
no checking | 12.0 | critical/other existing credit | education | 2096.0 | <100 | 4<=X<7 | 2.0 | male single | none | 3.0 | real estate | 49.0 | none | own | 1.0 | unskilled resident | 2.0 | none | yes | good
... ...
"""

atp_df = """
Tournament | Date | Series | Court | Surface | Round | Best of | Player_1 | Player_2 | Winner | Rank_1 | Rank_2 | Pts_1 | Pts_2 | Odd_1 | Odd_2 | score  
Brisbane International | 2012-12-31 | ATP250 | Outdoor | Hard | 1st Round | 3 | Mayer F. | Giraldo S. | Mayer F. | 28 | 57 | 1215 | 778 | 1.36 | 3.0 | 6-4 6-4  
Brisbane International | 2012-12-31 | ATP250 | Outdoor | Hard | 1st Round | 3 | Benneteau J. | Nieminen J. | Nieminen J. | 35 | 41 | 1075 | 927 | 2.2 | 1.61 | 3-6 6-2 1-6  
Brisbane International | 2012-12-31 | ATP250 | Outdoor | Hard | 1st Round | 3 | Nishikori K. | Matosevic M. | Nishikori K. | 19 | 49 | 1830 | 845 | 1.25 | 3.75 | 7-5 6-2  
... ...
"""

fastfood_df = """
restaurant | item | calories | cal_fat | total_fat | sat_fat | trans_fat | cholesterol | sodium | total_carb | fiber | sugar | protein | vit_a | vit_c | calcium | salad
Mcdonalds | Artisan Grilled Chicken Sandwich | 380 | 60 | 7 | 2 | 0 | 95 | 1110 | 44 | 3 | 11 | 37 | 4 | 20 | 20 | Other
Mcdonalds | Single Bacon Smokehouse Burger | 840 | 410 | 45 | 17 | 1.5 | 130 | 1580 | 62 | 2 | 18 | 46 | 6 | 20 | 20 | Other
Mcdonalds | Double Bacon Smokehouse Burger | 1130 | 600 | 67 | 27 | 3 | 220 | 1920 | 63 | 3 | 18 | 70 | 10 | 20 | 50 | Other 
... ...
"""

melb_df = """
Suburb | Address | Rooms | Type | Price | Method | SellerG | Date | Distance | Postcode | Bedroom2 | Bathroom | Car | Landsize | BuildingArea | YearBuilt | CouncilArea | Lattitude | Longtitude | Regionname | Propertycount
Abbotsford | 85 Turner St | 2 | h | 1480000.0 | S | Biggin | 3/12/2016 | 2.5 | 3067.0 | 2.0 | 1.0 | 1.0 | 202.0 |  |  | Yarra | -37.7996 | 144.9984 | Northern Metropolitan | 4019.0
Abbotsford | 25 Bloomburg St | 2 | h | 1035000.0 | S | Biggin | 4/02/2016 | 2.5 | 3067.0 | 2.0 | 1.0 | 0.0 | 156.0 | 79.0 | 1900.0 | Yarra | -37.8079 | 144.9934 | Northern Metropolitan | 4019.0
Abbotsford | 5 Charles St | 3 | h | 1465000.0 | SP | Biggin | 4/03/2017 | 2.5 | 3067.0 | 3.0 | 2.0 | 0.0 | 134.0 | 150.0 | 1900.0 | Yarra | -37.8093 | 144.9944 | Northern Metropolitan | 4019.0
... ...
"""

laptop_df = """
Manufacturer | Model Name | Category | Screen Size | Screen | CPU | RAM |  Storage | GPU | Operating System | Operating System Version | Weight | Price
HP | 15-bs053od (i7-7500U/6GB/1TB/W10) | Notebook | 15.6" | 1366x768 | Intel Core i7 7500U 2.7GHz | 6GB | 1TB HDD | Intel HD Graphics 620 | Windows | 10 | 2.04kg | 5148468
Asus | Rog GL753VE-DS74 | Gaming | 17.3" | Full HD 1920x1080 | Intel Core i7 7700HQ 2.8GHz | 16GB | 256GB SSD +  1TB HDD | Nvidia GeForce GTX 1050 Ti | Windows | 10 | 2.99kg | 15552108
Dell | Inspiron 7579 | 2 in 1 Convertible | 15.6" | IPS Panel Full HD / Touchscreen 1920x1080 | Intel Core i7 7500U 2.7GHz | 12GB | 512GB SSD | Intel HD Graphics 620 | Windows | 10 | 2.19kg | 11550708
... ...
"""

prompt_react_code_hist_analysis_ATP = """
Solve a question answering task with interleaving Thought, Code, Action, Results steps. Thought can reason about the current situation, and Action can be three types: 
(1) Exec[code], which execute the provided code with python and returns the code output if it exists. 
(2) Terminate[answer], which returns the answer and finishes the task.
Here is an examples.

------------------------------ Example Start: ------------------------------
The database table credit_card_risk is shown as follows: 
checking_status | duration | credit_history | purpose | credit_amount | savings_status | employment | installment_commitment | personal_status | other_parties | residence_since | property_magnitude | age | other_payment_plans | housing | existing_credits | job | num_dependents | own_telephone | foreign_worker | class
<0 | 6.0 | critical/other existing credit | radio/tv | 1169.0 | no known savings | >=7 | 4.0 | male single | none | 4.0 | real estate | 67.0 | none | own | 2.0 | skilled | 1.0 | yes | yes | good
0<=X<200 | 48.0 | existing paid | radio/tv | 5951.0 | <100 | 1<=X<4 | 2.0 | female div/dep/mar | none | 2.0 | real estate | 22.0 | none | own | 1.0 | skilled | 1.0 | none | yes | bad
no checking | 12.0 | critical/other existing credit | education | 2096.0 | <100 | 4<=X<7 | 2.0 | male single | none | 3.0 | real estate | 49.0 | none | own | 1.0 | unskilled resident | 2.0 | none | yes | good
... ...

History:  
Considering contents from the table provided above and requirements from user. Please note DO NOT CHANGE FILE AND VARIABLE NAMES THAT I HAVE SET!  
Interactions begin:  
[USER (data scientist)]: Hi, ... ...  
  
[YOU (AI assistant)]: ... ... [PREVIOUS_CODE]  

... ...

[USER (data scientist)]: What is the optimal number of clusters using the silhouette score? 

Options:  
A. 1  
B. 3 
C. 4
D. None of above 

NOTE: You have at most 5 turns to generate final results, which means you have to provide the answer at or earlier than Thought 5.

[YOU (AI assistant)]: Let's break down the code generation and solve the multi-choice question turn by turn!
################### Your Answer Starts Here: ################### 
Turn 1:
# 5 turns left to provide final answer.
Thought 1: First import all packages needed and load the dataset.
Code 1:
'''
import pandas as pd  
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score 
import matplotlib.pyplot as plt

credit_customers = pd.read_csv("credit_customers.csv")
print(credit_customers)
'''
Act 1: Exec[Code 1]
Result 1: 
checking_status | duration | credit_history | purpose | credit_amount | savings_status | employment | installment_commitment | personal_status | other_parties | residence_since | property_magnitude | age | other_payment_plans | housing | existing_credits | job | num_dependents | own_telephone | foreign_worker | class
<0 | 6.0 | critical/other existing credit | radio/tv | 1169.0 | no known savings | >=7 | 4.0 | male single | none | 4.0 | real estate | 67.0 | none | own | 2.0 | skilled | 1.0 | yes | yes | good
0<=X<200 | 48.0 | existing paid | radio/tv | 5951.0 | <100 | 1<=X<4 | 2.0 | female div/dep/mar | none | 2.0 | real estate | 22.0 | none | own | 1.0 | skilled | 1.0 | none | yes | bad
no checking | 12.0 | critical/other existing credit | education | 2096.0 | <100 | 4<=X<7 | 2.0 | male single | none | 3.0 | real estate | 49.0 | none | own | 1.0 | unskilled resident | 2.0 | none | yes | good
... ...

Turn 2:
# 4 turns left to provide final answer.
Thought 2: Calculate the silhouette scores.
Code 2:
'''
silhouette_scores = [] 
for cluster_num in range(2, 15):  # Starting from 2 because silhouette score is not defined for 1 cluster 
    kmeans = KMeans(n_clusters=cluster_num, random_state=42) 
    cluster_labels = kmeans.fit_predict(data_for_clustering_scaled) 
    silhouette_avg = silhouette_score(data_for_clustering_scaled, cluster_labels) 
    silhouette_scores.append(silhouette_avg) 

print(silhouette_scores)
'''
Act 2: Exec[Code 3]
Result 2: [0.25, 0.34, 0.45, 0.67, 0.53, 0.37]

Turn 3:
# 3 turns left to provide final answer.
Thought 3: Plot the silhouette scores.
Code 3:
'''
# Plot silhouette scores 
plt.figure(figsize=(10, 6)) 
plt.plot(range(2, 15), silhouette_scores, marker='o', linestyle='--') 
plt.title('Silhouette Scores for Different Number of Clusters') 
plt.xlabel('Number of Clusters') 
plt.ylabel('Silhouette Score') 
plt.grid(True) 
plt.savefig('pred_result/Silhouette_Scores.png') 
plt.show() 
'''
Act 3: Exec[Code 3]
Result 3: [PLOT]

Turn 4:
# 2 turns left to provide final answer.
Thought 4: Analyze all results I have.
Analysis: According to the plot, when the number of clusters equals to 4 lead to highest Silhouette Scores, so we choose C.
Act 4: Terminate[C]
Answer: C
################### Your Answer Ends Here: ################### 
------------------------------ Example End ------------------------------ 
  
The database table atp_tennis is shown as follows:   
Tournament | Date | Series | Court | Surface | Round | Best of | Player_1 | Player_2 | Winner | Rank_1 | Rank_2 | Pts_1 | Pts_2 | Odd_1 | Odd_2 | score  
Brisbane International | 2012-12-31 | ATP250 | Outdoor | Hard | 1st Round | 3 | Mayer F. | Giraldo S. | Mayer F. | 28 | 57 | 1215 | 778 | 1.36 | 3.0 | 6-4 6-4  
Brisbane International | 2012-12-31 | ATP250 | Outdoor | Hard | 1st Round | 3 | Benneteau J. | Nieminen J. | Nieminen J. | 35 | 41 | 1075 | 927 | 2.2 | 1.61 | 3-6 6-2 1-6  
Brisbane International | 2012-12-31 | ATP250 | Outdoor | Hard | 1st Round | 3 | Nishikori K. | Matosevic M. | Nishikori K. | 19 | 49 | 1830 | 845 | 1.25 | 3.75 | 7-5 6-2  
... ...

Please note that the pandas package version is 2.0.3 and matplotlib package version is 3.7.4 in all code here.

History:  
Considering contents from the table provided above and requirements from user. Please note DO NOT CHANGE FILE AND VARIABLE NAMES THAT I HAVE SET!  
Interactions begin:

{dialogue_hist}

NOTE: Please follow my example between "Example Start" and "Example End" above to answer task with interleaving Thought, Code, Action, Result turns. Each turn contains one segment of the target code to solve the problem. 
Please generate ONLY one turn this time and wait for User to give Result based on your generated code segment, DON'T generate the whole code in a single turn! 
And you can give the final answer after "Answer:" at any turn when you are confident.

[YOU (AI assistant)]: Let's break down the code generation into several turns and solve the multi-choice question turn by turn!
################### Your Answer Starts Here: ################### 
Turn 1:
# 5 turns left to provide final answer. Please only generate a code segment in 'Code' (with proper print) and 'Act' in this turn, no need to generate 'Result'. DON'T generate the whole code in a single turn! 
Thought 1: First import all packages needed and load the dataset.
Code 1:
"""

prompt_react_code_hist_analysis_fastfood = """
Solve a question answering task with interleaving Thought, Code, Action, Results steps. Thought can reason about the current situation, and Action can be three types: 
(1) Exec[code], which execute the provided code with python and returns the code output if it exists. 
(2) Terminate[answer], which returns the answer and finishes the task.
Here is an examples.

------------------------------ Example Start: ------------------------------
The database table credit_card_risk is shown as follows: 
checking_status | duration | credit_history | purpose | credit_amount | savings_status | employment | installment_commitment | personal_status | other_parties | residence_since | property_magnitude | age | other_payment_plans | housing | existing_credits | job | num_dependents | own_telephone | foreign_worker | class
<0 | 6.0 | critical/other existing credit | radio/tv | 1169.0 | no known savings | >=7 | 4.0 | male single | none | 4.0 | real estate | 67.0 | none | own | 2.0 | skilled | 1.0 | yes | yes | good
0<=X<200 | 48.0 | existing paid | radio/tv | 5951.0 | <100 | 1<=X<4 | 2.0 | female div/dep/mar | none | 2.0 | real estate | 22.0 | none | own | 1.0 | skilled | 1.0 | none | yes | bad
no checking | 12.0 | critical/other existing credit | education | 2096.0 | <100 | 4<=X<7 | 2.0 | male single | none | 3.0 | real estate | 49.0 | none | own | 1.0 | unskilled resident | 2.0 | none | yes | good
... ...

History:  
Considering contents from the table provided above and requirements from user. Please note DO NOT CHANGE FILE AND VARIABLE NAMES THAT I HAVE SET!  
Interactions begin:  
[USER (data scientist)]: Hi, ... ...  
  
[YOU (AI assistant)]: ... ... [PREVIOUS_CODE]  

... ...

[USER (data scientist)]: What is the optimal number of clusters using the silhouette score? 

Options:  
A. 1  
B. 3 
C. 4
D. None of above 

NOTE: You have at most 5 turns to generate final results, which means you have to provide the answer at or earlier than Thought 5.

[YOU (AI assistant)]: Let's break down the code generation and solve the multi-choice question turn by turn!
################### Your Answer Starts Here: ################### 
Turn 1:
# 5 turns left to provide final answer.
Thought 1: First import all packages needed and load the dataset.
Code 1:
'''
import pandas as pd  
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score 
import matplotlib.pyplot as plt

credit_customers = pd.read_csv("credit_customers.csv")
print(credit_customers)
'''
Act 1: Exec[Code 1]
Result 1: 
checking_status | duration | credit_history | purpose | credit_amount | savings_status | employment | installment_commitment | personal_status | other_parties | residence_since | property_magnitude | age | other_payment_plans | housing | existing_credits | job | num_dependents | own_telephone | foreign_worker | class
<0 | 6.0 | critical/other existing credit | radio/tv | 1169.0 | no known savings | >=7 | 4.0 | male single | none | 4.0 | real estate | 67.0 | none | own | 2.0 | skilled | 1.0 | yes | yes | good
0<=X<200 | 48.0 | existing paid | radio/tv | 5951.0 | <100 | 1<=X<4 | 2.0 | female div/dep/mar | none | 2.0 | real estate | 22.0 | none | own | 1.0 | skilled | 1.0 | none | yes | bad
no checking | 12.0 | critical/other existing credit | education | 2096.0 | <100 | 4<=X<7 | 2.0 | male single | none | 3.0 | real estate | 49.0 | none | own | 1.0 | unskilled resident | 2.0 | none | yes | good
... ...

Turn 2:
# 4 turns left to provide final answer.
Thought 2: Calculate the silhouette scores.
Code 2:
'''
silhouette_scores = [] 
for cluster_num in range(2, 15):  # Starting from 2 because silhouette score is not defined for 1 cluster 
    kmeans = KMeans(n_clusters=cluster_num, random_state=42) 
    cluster_labels = kmeans.fit_predict(data_for_clustering_scaled) 
    silhouette_avg = silhouette_score(data_for_clustering_scaled, cluster_labels) 
    silhouette_scores.append(silhouette_avg) 

print(silhouette_scores)
'''
Act 2: Exec[Code 3]
Result 2: [0.25, 0.34, 0.45, 0.67, 0.53, 0.37]

Turn 3:
# 3 turns left to provide final answer.
Thought 3: Plot the silhouette scores.
Code 3:
'''
# Plot silhouette scores 
plt.figure(figsize=(10, 6)) 
plt.plot(range(2, 15), silhouette_scores, marker='o', linestyle='--') 
plt.title('Silhouette Scores for Different Number of Clusters') 
plt.xlabel('Number of Clusters') 
plt.ylabel('Silhouette Score') 
plt.grid(True) 
plt.savefig('pred_result/Silhouette_Scores.png') 
plt.show() 
'''
Act 3: Exec[Code 3]
Result 3: [PLOT]

Turn 4:
# 2 turns left to provide final answer.
Thought 4: Analyze all results I have.
Analysis: According to the plot, when the number of clusters equals to 4 lead to highest Silhouette Scores, so we choose C.
Act 4: Terminate[C]
Answer: C
################### Your Answer Ends Here: ################### 
------------------------------ Example End ------------------------------ 
  
The database table atp_tennis is shown as follows:   
restaurant | item | calories | cal_fat | total_fat | sat_fat | trans_fat | cholesterol | sodium | total_carb | fiber | sugar | protein | vit_a | vit_c | calcium | salad
Mcdonalds | Artisan Grilled Chicken Sandwich | 380 | 60 | 7 | 2 | 0 | 95 | 1110 | 44 | 3 | 11 | 37 | 4 | 20 | 20 | Other
Mcdonalds | Single Bacon Smokehouse Burger | 840 | 410 | 45 | 17 | 1.5 | 130 | 1580 | 62 | 2 | 18 | 46 | 6 | 20 | 20 | Other
Mcdonalds | Double Bacon Smokehouse Burger | 1130 | 600 | 67 | 27 | 3 | 220 | 1920 | 63 | 3 | 18 | 70 | 10 | 20 | 50 | Other 
... ...

Please note that the pandas package version is 2.0.3 and matplotlib package version is 3.7.4 in all code here.

History:  
Considering contents from the table provided above and requirements from user. Please note DO NOT CHANGE FILE AND VARIABLE NAMES THAT I HAVE SET!  
Interactions begin:

{dialogue_hist}

NOTE: Please follow my example between "Example Start" and "Example End" above to answer task with interleaving Thought, Code, Action, Result turns. Each turn contains one segment of the target code to solve the problem. 
Please generate ONLY one turn this time and wait for User to give Result based on your generated code segment, DON'T generate the whole code in a single turn! 
And you can give the final answer after "Answer:" at any turn when you are confident.

[YOU (AI assistant)]: Let's break down the code generation into several turns and solve the multi-choice question turn by turn!
################### Your Answer Starts Here: ################### 
Turn 1:
# 5 turns left to provide final answer. Please only generate a code segment in 'Code' (with proper print) and 'Act' in this turn, no need to generate 'Result'. DON'T generate the whole code in a single turn! 
Thought 1: First import all packages needed and load the dataset.
Code 1: Load the data from 'fastfood.csv'.
'''
"""

prompt_react_code_hist_analysis_laptop_AIR = """
Solve a question answering task with interleaving Thought, Code, Action, Results steps. Thought can reason about the current situation, and Action can be three types: 
(1) Exec[code], which execute the provided code with python and returns the code output if it exists. 
(2) Terminate[answer], which returns the answer and finishes the task.
Here is an examples.

------------------------------ Example Start: ------------------------------
The database table credit_card_risk is shown as follows: 
checking_status | duration | credit_history | purpose | credit_amount | savings_status | employment | installment_commitment | personal_status | other_parties | residence_since | property_magnitude | age | other_payment_plans | housing | existing_credits | job | num_dependents | own_telephone | foreign_worker | class
<0 | 6.0 | critical/other existing credit | radio/tv | 1169.0 | no known savings | >=7 | 4.0 | male single | none | 4.0 | real estate | 67.0 | none | own | 2.0 | skilled | 1.0 | yes | yes | good
0<=X<200 | 48.0 | existing paid | radio/tv | 5951.0 | <100 | 1<=X<4 | 2.0 | female div/dep/mar | none | 2.0 | real estate | 22.0 | none | own | 1.0 | skilled | 1.0 | none | yes | bad
no checking | 12.0 | critical/other existing credit | education | 2096.0 | <100 | 4<=X<7 | 2.0 | male single | none | 3.0 | real estate | 49.0 | none | own | 1.0 | unskilled resident | 2.0 | none | yes | good
... ...

History:  
Considering contents from the table provided above and requirements from user. Please note DO NOT CHANGE FILE AND VARIABLE NAMES THAT I HAVE SET!  
Interactions begin:  
[USER (data scientist)]: Hi, ... ...  
  
[YOU (AI assistant)]: ... ... [PREVIOUS_CODE]  

... ...

[USER (data scientist)]: What is the optimal number of clusters using the silhouette score? 

Options:  
A. 1  
B. 3 
C. 4
D. None of above 

Please firstly analysis the given pseudocode to answer question with interleaving Thought, Code, Action, Result turns. Each turn contains one segment of the target code to solve the problem. 
--- Pseudocode Starts: ---
START
  IMPORT pandas library as pd
  IMPORT silhouette_score from sklearn.metrics
  IMPORT pickle
  IMPORT matplotlib.pyplot as plt

  SET credit_customers TO pd.read_csv("credit_customers.csv")

  ... ...

  PLOT silhouette_score
  CALL plt.savefig('pred_result/Silhouette_Scores.png') 
END
--- Pseudocode Ends ---

NOTE: You have at most 5 turns to generate final results, which means you have to provide the answer at or earlier than Thought 5.

[YOU (AI assistant)]: Let's break down the code generation and solve the multi-choice question turn by turn!
################### Your Answer Starts Here: ################### 
Turn 1:
# 5 turns left to provide final answer.
Thought 1: First import all packages needed and load the dataset.
Code 1:
'''
import pandas as pd  
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score 
import matplotlib.pyplot as plt

credit_customers = pd.read_csv("credit_customers.csv")
print(credit_customers)
'''
Act 1: Exec[Code 1]
Result 1: 
checking_status | duration | credit_history | purpose | credit_amount | savings_status | employment | installment_commitment | personal_status | other_parties | residence_since | property_magnitude | age | other_payment_plans | housing | existing_credits | job | num_dependents | own_telephone | foreign_worker | class
<0 | 6.0 | critical/other existing credit | radio/tv | 1169.0 | no known savings | >=7 | 4.0 | male single | none | 4.0 | real estate | 67.0 | none | own | 2.0 | skilled | 1.0 | yes | yes | good
0<=X<200 | 48.0 | existing paid | radio/tv | 5951.0 | <100 | 1<=X<4 | 2.0 | female div/dep/mar | none | 2.0 | real estate | 22.0 | none | own | 1.0 | skilled | 1.0 | none | yes | bad
no checking | 12.0 | critical/other existing credit | education | 2096.0 | <100 | 4<=X<7 | 2.0 | male single | none | 3.0 | real estate | 49.0 | none | own | 1.0 | unskilled resident | 2.0 | none | yes | good
... ...

Turn 2:
# 4 turns left to provide final answer.
Thought 2: Calculate the silhouette scores.
Code 2:
'''
silhouette_scores = [] 
for cluster_num in range(2, 15):  # Starting from 2 because silhouette score is not defined for 1 cluster 
    kmeans = KMeans(n_clusters=cluster_num, random_state=42) 
    cluster_labels = kmeans.fit_predict(data_for_clustering_scaled) 
    silhouette_avg = silhouette_score(data_for_clustering_scaled, cluster_labels) 
    silhouette_scores.append(silhouette_avg) 

print(silhouette_scores)
'''
Act 2: Exec[Code 3]
Result 2: [0.25, 0.34, 0.45, 0.67, 0.53, 0.37]

Turn 3:
# 3 turns left to provide final answer.
Thought 3: Plot the silhouette scores.
Code 3:
'''
# Plot silhouette scores 
plt.figure(figsize=(10, 6)) 
plt.plot(range(2, 15), silhouette_scores, marker='o', linestyle='--') 
plt.title('Silhouette Scores for Different Number of Clusters') 
plt.xlabel('Number of Clusters') 
plt.ylabel('Silhouette Score') 
plt.grid(True) 
plt.savefig('pred_result/Silhouette_Scores.png') 
plt.show() 
'''
Act 3: Exec[Code 3]
Result 3: [PLOT]

Turn 4:
# 2 turns left to provide final answer.
Thought 4: Analyze all results I have.
Analysis: According to the plot, when the number of clusters equals to 4 lead to highest Silhouette Scores, so we choose C.
Act 4: Terminate[C]
Answer: C
################### Your Answer Ends Here: ################### 
------------------------------ Example End ------------------------------ 
  
The database table atp_tennis is shown as follows:   
Manufacturer | Model Name | Category | Screen Size | Screen | CPU | RAM |  Storage | GPU | Operating System | Operating System Version | Weight | Price
HP | 15-bs053od (i7-7500U/6GB/1TB/W10) | Notebook | 15.6" | 1366x768 | Intel Core i7 7500U 2.7GHz | 6GB | 1TB HDD | Intel HD Graphics 620 | Windows | 10 | 2.04kg | 5148468
Asus | Rog GL753VE-DS74 | Gaming | 17.3" | Full HD 1920x1080 | Intel Core i7 7700HQ 2.8GHz | 16GB | 256GB SSD +  1TB HDD | Nvidia GeForce GTX 1050 Ti | Windows | 10 | 2.99kg | 15552108
Dell | Inspiron 7579 | 2 in 1 Convertible | 15.6" | IPS Panel Full HD / Touchscreen 1920x1080 | Intel Core i7 7500U 2.7GHz | 12GB | 512GB SSD | Intel HD Graphics 620 | Windows | 10 | 2.19kg | 11550708
... ...

Please note that the pandas package version is 2.0.3 and matplotlib package version is 3.7.4 in all code here.

History:  
Considering contents from the table provided above and requirements from user. Please note DO NOT CHANGE FILE AND VARIABLE NAMES THAT I HAVE SET!  
Interactions begin:

{dialogue_hist}

Please firstly analysis the given pseudocode to answer question and follow my example between "Example Start" and "Example End" above to answer question with interleaving Thought, Code, Action, Result turns. Each turn contains one segment of the target code to solve the problem. 
--- Pseudocode Starts: ---
{pseudo_code}
--- Pseudocode Ends ---

NOTE: Please generate ONLY one turn this time and wait for User to give Result based on your generated code segment, DON'T generate the whole code in a single turn! 
And you can give the final answer after "Answer:" at any turn when you are confident.

[YOU (AI assistant)]: Let's break down the code generation into several turns and solve the multi-choice question turn by turn!
################### Your Answer Starts Here: ################### 
Turn 1:
# 5 turns left to provide final answer. Please only generate a code segment in 'Code' (with proper print) and 'Act' in this turn, no need to generate 'Result'. DON'T generate the whole code in a single turn! 
Thought 1: First import all packages needed and load the dataset.
Code 1: Load the data from 'laptops_price.csv'.
'''
"""

prompt_react_code_hist_analysis_laptop = """
Solve a question answering task with interleaving Thought, Code, Action, Results steps. Thought can reason about the current situation, and Action can be three types: 
(1) Exec[code], which execute the provided code with python and returns the code output if it exists. 
(2) Terminate[answer], which returns the answer and finishes the task.
Here is an examples.

------------------------------ Example Start: ------------------------------
The database table credit_card_risk is shown as follows: 
checking_status | duration | credit_history | purpose | credit_amount | savings_status | employment | installment_commitment | personal_status | other_parties | residence_since | property_magnitude | age | other_payment_plans | housing | existing_credits | job | num_dependents | own_telephone | foreign_worker | class
<0 | 6.0 | critical/other existing credit | radio/tv | 1169.0 | no known savings | >=7 | 4.0 | male single | none | 4.0 | real estate | 67.0 | none | own | 2.0 | skilled | 1.0 | yes | yes | good
0<=X<200 | 48.0 | existing paid | radio/tv | 5951.0 | <100 | 1<=X<4 | 2.0 | female div/dep/mar | none | 2.0 | real estate | 22.0 | none | own | 1.0 | skilled | 1.0 | none | yes | bad
no checking | 12.0 | critical/other existing credit | education | 2096.0 | <100 | 4<=X<7 | 2.0 | male single | none | 3.0 | real estate | 49.0 | none | own | 1.0 | unskilled resident | 2.0 | none | yes | good
... ...

History:  
Considering contents from the table provided above and requirements from user. Please note DO NOT CHANGE FILE AND VARIABLE NAMES THAT I HAVE SET!  
Interactions begin:  
[USER (data scientist)]: Hi, ... ...  
  
[YOU (AI assistant)]: ... ... [PREVIOUS_CODE]  

... ...

[USER (data scientist)]: What is the optimal number of clusters using the silhouette score? 

Options:  
A. 1  
B. 3 
C. 4
D. None of above 

NOTE: You have at most 5 turns to generate final results, which means you have to provide the answer at or earlier than Thought 5.

[YOU (AI assistant)]: Let's break down the code generation and solve the multi-choice question turn by turn!
################### Your Answer Starts Here: ################### 
Turn 1:
# 5 turns left to provide final answer.
Thought 1: First import all packages needed and load the dataset.
Code 1:
'''
import pandas as pd  
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score 
import matplotlib.pyplot as plt

credit_customers = pd.read_csv("credit_customers.csv")
print(credit_customers)
'''
Act 1: Exec[Code 1]
Result 1: 
checking_status | duration | credit_history | purpose | credit_amount | savings_status | employment | installment_commitment | personal_status | other_parties | residence_since | property_magnitude | age | other_payment_plans | housing | existing_credits | job | num_dependents | own_telephone | foreign_worker | class
<0 | 6.0 | critical/other existing credit | radio/tv | 1169.0 | no known savings | >=7 | 4.0 | male single | none | 4.0 | real estate | 67.0 | none | own | 2.0 | skilled | 1.0 | yes | yes | good
0<=X<200 | 48.0 | existing paid | radio/tv | 5951.0 | <100 | 1<=X<4 | 2.0 | female div/dep/mar | none | 2.0 | real estate | 22.0 | none | own | 1.0 | skilled | 1.0 | none | yes | bad
no checking | 12.0 | critical/other existing credit | education | 2096.0 | <100 | 4<=X<7 | 2.0 | male single | none | 3.0 | real estate | 49.0 | none | own | 1.0 | unskilled resident | 2.0 | none | yes | good
... ...

Turn 2:
# 4 turns left to provide final answer.
Thought 2: Calculate the silhouette scores.
Code 2:
'''
silhouette_scores = [] 
for cluster_num in range(2, 15):  # Starting from 2 because silhouette score is not defined for 1 cluster 
    kmeans = KMeans(n_clusters=cluster_num, random_state=42) 
    cluster_labels = kmeans.fit_predict(data_for_clustering_scaled) 
    silhouette_avg = silhouette_score(data_for_clustering_scaled, cluster_labels) 
    silhouette_scores.append(silhouette_avg) 

print(silhouette_scores)
'''
Act 2: Exec[Code 3]
Result 2: [0.25, 0.34, 0.45, 0.67, 0.53, 0.37]

Turn 3:
# 3 turns left to provide final answer.
Thought 3: Plot the silhouette scores.
Code 3:
'''
# Plot silhouette scores 
plt.figure(figsize=(10, 6)) 
plt.plot(range(2, 15), silhouette_scores, marker='o', linestyle='--') 
plt.title('Silhouette Scores for Different Number of Clusters') 
plt.xlabel('Number of Clusters') 
plt.ylabel('Silhouette Score') 
plt.grid(True) 
plt.savefig('pred_result/Silhouette_Scores.png') 
plt.show() 
'''
Act 3: Exec[Code 3]
Result 3: [PLOT]

Turn 4:
# 2 turns left to provide final answer.
Thought 4: Analyze all results I have.
Analysis: According to the plot, when the number of clusters equals to 4 lead to highest Silhouette Scores, so we choose C.
Act 4: Terminate[C]
Answer: C
################### Your Answer Ends Here: ################### 
------------------------------ Example End ------------------------------ 
  
The database table atp_tennis is shown as follows:   
Manufacturer | Model Name | Category | Screen Size | Screen | CPU | RAM |  Storage | GPU | Operating System | Operating System Version | Weight | Price
HP | 15-bs053od (i7-7500U/6GB/1TB/W10) | Notebook | 15.6" | 1366x768 | Intel Core i7 7500U 2.7GHz | 6GB | 1TB HDD | Intel HD Graphics 620 | Windows | 10 | 2.04kg | 5148468
Asus | Rog GL753VE-DS74 | Gaming | 17.3" | Full HD 1920x1080 | Intel Core i7 7700HQ 2.8GHz | 16GB | 256GB SSD +  1TB HDD | Nvidia GeForce GTX 1050 Ti | Windows | 10 | 2.99kg | 15552108
Dell | Inspiron 7579 | 2 in 1 Convertible | 15.6" | IPS Panel Full HD / Touchscreen 1920x1080 | Intel Core i7 7500U 2.7GHz | 12GB | 512GB SSD | Intel HD Graphics 620 | Windows | 10 | 2.19kg | 11550708
... ...

Please note that the pandas package version is 2.0.3 and matplotlib package version is 3.7.4 in all code here.

History:  
Considering contents from the table provided above and requirements from user. Please note DO NOT CHANGE FILE AND VARIABLE NAMES THAT I HAVE SET!  
Interactions begin:

{dialogue_hist}

NOTE: Please follow my example between "Example Start" and "Example End" above to answer task with interleaving Thought, Code, Action, Result turns. Each turn contains one segment of the target code to solve the problem. 
Please generate ONLY one turn this time and wait for User to give Result based on your generated code segment, DON'T generate the whole code in a single turn! 
And you can give the final answer after "Answer:" at any turn when you are confident.

[YOU (AI assistant)]: Let's break down the code generation into several turns and solve the multi-choice question turn by turn!
################### Your Answer Starts Here: ################### 
Turn 1:
# 5 turns left to provide final answer. Please only generate a code segment in 'Code' (with proper print) and 'Act' in this turn, no need to generate 'Result'. DON'T generate the whole code in a single turn! 
Thought 1: First import all packages needed and load the dataset.
Code 1: Load the data from 'laptops_price.csv'.
'''
"""


prompt_react_code_hist_analysis_melb_AIR = """
Solve a question answering task with interleaving Thought, Code, Action, Results steps. Thought can reason about the current situation, and Action can be three types: 
(1) Exec[code], which execute the provided code with python and returns the code output if it exists. 
(2) Terminate[answer], which returns the answer and finishes the task.
Here is an examples.

------------------------------ Example Start: ------------------------------
The database table credit_card_risk is shown as follows: 
checking_status | duration | credit_history | purpose | credit_amount | savings_status | employment | installment_commitment | personal_status | other_parties | residence_since | property_magnitude | age | other_payment_plans | housing | existing_credits | job | num_dependents | own_telephone | foreign_worker | class
<0 | 6.0 | critical/other existing credit | radio/tv | 1169.0 | no known savings | >=7 | 4.0 | male single | none | 4.0 | real estate | 67.0 | none | own | 2.0 | skilled | 1.0 | yes | yes | good
0<=X<200 | 48.0 | existing paid | radio/tv | 5951.0 | <100 | 1<=X<4 | 2.0 | female div/dep/mar | none | 2.0 | real estate | 22.0 | none | own | 1.0 | skilled | 1.0 | none | yes | bad
no checking | 12.0 | critical/other existing credit | education | 2096.0 | <100 | 4<=X<7 | 2.0 | male single | none | 3.0 | real estate | 49.0 | none | own | 1.0 | unskilled resident | 2.0 | none | yes | good
... ...

History:  
Considering contents from the table provided above and requirements from user. Please note DO NOT CHANGE FILE AND VARIABLE NAMES THAT I HAVE SET!  
Interactions begin:  
[USER (data scientist)]: Hi, ... ...  
  
[YOU (AI assistant)]: ... ... [PREVIOUS_CODE]  

... ...

[USER (data scientist)]: What is the optimal number of clusters using the silhouette score? 

Options:  
A. 1  
B. 3 
C. 4
D. None of above 

Please firstly analysis the given pseudocode to answer question with interleaving Thought, Code, Action, Result turns. Each turn contains one segment of the target code to solve the problem. 
--- Pseudocode Starts: ---
START
  IMPORT pandas library as pd
  IMPORT silhouette_score from sklearn.metrics
  IMPORT pickle
  IMPORT matplotlib.pyplot as plt

  SET credit_customers TO pd.read_csv("credit_customers.csv")

  ... ...

  PLOT silhouette_score
  CALL plt.savefig('pred_result/Silhouette_Scores.png') 
END
--- Pseudocode Ends ---

NOTE: You have at most 5 turns to generate final results, which means you have to provide the answer at or earlier than Thought 5.

[YOU (AI assistant)]: Let's break down the code generation and solve the multi-choice question turn by turn!
################### Your Answer Starts Here: ################### 
Turn 1:
# 5 turns left to provide final answer.
Thought 1: First import all packages needed and load the dataset.
Code 1:
'''
import pandas as pd  
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score 
import matplotlib.pyplot as plt

credit_customers = pd.read_csv("credit_customers.csv")
print(credit_customers)
'''
Act 1: Exec[Code 1]
Result 1: 
checking_status | duration | credit_history | purpose | credit_amount | savings_status | employment | installment_commitment | personal_status | other_parties | residence_since | property_magnitude | age | other_payment_plans | housing | existing_credits | job | num_dependents | own_telephone | foreign_worker | class
<0 | 6.0 | critical/other existing credit | radio/tv | 1169.0 | no known savings | >=7 | 4.0 | male single | none | 4.0 | real estate | 67.0 | none | own | 2.0 | skilled | 1.0 | yes | yes | good
0<=X<200 | 48.0 | existing paid | radio/tv | 5951.0 | <100 | 1<=X<4 | 2.0 | female div/dep/mar | none | 2.0 | real estate | 22.0 | none | own | 1.0 | skilled | 1.0 | none | yes | bad
no checking | 12.0 | critical/other existing credit | education | 2096.0 | <100 | 4<=X<7 | 2.0 | male single | none | 3.0 | real estate | 49.0 | none | own | 1.0 | unskilled resident | 2.0 | none | yes | good
... ...

Turn 2:
# 4 turns left to provide final answer.
Thought 2: Calculate the silhouette scores.
Code 2:
'''
silhouette_scores = [] 
for cluster_num in range(2, 15):  # Starting from 2 because silhouette score is not defined for 1 cluster 
    kmeans = KMeans(n_clusters=cluster_num, random_state=42) 
    cluster_labels = kmeans.fit_predict(data_for_clustering_scaled) 
    silhouette_avg = silhouette_score(data_for_clustering_scaled, cluster_labels) 
    silhouette_scores.append(silhouette_avg) 

print(silhouette_scores)
'''
Act 2: Exec[Code 3]
Result 2: [0.25, 0.34, 0.45, 0.67, 0.53, 0.37]

Turn 3:
# 3 turns left to provide final answer.
Thought 3: Plot the silhouette scores.
Code 3:
'''
# Plot silhouette scores 
plt.figure(figsize=(10, 6)) 
plt.plot(range(2, 15), silhouette_scores, marker='o', linestyle='--') 
plt.title('Silhouette Scores for Different Number of Clusters') 
plt.xlabel('Number of Clusters') 
plt.ylabel('Silhouette Score') 
plt.grid(True) 
plt.savefig('pred_result/Silhouette_Scores.png') 
plt.show() 
'''
Act 3: Exec[Code 3]
Result 3: [PLOT]

Turn 4:
# 2 turns left to provide final answer.
Thought 4: Analyze all results I have.
Analysis: According to the plot, when the number of clusters equals to 4 lead to highest Silhouette Scores, so we choose C.
Act 4: Terminate[C]
Answer: C
################### Your Answer Ends Here: ################### 
------------------------------ Example End ------------------------------ 
  
The database table atp_tennis is shown as follows:   
Suburb | Address | Rooms | Type | Price | Method | SellerG | Date | Distance | Postcode | Bedroom2 | Bathroom | Car | Landsize | BuildingArea | YearBuilt | CouncilArea | Lattitude | Longtitude | Regionname | Propertycount
Abbotsford | 85 Turner St | 2 | h | 1480000.0 | S | Biggin | 3/12/2016 | 2.5 | 3067.0 | 2.0 | 1.0 | 1.0 | 202.0 |  |  | Yarra | -37.7996 | 144.9984 | Northern Metropolitan | 4019.0
Abbotsford | 25 Bloomburg St | 2 | h | 1035000.0 | S | Biggin | 4/02/2016 | 2.5 | 3067.0 | 2.0 | 1.0 | 0.0 | 156.0 | 79.0 | 1900.0 | Yarra | -37.8079 | 144.9934 | Northern Metropolitan | 4019.0
Abbotsford | 5 Charles St | 3 | h | 1465000.0 | SP | Biggin | 4/03/2017 | 2.5 | 3067.0 | 3.0 | 2.0 | 0.0 | 134.0 | 150.0 | 1900.0 | Yarra | -37.8093 | 144.9944 | Northern Metropolitan | 4019.0
... ...

Please note that the pandas package version is 2.0.3 and matplotlib package version is 3.7.4 in all code here.

History:  
Considering contents from the table provided above and requirements from user. Please note DO NOT CHANGE FILE AND VARIABLE NAMES THAT I HAVE SET!  
Interactions begin:

{dialogue_hist}

Please firstly analysis the given pseudocode to answer question and follow my example between "Example Start" and "Example End" above to answer question with interleaving Thought, Code, Action, Result turns. Each turn contains one segment of the target code to solve the problem. 
--- Pseudocode Starts: ---
{pseudo_code}
--- Pseudocode Ends ---

NOTE: Please generate ONLY one turn this time and wait for User to give Result based on your generated code segment, DON'T generate the whole code in a single turn! 
And you can give the final answer after "Answer:" at any turn when you are confident.

[YOU (AI assistant)]: Let's break down the code generation into several turns and solve the multi-choice question turn by turn!
################### Your Answer Starts Here: ################### 
Turn 1:
# 5 turns left to provide final answer. Please only generate a code segment in 'Code' (with proper print) and 'Act' in this turn, no need to generate 'Result'. DON'T generate the whole code in a single turn! 
Thought 1: First import all packages needed and load the dataset.
Code 1: Load the data from 'melb_data.csv'.
'''
"""

prompt_react_code_hist_analysis_melb = """
Solve a question answering task with interleaving Thought, Code, Action, Results steps. Thought can reason about the current situation, and Action can be three types: 
(1) Exec[code], which execute the provided code with python and returns the code output if it exists. 
(2) Terminate[answer], which returns the answer and finishes the task.
Here is an examples.

------------------------------ Example Start: ------------------------------
The database table credit_card_risk is shown as follows: 
checking_status | duration | credit_history | purpose | credit_amount | savings_status | employment | installment_commitment | personal_status | other_parties | residence_since | property_magnitude | age | other_payment_plans | housing | existing_credits | job | num_dependents | own_telephone | foreign_worker | class
<0 | 6.0 | critical/other existing credit | radio/tv | 1169.0 | no known savings | >=7 | 4.0 | male single | none | 4.0 | real estate | 67.0 | none | own | 2.0 | skilled | 1.0 | yes | yes | good
0<=X<200 | 48.0 | existing paid | radio/tv | 5951.0 | <100 | 1<=X<4 | 2.0 | female div/dep/mar | none | 2.0 | real estate | 22.0 | none | own | 1.0 | skilled | 1.0 | none | yes | bad
no checking | 12.0 | critical/other existing credit | education | 2096.0 | <100 | 4<=X<7 | 2.0 | male single | none | 3.0 | real estate | 49.0 | none | own | 1.0 | unskilled resident | 2.0 | none | yes | good
... ...

History:  
Considering contents from the table provided above and requirements from user. Please note DO NOT CHANGE FILE AND VARIABLE NAMES THAT I HAVE SET!  
Interactions begin:  
[USER (data scientist)]: Hi, ... ...  
  
[YOU (AI assistant)]: ... ... [PREVIOUS_CODE]  

... ...

[USER (data scientist)]: What is the optimal number of clusters using the silhouette score? 

Options:  
A. 1  
B. 3 
C. 4
D. None of above 

NOTE: You have at most 5 turns to generate final results, which means you have to provide the answer at or earlier than Thought 5.

[YOU (AI assistant)]: Let's break down the code generation and solve the multi-choice question turn by turn!
################### Your Answer Starts Here: ################### 
Turn 1:
# 5 turns left to provide final answer.
Thought 1: First import all packages needed and load the dataset.
Code 1:
'''
import pandas as pd  
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score 
import matplotlib.pyplot as plt

credit_customers = pd.read_csv("credit_customers.csv")
print(credit_customers)
'''
Act 1: Exec[Code 1]
Result 1: 
checking_status | duration | credit_history | purpose | credit_amount | savings_status | employment | installment_commitment | personal_status | other_parties | residence_since | property_magnitude | age | other_payment_plans | housing | existing_credits | job | num_dependents | own_telephone | foreign_worker | class
<0 | 6.0 | critical/other existing credit | radio/tv | 1169.0 | no known savings | >=7 | 4.0 | male single | none | 4.0 | real estate | 67.0 | none | own | 2.0 | skilled | 1.0 | yes | yes | good
0<=X<200 | 48.0 | existing paid | radio/tv | 5951.0 | <100 | 1<=X<4 | 2.0 | female div/dep/mar | none | 2.0 | real estate | 22.0 | none | own | 1.0 | skilled | 1.0 | none | yes | bad
no checking | 12.0 | critical/other existing credit | education | 2096.0 | <100 | 4<=X<7 | 2.0 | male single | none | 3.0 | real estate | 49.0 | none | own | 1.0 | unskilled resident | 2.0 | none | yes | good
... ...

Turn 2:
# 4 turns left to provide final answer.
Thought 2: Calculate the silhouette scores.
Code 2:
'''
silhouette_scores = [] 
for cluster_num in range(2, 15):  # Starting from 2 because silhouette score is not defined for 1 cluster 
    kmeans = KMeans(n_clusters=cluster_num, random_state=42) 
    cluster_labels = kmeans.fit_predict(data_for_clustering_scaled) 
    silhouette_avg = silhouette_score(data_for_clustering_scaled, cluster_labels) 
    silhouette_scores.append(silhouette_avg) 

print(silhouette_scores)
'''
Act 2: Exec[Code 3]
Result 2: [0.25, 0.34, 0.45, 0.67, 0.53, 0.37]

Turn 3:
# 3 turns left to provide final answer.
Thought 3: Plot the silhouette scores.
Code 3:
'''
# Plot silhouette scores 
plt.figure(figsize=(10, 6)) 
plt.plot(range(2, 15), silhouette_scores, marker='o', linestyle='--') 
plt.title('Silhouette Scores for Different Number of Clusters') 
plt.xlabel('Number of Clusters') 
plt.ylabel('Silhouette Score') 
plt.grid(True) 
plt.savefig('pred_result/Silhouette_Scores.png') 
plt.show() 
'''
Act 3: Exec[Code 3]
Result 3: [PLOT]

Turn 4:
# 2 turns left to provide final answer.
Thought 4: Analyze all results I have.
Analysis: According to the plot, when the number of clusters equals to 4 lead to highest Silhouette Scores, so we choose C.
Act 4: Terminate[C]
Answer: C
################### Your Answer Ends Here: ################### 
------------------------------ Example End ------------------------------ 
  
The database table atp_tennis is shown as follows:   
Suburb | Address | Rooms | Type | Price | Method | SellerG | Date | Distance | Postcode | Bedroom2 | Bathroom | Car | Landsize | BuildingArea | YearBuilt | CouncilArea | Lattitude | Longtitude | Regionname | Propertycount
Abbotsford | 85 Turner St | 2 | h | 1480000.0 | S | Biggin | 3/12/2016 | 2.5 | 3067.0 | 2.0 | 1.0 | 1.0 | 202.0 |  |  | Yarra | -37.7996 | 144.9984 | Northern Metropolitan | 4019.0
Abbotsford | 25 Bloomburg St | 2 | h | 1035000.0 | S | Biggin | 4/02/2016 | 2.5 | 3067.0 | 2.0 | 1.0 | 0.0 | 156.0 | 79.0 | 1900.0 | Yarra | -37.8079 | 144.9934 | Northern Metropolitan | 4019.0
Abbotsford | 5 Charles St | 3 | h | 1465000.0 | SP | Biggin | 4/03/2017 | 2.5 | 3067.0 | 3.0 | 2.0 | 0.0 | 134.0 | 150.0 | 1900.0 | Yarra | -37.8093 | 144.9944 | Northern Metropolitan | 4019.0
... ...

Please note that the pandas package version is 2.0.3 and matplotlib package version is 3.7.4 in all code here.

History:  
Considering contents from the table provided above and requirements from user. Please note DO NOT CHANGE FILE AND VARIABLE NAMES THAT I HAVE SET!  
Interactions begin:

{dialogue_hist}

NOTE: Please follow my example between "Example Start" and "Example End" above to answer task with interleaving Thought, Code, Action, Result turns. Each turn contains one segment of the target code to solve the problem. 
Please generate ONLY one turn this time and wait for User to give Result based on your generated code segment, DON'T generate the whole code in a single turn! 
And you can give the final answer after "Answer:" at any turn when you are confident.

[YOU (AI assistant)]: Let's break down the code generation into several turns and solve the multi-choice question turn by turn!
################### Your Answer Starts Here: ################### 
Turn 1:
# 5 turns left to provide final answer. Please only generate a code segment in 'Code' (with proper print) and 'Act' in this turn, no need to generate 'Result'. DON'T generate the whole code in a single turn! 
Thought 1: First import all packages needed and load the dataset.
Code 1: Load the data from 'melb_data.csv'.
'''
"""


prompt_analysis_baseline = """"
Please generate the python code (with pandas version 2.0.3 and matplotlib version 3.7.4) between <code>...</code> to answer the first question and based on the answer choose the most appropriate option and directly provide the choice between <choice>...</choice>.
"""

prompt_react_code_hist_analysis_credit = """
Solve a question answering task with interleaving Thought, Code, Action, Results steps. Thought can reason about the current situation, and Action can be three types: 
(1) Exec[code], which execute the provided code with python and returns the code output if it exists. 
(2) Terminate[answer], which returns the answer and finishes the task.
Here is an examples.

------------------------------ Example Start: ------------------------------
The database table atp_tennis is shown as follows: 
Tournament | Date | Series | Court | Surface | Round | Best of | Player_1 | Player_2 | Winner | Rank_1 | Rank_2 | Pts_1 | Pts_2 | Odd_1 | Odd_2 | score  
Brisbane International | 2012-12-31 | ATP250 | Outdoor | Hard | 1st Round | 3 | Mayer F. | Giraldo S. | Mayer F. | 28 | 57 | 1215 | 778 | 1.36 | 3.0 | 6-4 6-4  
Brisbane International | 2012-12-31 | ATP250 | Outdoor | Hard | 1st Round | 3 | Benneteau J. | Nieminen J. | Nieminen J. | 35 | 41 | 1075 | 927 | 2.2 | 1.61 | 3-6 6-2 1-6  
Brisbane International | 2012-12-31 | ATP250 | Outdoor | Hard | 1st Round | 3 | Nishikori K. | Matosevic M. | Nishikori K. | 19 | 49 | 1830 | 845 | 1.25 | 3.75 | 7-5 6-2  
... ...

Please note that the pandas package version is 2.0.3 and matplotlib package version is 3.7.4 in all code here.

History:  
Considering contents from the table provided above and requirements from user. Please note DO NOT CHANGE FILE AND VARIABLE NAMES THAT I HAVE SET!  
Interactions begin:  
[USER (data scientist)]: Hi, ... ...  
  
[YOU (AI assistant)]: ... ... [PREVIOUS_CODE]  

... ...

[USER (data scientist)]: Hi, let's start by assessing the data quality of the atp_tennis dataset. To achieve this, we'll create a Data Quality Report DataFrame that covers two main aspects: a list of columns along with the number of missing values in each column, A list of unique values for each categorical column, such as Tournament, Series, Court, Surface, and Round. And please answer the following question: How many unique value of Court?

Options:  
A. 1  
B. 2  
C. 3  
D. None of above 

NOTE: You have at most 5 turns to generate final results, which means you have to provide the answer at or earlier than Thought 5.

[YOU (AI assistant)]: Let's break down the code generation and solve the multi-choice question turn by turn!
################### Your Answer Starts Here: ################### 
Turn 1:
# 5 turns left to provide final answer.
Thought 1: First import all packages needed and load the dataset.
Code 1:
'''
import pandas as pd
import pickle

atp_tennis = pd.read_csv('atp_tennis.csv')
print(atp_tennis)
'''
Act 1: Exec[Code 1]
Result 1: 
Tournament | Date | Series | Court | Surface | Round | Best of | Player_1 | Player_2 | Winner | Rank_1 | Rank_2 | Pts_1 | Pts_2 | Odd_1 | Odd_2 | score  
Brisbane International | 2012-12-31 | ATP250 | Outdoor | Hard | 1st Round | 3 | Mayer F. | Giraldo S. | Mayer F. | 28 | 57 | 1215 | 778 | 1.36 | 3.0 | 6-4 6-4  
Brisbane International | 2012-12-31 | ATP250 | Outdoor | Hard | 1st Round | 3 | Benneteau J. | Nieminen J. | Nieminen J. | 35 | 41 | 1075 | 927 | 2.2 | 1.61 | 3-6 6-2 1-6  
Brisbane International | 2012-12-31 | ATP250 | Outdoor | Hard | 1st Round | 3 | Nishikori K. | Matosevic M. | Nishikori K. | 19 | 49 | 1830 | 845 | 1.25 | 3.75 | 7-5 6-2  
... ...

Turn 2:
# 4 turns left to provide final answer.
Thought 2: Then create a Data Quality Report Dataframe for the atp_tennis dataset.
Code 2:
'''
missing_values = atp_tennis.isnull().sum()
unique_values = atp_tennis[['Tournament', 'Series', 'Court', 'Surface', 'Round']].nunique()

data_quality_report = pd.DataFrame('Missing Values': missing_values, 'Unique Values': unique_values)
print(data_quality_report)
'''
Act 2: Exec[Code 3]
Result 2:  
Missing Values | Unique Values
Best of | 0 | NaN
Court | 0 | 2.0
Date | 0 | NaN
Odd_1 | 0 | NaN
... ...

Turn 3:
# 3 turns left to provide final answer.
Thought 3: Print the data quality report and save it to my computer.
Code 3:
'''
print(data_quality_report)
pickle.dump(data_quality_report,open("./pred_result/data_quality_report.pkl","wb"))
'''
Act 3: Exec[Code 3]
Result 3: 
Missing Values | Unique Values
Best of | 0 | NaN
Court | 0 | 2.0
Date | 0 | NaN
Odd_1 | 0 | NaN
... ...

Turn 4:
# 2 turns left to provide final answer.
Thought 4: Analyze all results I have.
Analysis: According to the data quality report, the unique value of Court is 2.0, so we choose B.
Act 4: Terminate[B]
Answer: B
################### Your Answer Ends Here: ################### 
------------------------------ Example End ------------------------------ 
  
The database table credit_card_risk is shown as follows:   
checking_status | duration | credit_history | purpose | credit_amount | savings_status | employment | installment_commitment | personal_status | other_parties | residence_since | property_magnitude | age | other_payment_plans | housing | existing_credits | job | num_dependents | own_telephone | foreign_worker | class
<0 | 6.0 | critical/other existing credit | radio/tv | 1169.0 | no known savings | >=7 | 4.0 | male single | none | 4.0 | real estate | 67.0 | none | own | 2.0 | skilled | 1.0 | yes | yes | good
0<=X<200 | 48.0 | existing paid | radio/tv | 5951.0 | <100 | 1<=X<4 | 2.0 | female div/dep/mar | none | 2.0 | real estate | 22.0 | none | own | 1.0 | skilled | 1.0 | none | yes | bad
no checking | 12.0 | critical/other existing credit | education | 2096.0 | <100 | 4<=X<7 | 2.0 | male single | none | 3.0 | real estate | 49.0 | none | own | 1.0 | unskilled resident | 2.0 | none | yes | good
... ...
  
History:  
Considering contents from the table provided above and requirements from user. Please note DO NOT CHANGE FILE AND VARIABLE NAMES THAT I HAVE SET!  
Interactions begin:

{dialogue_hist}

NOTE: Please follow my example between "Example Start" and "Example End" above to answer task with interleaving Thought, Code, Action, Result turns. Each turn contains one segment of the target code to solve the problem. 
Please generate ONLY one turn this time and wait for User to give Result based on your generated code segment, DON'T generate the whole code in a single turn! 
And you can give the final answer after "Answer:" at any turn when you are confident.

[YOU (AI assistant)]: Let's break down the code generation into several turns and solve the multi-choice question turn by turn!
################### Your Answer Starts Here: ################### 
Turn 1:
# 5 turns left to provide final answer. Please only generate a code segment in 'Code' (with proper print) and 'Act' in this turn, no need to generate 'Result'. DON'T generate the whole code in a single turn! 
Thought 1: First import all packages needed and load the dataset.
Code 1:
"""


prompt_react_code_hist_analysis_fastfood_AIR = """
Solve a question answering task with interleaving Thought, Code, Action, Results steps. Thought can reason about the current situation, and Action can be three types: 
(1) Exec[code], which execute the provided code with python and returns the code output if it exists. 
(2) Terminate[answer], which returns the answer and finishes the task.
Here is an examples.

------------------------------ Example Start: ------------------------------
The database table credit_card_risk is shown as follows: 
checking_status | duration | credit_history | purpose | credit_amount | savings_status | employment | installment_commitment | personal_status | other_parties | residence_since | property_magnitude | age | other_payment_plans | housing | existing_credits | job | num_dependents | own_telephone | foreign_worker | class
<0 | 6.0 | critical/other existing credit | radio/tv | 1169.0 | no known savings | >=7 | 4.0 | male single | none | 4.0 | real estate | 67.0 | none | own | 2.0 | skilled | 1.0 | yes | yes | good
0<=X<200 | 48.0 | existing paid | radio/tv | 5951.0 | <100 | 1<=X<4 | 2.0 | female div/dep/mar | none | 2.0 | real estate | 22.0 | none | own | 1.0 | skilled | 1.0 | none | yes | bad
no checking | 12.0 | critical/other existing credit | education | 2096.0 | <100 | 4<=X<7 | 2.0 | male single | none | 3.0 | real estate | 49.0 | none | own | 1.0 | unskilled resident | 2.0 | none | yes | good
... ...

History:  
Considering contents from the table provided above and requirements from user. Please note DO NOT CHANGE FILE AND VARIABLE NAMES THAT I HAVE SET!  
Interactions begin:  
[USER (data scientist)]: Hi, ... ...  
  
[YOU (AI assistant)]: ... ... [PREVIOUS_CODE]  

... ...

[USER (data scientist)]: What is the optimal number of clusters using the silhouette score? 

Options:  
A. 1  
B. 3 
C. 4
D. None of above 

Please firstly analysis the given pseudocode to answer question with interleaving Thought, Code, Action, Result turns. Each turn contains one segment of the target code to solve the problem. 
--- Pseudocode Starts: ---
START
  IMPORT pandas library as pd
  IMPORT silhouette_score from sklearn.metrics
  IMPORT pickle
  IMPORT matplotlib.pyplot as plt

  SET credit_customers TO pd.read_csv("credit_customers.csv")

  ... ...

  PLOT silhouette_score
  CALL plt.savefig('pred_result/Silhouette_Scores.png') 
END
--- Pseudocode Ends ---

NOTE: You have at most 5 turns to generate final results, which means you have to provide the answer at or earlier than Thought 5.

[YOU (AI assistant)]: Let's break down the code generation and solve the multi-choice question turn by turn!
################### Your Answer Starts Here: ################### 
Turn 1:
# 5 turns left to provide final answer.
Thought 1: First import all packages needed and load the dataset.
Code 1:
'''
import pandas as pd  
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score 
import matplotlib.pyplot as plt

credit_customers = pd.read_csv("credit_customers.csv")
print(credit_customers)
'''
Act 1: Exec[Code 1]
Result 1: 
checking_status | duration | credit_history | purpose | credit_amount | savings_status | employment | installment_commitment | personal_status | other_parties | residence_since | property_magnitude | age | other_payment_plans | housing | existing_credits | job | num_dependents | own_telephone | foreign_worker | class
<0 | 6.0 | critical/other existing credit | radio/tv | 1169.0 | no known savings | >=7 | 4.0 | male single | none | 4.0 | real estate | 67.0 | none | own | 2.0 | skilled | 1.0 | yes | yes | good
0<=X<200 | 48.0 | existing paid | radio/tv | 5951.0 | <100 | 1<=X<4 | 2.0 | female div/dep/mar | none | 2.0 | real estate | 22.0 | none | own | 1.0 | skilled | 1.0 | none | yes | bad
no checking | 12.0 | critical/other existing credit | education | 2096.0 | <100 | 4<=X<7 | 2.0 | male single | none | 3.0 | real estate | 49.0 | none | own | 1.0 | unskilled resident | 2.0 | none | yes | good
... ...

Turn 2:
# 4 turns left to provide final answer.
Thought 2: Calculate the silhouette scores.
Code 2:
'''
silhouette_scores = [] 
for cluster_num in range(2, 15):  # Starting from 2 because silhouette score is not defined for 1 cluster 
    kmeans = KMeans(n_clusters=cluster_num, random_state=42) 
    cluster_labels = kmeans.fit_predict(data_for_clustering_scaled) 
    silhouette_avg = silhouette_score(data_for_clustering_scaled, cluster_labels) 
    silhouette_scores.append(silhouette_avg) 

print(silhouette_scores)
'''
Act 2: Exec[Code 3]
Result 2: [0.25, 0.34, 0.45, 0.67, 0.53, 0.37]

Turn 3:
# 3 turns left to provide final answer.
Thought 3: Plot the silhouette scores.
Code 3:
'''
# Plot silhouette scores 
plt.figure(figsize=(10, 6)) 
plt.plot(range(2, 15), silhouette_scores, marker='o', linestyle='--') 
plt.title('Silhouette Scores for Different Number of Clusters') 
plt.xlabel('Number of Clusters') 
plt.ylabel('Silhouette Score') 
plt.grid(True) 
plt.savefig('pred_result/Silhouette_Scores.png') 
plt.show() 
'''
Act 3: Exec[Code 3]
Result 3: [PLOT]

Turn 4:
# 2 turns left to provide final answer.
Thought 4: Analyze all results I have.
Analysis: According to the plot, when the number of clusters equals to 4 lead to highest Silhouette Scores, so we choose C.
Act 4: Terminate[C]
Answer: C
################### Your Answer Ends Here: ################### 
------------------------------ Example End ------------------------------ 
  
The database table atp_tennis is shown as follows:   
restaurant | item | calories | cal_fat | total_fat | sat_fat | trans_fat | cholesterol | sodium | total_carb | fiber | sugar | protein | vit_a | vit_c | calcium | salad
Mcdonalds | Artisan Grilled Chicken Sandwich | 380 | 60 | 7 | 2 | 0 | 95 | 1110 | 44 | 3 | 11 | 37 | 4 | 20 | 20 | Other
Mcdonalds | Single Bacon Smokehouse Burger | 840 | 410 | 45 | 17 | 1.5 | 130 | 1580 | 62 | 2 | 18 | 46 | 6 | 20 | 20 | Other
Mcdonalds | Double Bacon Smokehouse Burger | 1130 | 600 | 67 | 27 | 3 | 220 | 1920 | 63 | 3 | 18 | 70 | 10 | 20 | 50 | Other 
... ...

Please note that the pandas package version is 2.0.3 and matplotlib package version is 3.7.4 in all code here.

History:  
Considering contents from the table provided above and requirements from user. Please note DO NOT CHANGE FILE AND VARIABLE NAMES THAT I HAVE SET!  
Interactions begin:

{dialogue_hist}

Please firstly analysis the given pseudocode to answer question and follow my example between "Example Start" and "Example End" above to answer question with interleaving Thought, Code, Action, Result turns. Each turn contains one segment of the target code to solve the problem. 
--- Pseudocode Starts: ---
{pseudo_code}
--- Pseudocode Ends ---

NOTE: Please generate ONLY one turn this time and wait for User to give Result based on your generated code segment, DON'T generate the whole code in a single turn! 
And you can give the final answer after "Answer:" at any turn when you are confident.

[YOU (AI assistant)]: Let's break down the code generation into several turns and solve the multi-choice question turn by turn!
################### Your Answer Starts Here: ################### 
Turn 1:
# 5 turns left to provide final answer. Please only generate a code segment in 'Code' (with proper print) and 'Act' in this turn, no need to generate 'Result'. DON'T generate the whole code in a single turn! 
Thought 1: First import all packages needed and load the dataset.
Code 1: Load the data from 'fastfood.csv'.
'''
"""


prompt_react_code_hist_analysis_ATP_AIR = """
Solve a question answering task with interleaving Thought, Code, Action, Results steps. Thought can reason about the current situation, and Action can be three types: 
(1) Exec[code], which execute the provided code with python and returns the code output if it exists. 
(2) Terminate[answer], which returns the answer and finishes the task.
Here is an examples.

------------------------------ Example Start: ------------------------------
The database table credit_card_risk is shown as follows: 
checking_status | duration | credit_history | purpose | credit_amount | savings_status | employment | installment_commitment | personal_status | other_parties | residence_since | property_magnitude | age | other_payment_plans | housing | existing_credits | job | num_dependents | own_telephone | foreign_worker | class
<0 | 6.0 | critical/other existing credit | radio/tv | 1169.0 | no known savings | >=7 | 4.0 | male single | none | 4.0 | real estate | 67.0 | none | own | 2.0 | skilled | 1.0 | yes | yes | good
0<=X<200 | 48.0 | existing paid | radio/tv | 5951.0 | <100 | 1<=X<4 | 2.0 | female div/dep/mar | none | 2.0 | real estate | 22.0 | none | own | 1.0 | skilled | 1.0 | none | yes | bad
no checking | 12.0 | critical/other existing credit | education | 2096.0 | <100 | 4<=X<7 | 2.0 | male single | none | 3.0 | real estate | 49.0 | none | own | 1.0 | unskilled resident | 2.0 | none | yes | good
... ...

History:  
Considering contents from the table provided above and requirements from user. Please note DO NOT CHANGE FILE AND VARIABLE NAMES THAT I HAVE SET!  
Interactions begin:  
[USER (data scientist)]: Hi, ... ...  
  
[YOU (AI assistant)]: ... ... [PREVIOUS_CODE]  

... ...

[USER (data scientist)]: What is the optimal number of clusters using the silhouette score? 

Options:  
A. 1  
B. 3 
C. 4
D. None of above 

Please firstly analysis the given pseudocode to answer question with interleaving Thought, Code, Action, Result turns. Each turn contains one segment of the target code to solve the problem. 
--- Pseudocode Starts: ---
START
  IMPORT pandas library as pd
  IMPORT silhouette_score from sklearn.metrics
  IMPORT pickle
  IMPORT matplotlib.pyplot as plt

  SET credit_customers TO pd.read_csv("credit_customers.csv")

  ... ...

  PLOT silhouette_score
  CALL plt.savefig('pred_result/Silhouette_Scores.png') 
END
--- Pseudocode Ends ---

NOTE: You have at most 5 turns to generate final results, which means you have to provide the answer at or earlier than Thought 5.

[YOU (AI assistant)]: Let's break down the code generation and solve the multi-choice question turn by turn!
################### Your Answer Starts Here: ################### 
Turn 1:
# 5 turns left to provide final answer.
Thought 1: First import all packages needed and load the dataset.
Code 1:
'''
import pandas as pd  
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score 
import matplotlib.pyplot as plt

credit_customers = pd.read_csv("credit_customers.csv")
print(credit_customers)
'''
Act 1: Exec[Code 1]
Result 1: 
checking_status | duration | credit_history | purpose | credit_amount | savings_status | employment | installment_commitment | personal_status | other_parties | residence_since | property_magnitude | age | other_payment_plans | housing | existing_credits | job | num_dependents | own_telephone | foreign_worker | class
<0 | 6.0 | critical/other existing credit | radio/tv | 1169.0 | no known savings | >=7 | 4.0 | male single | none | 4.0 | real estate | 67.0 | none | own | 2.0 | skilled | 1.0 | yes | yes | good
0<=X<200 | 48.0 | existing paid | radio/tv | 5951.0 | <100 | 1<=X<4 | 2.0 | female div/dep/mar | none | 2.0 | real estate | 22.0 | none | own | 1.0 | skilled | 1.0 | none | yes | bad
no checking | 12.0 | critical/other existing credit | education | 2096.0 | <100 | 4<=X<7 | 2.0 | male single | none | 3.0 | real estate | 49.0 | none | own | 1.0 | unskilled resident | 2.0 | none | yes | good
... ...

Turn 2:
# 4 turns left to provide final answer.
Thought 2: Calculate the silhouette scores.
Code 2:
'''
silhouette_scores = [] 
for cluster_num in range(2, 15):  # Starting from 2 because silhouette score is not defined for 1 cluster 
    kmeans = KMeans(n_clusters=cluster_num, random_state=42) 
    cluster_labels = kmeans.fit_predict(data_for_clustering_scaled) 
    silhouette_avg = silhouette_score(data_for_clustering_scaled, cluster_labels) 
    silhouette_scores.append(silhouette_avg) 

print(silhouette_scores)
'''
Act 2: Exec[Code 3]
Result 2: [0.25, 0.34, 0.45, 0.67, 0.53, 0.37]

Turn 3:
# 3 turns left to provide final answer.
Thought 3: Plot the silhouette scores.
Code 3:
'''
# Plot silhouette scores 
plt.figure(figsize=(10, 6)) 
plt.plot(range(2, 15), silhouette_scores, marker='o', linestyle='--') 
plt.title('Silhouette Scores for Different Number of Clusters') 
plt.xlabel('Number of Clusters') 
plt.ylabel('Silhouette Score') 
plt.grid(True) 
plt.savefig('pred_result/Silhouette_Scores.png') 
plt.show() 
'''
Act 3: Exec[Code 3]
Result 3: [PLOT]

Turn 4:
# 2 turns left to provide final answer.
Thought 4: Analyze all results I have.
Analysis: According to the plot, when the number of clusters equals to 4 lead to highest Silhouette Scores, so we choose C.
Act 4: Terminate[C]
Answer: C
################### Your Answer Ends Here: ################### 
------------------------------ Example End ------------------------------ 
  
The database table atp_tennis is shown as follows:   
Tournament | Date | Series | Court | Surface | Round | Best of | Player_1 | Player_2 | Winner | Rank_1 | Rank_2 | Pts_1 | Pts_2 | Odd_1 | Odd_2 | score  
Brisbane International | 2012-12-31 | ATP250 | Outdoor | Hard | 1st Round | 3 | Mayer F. | Giraldo S. | Mayer F. | 28 | 57 | 1215 | 778 | 1.36 | 3.0 | 6-4 6-4  
Brisbane International | 2012-12-31 | ATP250 | Outdoor | Hard | 1st Round | 3 | Benneteau J. | Nieminen J. | Nieminen J. | 35 | 41 | 1075 | 927 | 2.2 | 1.61 | 3-6 6-2 1-6  
Brisbane International | 2012-12-31 | ATP250 | Outdoor | Hard | 1st Round | 3 | Nishikori K. | Matosevic M. | Nishikori K. | 19 | 49 | 1830 | 845 | 1.25 | 3.75 | 7-5 6-2  
... ...

Please note that the pandas package version is 2.0.3 and matplotlib package version is 3.7.4 in all code here.

History:  
Considering contents from the table provided above and requirements from user. Please note DO NOT CHANGE FILE AND VARIABLE NAMES THAT I HAVE SET!  
Interactions begin:

{dialogue_hist}

Please firstly analysis the given pseudocode to answer question and follow my example between "Example Start" and "Example End" above to answer question with interleaving Thought, Code, Action, Result turns. Each turn contains one segment of the target code to solve the problem. 
--- Pseudocode Starts: ---
{pseudo_code}
--- Pseudocode Ends ---

NOTE: Please generate ONLY one turn this time and wait for User to give Result based on your generated code segment, DON'T generate the whole code in a single turn! 
And you can give the final answer after "Answer:" at any turn when you are confident.

[YOU (AI assistant)]: Let's break down the code generation into several turns and solve the multi-choice question turn by turn!
################### Your Answer Starts Here: ################### 
Turn 1:
# 5 turns left to provide final answer. Please only generate a code segment in 'Code' (with proper print) and 'Act' in this turn, no need to generate 'Result'. DON'T generate the whole code in a single turn! 
Thought 1: First import all packages needed and load the dataset.
Code 1:
"""

prompt_react_code_hist_analysis_credit_AIR = """
Solve a question answering task with interleaving Thought, Code, Action, Results steps. Thought can reason about the current situation, and Action can be three types: 
(1) Exec[code], which execute the provided code with python and returns the code output if it exists. 
(2) Terminate[answer], which returns the answer and finishes the task.
Here is an examples.

------------------------------ Example Start: ------------------------------
The database table atp_tennis is shown as follows: 
Tournament | Date | Series | Court | Surface | Round | Best of | Player_1 | Player_2 | Winner | Rank_1 | Rank_2 | Pts_1 | Pts_2 | Odd_1 | Odd_2 | score  
Brisbane International | 2012-12-31 | ATP250 | Outdoor | Hard | 1st Round | 3 | Mayer F. | Giraldo S. | Mayer F. | 28 | 57 | 1215 | 778 | 1.36 | 3.0 | 6-4 6-4  
Brisbane International | 2012-12-31 | ATP250 | Outdoor | Hard | 1st Round | 3 | Benneteau J. | Nieminen J. | Nieminen J. | 35 | 41 | 1075 | 927 | 2.2 | 1.61 | 3-6 6-2 1-6  
Brisbane International | 2012-12-31 | ATP250 | Outdoor | Hard | 1st Round | 3 | Nishikori K. | Matosevic M. | Nishikori K. | 19 | 49 | 1830 | 845 | 1.25 | 3.75 | 7-5 6-2  
... ...

Please note that the pandas package version is 2.0.3 and matplotlib package version is 3.7.4 in all code here.

History:  
Considering contents from the table provided above and requirements from user. Please note DO NOT CHANGE FILE AND VARIABLE NAMES THAT I HAVE SET!  
Interactions begin:  
[USER (data scientist)]: Hi, ... ...  
  
[YOU (AI assistant)]: ... ... [PREVIOUS_CODE]  

... ...

[USER (data scientist)]: Hi, let's start by assessing the data quality of the atp_tennis dataset. To achieve this, we'll create a Data Quality Report DataFrame that covers two main aspects: a list of columns along with the number of missing values in each column, A list of unique values for each categorical column, such as Tournament, Series, Court, Surface, and Round. And please answer the following question: How many unique value of Court?

Options:  
A. 1  
B. 2  
C. 3  
D. None of above 


Please firstly analysis the given pseudocode to answer question with interleaving Thought, Code, Action, Result turns. Each turn contains one segment of the target code to solve the problem. 
--- Pseudocode Starts :---
START
  IMPORT pandas library as pd
  IMPORT pickle

  SET credit_customers TO pd.read_csv("atp_tennis.csv")

  ... ...

  PRINT data_quality_report
  CALL pickle.dump(data_quality_report,open("./pred_result/data_quality_report.pkl","wb"))
END
--- Pseudocode Ends ---

NOTE: You have at most 5 turns to generate final results, which means you have to provide the answer at or earlier than Thought 5.

[YOU (AI assistant)]: Let's break down the code generation and solve the multi-choice question turn by turn!
################### Your Answer Starts Here: ################### 
Turn 1:
# 5 turns left to provide final answer.
Thought 1: First import all packages needed and load the dataset.
Code 1:
'''
import pandas as pd
import pickle

atp_tennis = pd.read_csv('atp_tennis.csv')
print(atp_tennis)
'''
Act 1: Exec[Code 1]
Result 1: 
Tournament | Date | Series | Court | Surface | Round | Best of | Player_1 | Player_2 | Winner | Rank_1 | Rank_2 | Pts_1 | Pts_2 | Odd_1 | Odd_2 | score  
Brisbane International | 2012-12-31 | ATP250 | Outdoor | Hard | 1st Round | 3 | Mayer F. | Giraldo S. | Mayer F. | 28 | 57 | 1215 | 778 | 1.36 | 3.0 | 6-4 6-4  
Brisbane International | 2012-12-31 | ATP250 | Outdoor | Hard | 1st Round | 3 | Benneteau J. | Nieminen J. | Nieminen J. | 35 | 41 | 1075 | 927 | 2.2 | 1.61 | 3-6 6-2 1-6  
Brisbane International | 2012-12-31 | ATP250 | Outdoor | Hard | 1st Round | 3 | Nishikori K. | Matosevic M. | Nishikori K. | 19 | 49 | 1830 | 845 | 1.25 | 3.75 | 7-5 6-2  
... ...

Turn 2:
# 4 turns left to provide final answer.
Thought 2: Then create a Data Quality Report Dataframe for the atp_tennis dataset.
Code 2:
'''
missing_values = atp_tennis.isnull().sum()
unique_values = atp_tennis[['Tournament', 'Series', 'Court', 'Surface', 'Round']].nunique()

data_quality_report = pd.DataFrame('Missing Values': missing_values, 'Unique Values': unique_values)
print(data_quality_report)
'''
Act 2: Exec[Code 3]
Result 2:  
Missing Values | Unique Values
Best of | 0 | NaN
Court | 0 | 2.0
Date | 0 | NaN
Odd_1 | 0 | NaN
... ...

Turn 3:
# 3 turns left to provide final answer.
Thought 3: Print the data quality report and save it to my computer.
Code 3:
'''
print(data_quality_report)
pickle.dump(data_quality_report,open("./pred_result/data_quality_report.pkl","wb"))
'''
Act 3: Exec[Code 3]
Result 3: 
Missing Values | Unique Values
Best of | 0 | NaN
Court | 0 | 2.0
Date | 0 | NaN
Odd_1 | 0 | NaN
... ...

Turn 4:
# 2 turns left to provide final answer.
Thought 4: Analyze all results I have.
Analysis: According to the data quality report, the unique value of Court is 2.0, so we choose B.
Act 4: Terminate[B]
Answer: B
################### Your Answer Ends Here: ################### 
------------------------------ Example End ------------------------------ 
  
The database table credit_card_risk is shown as follows:   
checking_status | duration | credit_history | purpose | credit_amount | savings_status | employment | installment_commitment | personal_status | other_parties | residence_since | property_magnitude | age | other_payment_plans | housing | existing_credits | job | num_dependents | own_telephone | foreign_worker | class
<0 | 6.0 | critical/other existing credit | radio/tv | 1169.0 | no known savings | >=7 | 4.0 | male single | none | 4.0 | real estate | 67.0 | none | own | 2.0 | skilled | 1.0 | yes | yes | good
0<=X<200 | 48.0 | existing paid | radio/tv | 5951.0 | <100 | 1<=X<4 | 2.0 | female div/dep/mar | none | 2.0 | real estate | 22.0 | none | own | 1.0 | skilled | 1.0 | none | yes | bad
no checking | 12.0 | critical/other existing credit | education | 2096.0 | <100 | 4<=X<7 | 2.0 | male single | none | 3.0 | real estate | 49.0 | none | own | 1.0 | unskilled resident | 2.0 | none | yes | good
... ...
  
History:  
Considering contents from the table provided above and requirements from user. Please note DO NOT CHANGE FILE AND VARIABLE NAMES THAT I HAVE SET!  
Interactions begin:

{dialogue_hist}

Please firstly analysis the given pseudocode to answer question and follow my example between "Example Start" and "Example End" above to answer question with interleaving Thought, Code, Action, Result turns. Each turn contains one segment of the target code to solve the problem. 
--- Pseudocode Starts: ---
{pseudo_code}
--- Pseudocode Ends ---

NOTE: Please generate ONLY one turn this time and wait for User to give Result based on your generated code segment, DON'T generate the whole code in a single turn! 
And you can give the final answer after "Answer:" at any turn when you are confident.

[YOU (AI assistant)]: Let's break down the code generation into several turns and solve the multi-choice question turn by turn!
################### Your Answer Starts Here: ################### 
Turn 1:
# 5 turns left to provide final answer. Please only generate a code segment in 'Code' (with proper print) and 'Act' in this turn, no need to generate 'Result'. DON'T generate the whole code in a single turn! 
Thought 1: First import all packages needed and load the dataset.
Code 1:
"""

prompt_AIR_teacher_user = """
As a data science professor, your mission is to guide your students in creating Python code to tackle data science problems, using the latest successful interaction between the user and AI as a reference. You need to firstly analyze the Python code generated by AI in the latest User-AI interaction above. Then you need to develop a pseudocode that will guide students to write the correct python code as provided in my latest query. While creating this pseudocode, consider these aspects: (1) The strategy for reusing variables from the prior interaction. Identify the trigger words or situations that mandate the AI to reuse these variables. (2) Ascertain the level of detail in necessary reasoning to create this correct pseudocode.    
Please give me a STANDARD pseudocode format between <pseudocode>...</pseudocode> with proper hierarchy, neither bullet points nor python format! For exaqmple: 
################### Pseudocode Example ###################
<pseudocode>
// Load the data from ...
LOAD DATA FROM ...

// Iterate over the filtered ...
FOR EACH ... IN ...
    DETERMINE ... AND ... FROM ...
    INCREMENT ...

// Output the ...
DISPLAY ... 
</pseudocode>  
"""

prompt_AIR_teacher_AI = """
<pseudocode>
"""

prompt_AIR_student_user = """
{dialogue_hist}

################### Example Starts: ###################
{user_query}

[YOU (AI assistant)]:
{pseudo_code}
################### Example Ends ###################

As a data science professor, your mission is to guide your students in creating Python code to tackle data science problems, using the latest successful interaction between the user and AI as a reference. You need to firstly analyze the Python code generated by AI in the latest User-AI interaction above. Then you need to develop a pseudocode that will guide students to write the correct python code as provided in my latest query. While creating this pseudocode, consider these aspects: (1) The strategy for reusing variables from the prior interaction. Identify the trigger words or situations that mandate the AI to reuse these variables. (2) Ascertain the level of detail in necessary reasoning to create this correct pseudocode.    
Please give me a STANDARD pseudocode format between <pseudocode> YOUR-PSEUDOCODE </pseudocode> with proper hierarchy, neither bullet points nor python format!   
"""

prompt_AIR_student_AI = """
<pseudocode>
"""


prompt_AIR_02_curr = """
{head_info}
--- Interaction History: ---
{dialogue_hist}

As a data science professor, your mission is to guide your students in creating Python code to tackle data science problems in "New Query", by analyzing the given dialogue History and pseudocode and Python code generated in the successful example below. Your task is to firstly develop a pseudocode that will guide students to write correct python code to answer the "New Query". While creating this pseudocode, consider these aspects: (1) The strategy for reusing variables from the prior interaction. Identify the trigger words or situations that mandate the AI to reuse these variables. (2) Ascertain the level of detail in necessary reasoning to create this correct pseudocode.

################### Example Starts: ###################
{user_query}

[YOU (AI assistant)]:
# Step 1: pseudocode:
{pseudo_code}

# Step 2: Python code:
{python_code}
################### Example Ends ###################

--- New Query: ---
{new_query_user}
According to the example, let's generate the pseudocode and then analyze the generated pseudocode to generate python code.

{new_query_AI}
# Step 1: pseudocode:
<pseudocode> 
"""

clarification_user_01 = "You are talking with your user and your goal is to address the user's questions. It's a very serious task that you have to make sure all requriements from the user can be fulfilled without any uncertainty. Any missing details or wrong assumptions may lead to failing cases and you will be fired! Now, you have chance to ask User at most ONE question between <question>YOUR-QUESTION</question> if you are uncertain about my latest query. Otherwise, if you are very certain, you can directly answer user query. "
clarification_user_02 = "{clarification} Now you have to generate Python code based on the code snippet to answer my latest query. And you must keep all code comments in the code snippet unchanged."
clarification_AI_01 = """
According to the ground truth code for my latest query below:
--- Ground Truth Code: ---
'''
{ref_code}
'''

directly answer the latest question from AI assistant between <answer>YOUR-ANSWER</answer> in one normal sentence. 
"""


plotQA_credit = """
There is the dataset: credit_customers containing the following columns: ['checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount', 'savings_status', 'employment', 'installment_commitment', 'personal_status', 'other_parties', 'residence_since', 'property_magnitude', 'age', 'other_payment_plans', 'housing', 'existing_credits', 'job', 'num_dependents', 'own_telephone', 'foreign_worker', 'class']. 
--- The description for each column this dataset is: 
Checking_status: Status of the applicant's checking account ("no checking": No checking account, "<0": Overdrawn or negative balance, "0<=X<200": Low positive balance, e.g., between $0 and $200, and ">=200": Healthy balance)
Duration: Duration of the loan or credit term (measured in months)
Credit_history: Applicant's past handling of credit ("no credits/all paid": No prior credit or all previous credits paid off, "existing paid": Current credits are being paid off diligently, "critical/other existing credit": Past or current credits with issues, like late payments, "delayed previously": History of delayed payments)
Purpose: Reason for the loan or credit application (purchasing a car, financing education, buying electronics ...etc)
Credit_amount: Amount of money requested in the loan or credit application
Savings_status: Amount of savings the applicant has ("no known savings": No savings account or negligible savings, "<100": Savings less than $100, "100<=X<500": Savings between $100 and $500, "500<=X<1000": Savings between $500 and $1000, and ">=1000": Savings of $1000 or more)
Employment: Applicant's employment status or the length of their current employment ("unemployed": No current employment, "<1": Employed for less than a year, "1<=X<4": Employed between 1 and 4 years, ">=7": Employed for 7 years or more)
Installment_commitment: Portion of the applicant's disposable income that is allocated for loan repayments (represented as a percentage or a fixed numerical value)
Personal_status: Details of the applicant's personal and demographic information ("male single": A single male, "female div/dep/mar": A divorced, separated, or married female, "male div/sep": A divorced or separated male, and "male mar/wid": A married or widowed male)
Other_parties: Any third parties involved in the credit agreement ("none": No other parties involved, "guarantor": A guarantor is present who guarantees repayment, and "co-applicant": Another individual is co-signing the loan)
Residence_since: Length of time the applicant has been living at their current residence (measured in years)
Property_magnitude: Type or value of property the applicant owns, serving as potential collateral ("real estate": Owning a house or land, "life insurance": Having a life insurance policy with a cash value, "car": Owning a car, and "no known property": No property that can be used as collateral)
Age: Age of the applicant
Other_payment_plans: Any other ongoing loans or financial obligations the applicant has outside of the loan in question ("none": No other payment plans or ongoing loans, "bank": The applicant has other loans or payment plans with a bank, and "stores": The applicant has credit obligations with stores or retail chains)
Housing: Living situation of the applicant ("own": The applicant owns their home, "rent": The applicant is renting their living space, and "for free": The applicant lives in a space without paying rent (e.g., with family))
Existing_credits: Number of current credit lines or loans that the applicant already has (represented numerically)
Job: Applicant's employment type ("skilled": Professionally skilled job or a white-collar occupation, "unskilled resident": Unskilled job and residing in the country, "high qualif/self emp/mgmt": Highly qualified, self-employed, or in management positions, and "unemp/unskilled non res": Unemployed, lacks professional or skilled training, and is not a resident of the country)
Num_dependents: Number of individuals financially dependent on the applicant
Own_telephone: Whether the applicant has a telephone
Foreign_worker: Whether the applicant is a foreign worker
Class: Target variable for a predictive model, indicating the risk category of the credit application ("good": Low risk, indicating a high likelihood of timely repayment, or "bad": High risk, suggesting potential difficulties in repaying the loan)
---

You are a data scientist with an impressive array of skills including data analysis, statistics, machine learning, and proficiency in Pandas. The plot mentioned in the question below has been transformed into a Filtered Dataframe, and your task is to answer the following question from User based on this Filtered Dataframe below.

--- Filtered Dataframe: ---
{table}

--- Filtered Dataframe ---

{prompt_curr}
"""

plotQA_ATP = """
There is the dataset: atp_tennis containing the following columns: ['Tournament', 'Date', 'Series', 'Court', 'Surface', 'Round', 'Best of', 'Player_1', 'Player_2', 'Winner', 'Rank_1', 'Rank_2', 'Pts_1', 'Pts_2', 'Odd_1', 'Odd_2', 'score'].  
--- The description for each column this dataset is:
Tournament: Name of the tennis tournament (Brisbane International, Chennai Open, Qatar Exxon Mobil Open ...etc)
Date: Date the match was played (year-month-day)
Series: Category or level of the tennis tournament (ATP250, ATP500, Masters1000 and Grand Slams offer 250, 500, 1000, and 2000 ranking points to the winner seperately.)
Court: Place the match was held (Indoors or Outdoors)
Surface: Type of court surface (Hard, Grass and Clay)
Round: Stage of the tournament (1st Round, 2nd Round, Quarterfinals, Semifinal and The Final)
Best of: Tourament systems ("best of 3" or "best of 5")
Player_1: Name of the first competitor in each match
Player_2: Name of the second competitor in each match
Winner: Name of the Player_1 or Player_2 who won the match 
Rank_1: World rankings of the Player_1 at the time of the match
Rank_2: World rankings of the Player_2 at the time of the match
Pts_1: Ranking points held by Player_1 before the match (accumulated over the season and higher points typically indicate a more successful player)
Pts_2: Ranking points held by Player_2 before the match
Odd_1: Betting odds for Player_1 (reflecting their perceived chances of winning)
Odd_2: Betting odds for Player_2 
Score: Final score of the match ("score of Player_1" - "score of Player_2")
---

You are a data scientist with an impressive array of skills including data analysis, statistics, machine learning, and proficiency in Pandas. The plot mentioned in the question below has been transformed into a Filtered Dataframe, and your task is to answer the following question from User based on this Filtered Dataframe below.

--- Filtered Dataframe: ---
{table}

--- Filtered Dataframe ---

{prompt_curr}
"""


prompt_AIR_02_curr_new = """
{head_info}
=== Interaction History: ===
{dialogue_hist}

Your task is to analyze the following pseudocode and write python code to fill the code snippet of "New Query" below.
--- Pseudocode Starts: ---
{pseudo_new}
--- Pseudocode Ends ---

=== New Query: ===
{new_query}
My template of code snippet is:
--- Code Snippet Starts: ---
{code_prompt}
--- Code Snippet Ends ---

{code_start}
"""

prompt_AIR_02_curr_pseudo = """
{head_info}
--- Interaction History: ---
{dialogue_hist}

As a data science professor, your mission is to guide your students in creating Python code to tackle data science problems in "New Query". To achieve this, firstly write good pseudocode by learning the successful history. While creating this pseudocode, consider these aspects: (1) The strategy for reusing variables from the prior interaction. Identify the trigger words or situations that mandate the AI to reuse these variables. (2) Ascertain the level of detail in necessary reasoning.

################### Example Starts: ###################
{user_query}

[YOU (AI assistant)]:
<pseudocode>
{pseudo_code}
</pseudocode>
################### Example Ends ###################

--- New Query: ---
{new_query}

NOTE: According to the example, please only write good pseudocode to be the guide to answer the "New Query". The Pseudocode should be a whole complete pseudocode which starts with "// Load the dataset" and ends with "// Save results".

[YOU (AI assistant)]:
<pseudocode>
// Load the dataset 
"""