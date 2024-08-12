import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Calculates the BMI of the person
def calcBMI(df):
    return (df['weight']/pow(df['height']/100, 2))

df = pd.read_csv('./medical_examination.csv')

# Assign 1 if person is overweight, 0 if not
df['overweight'] = np.where(calcBMI(df) > 25, 1,0)

# Normalize 'gluc' and 'cholesterol' column
df['cholesterol'] = np.where(df['cholesterol'] == 1, 0, 1)
df['gluc'] = np.where(df['gluc'] == 1, 0,1)

def draw_cat_plot():
    # Create a dataframe for the catplot
    df_cat =df.melt(id_vars='cardio',
                    value_vars=['active',
                                'alco',
                                'cholesterol',
                                'gluc',
                                'overweight',
                                'smoke'])

    # Split it by cardio and show the counts of each feature
    df_cat = df_cat.groupby(['cardio',
                             'variable',
                             'value'])\
                    .size()\
                    .reset_index(name='total')
    
    # Draw the plot
    graph = sns.catplot(data=df_cat,
                        x='variable',
                        y='total',
                        col='cardio',
                        hue='value',
                        kind='bar')

    # Get the figure of the plot
    fig = graph.figure

    # Save the figure
    fig.savefig('catplot.png')
    return fig

def draw_heat_map():
    # Remove rows where diastolic blood pressure (ap_lo) is higher than systolic blood pressure (ap_hi),
    # as this represents incorrect or erroneous data.
    valid_bp_range = (df['ap_lo'] <= df['ap_hi'])

    # Let's remove the outliers
    height_lower_bound = df['height'].quantile(0.025)
    height_upper_bound = df['height'].quantile(0.975)
    height_outliers_removed = (df['height'] >= height_lower_bound)\
                                & (df['height'] <= height_upper_bound)

    weight_lower_bound = df['weight'].quantile(0.025)
    weight_upper_bound = df['weight'].quantile(0.975)
    weight_outliers_removed = (df['weight'] >= weight_lower_bound)\
                                & (df['weight'] <= weight_upper_bound)

   # Clean data 
    df_heat = df[valid_bp_range & weight_outliers_removed & height_outliers_removed]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(12,8))

    sns.heatmap(data=corr,
                mask=mask,
                annot=True,
                fmt='.1f')

    # Export figure as an image
    fig.savefig('heatmap.png')
    return fig
