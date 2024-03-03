#import of packages 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from scipy.interpolate import PchipInterpolator

def clean_data(raw,print_head=False):
    """
    Function to clean the raw data and calculate the percentages and cumulative percentages of AGI
    
    Args:
    raw: pandas DataFrame, raw data
    print_head: bool, print the head of the cleaned data
    
    Returns:
    data: pandas DataFrame, cleaned data
    """
    data = raw.copy()

    # transpose the data such that first row is made into a column added to the data
    all_returns = data.iloc[:2]

    # drop first column, subtracting returns with no AGI the and renaming columns
    all_returns = all_returns.drop(labels='Size of adjusted gross income', axis=1)
    all_returns = (all_returns.iloc[0]-all_returns.iloc[1])
    all_returns = all_returns.to_frame().T
    all_returns.columns = ['Total number of returns', 'Total adjusted gross income less deficit']

    # dropping first 2 rows with totals and no AGI
    data = data.drop(data.index[:2])

    # adding a 0 row to the data as the first row and resetting the index
    data = pd.concat([pd.DataFrame({'Size of adjusted gross income': ['0'], 'Number of returns': [0], 'Adjusted gross income less deficit': [0]}), data], ignore_index=True)
    data = data.reset_index(drop=True)

    # joining the data
    data = data.join(all_returns, how='cross')

    # calculating the percentage of total returns
    data['Percentage of total returns'] = data['Number of returns'] / data['Total number of returns']
    data['Percentage of total AGI'] = data['Adjusted gross income less deficit'] / data['Total adjusted gross income less deficit']

    data['Cumsum percentage of total returns'] = np.cumsum(data['Percentage of total returns'])
    data['Cumsum percentage of total AGI'] = np.cumsum(data['Percentage of total AGI'])

    if print_head:
        display(data.head())

    return data


def plot_lorentz_curve(data, shares=None, title='Lorentz Curve', data_label='Lorentz Curve', compare=None, compare_label='Lorentz Curve', pchip=False, save=None):
    """
    Function to plot the lorentz curve
    
    Args:
    data: pandas DataFrame, data to plot
    shares: list, used if specific income shares should be interpolated
    title: str, title of the plot
    data_label: str, label of the data
    compare: pandas DataFrame, data to compare
    compare_label: str, label of the compare data
    pchip: bool, if True, a plot of the data interpolated with PCHIP is added for comparison
    save: str, if not None, save the plot to the specified path
    
    Returns: None"""
    
    plt.figure(figsize=(9,8))

    plt.plot(np.linspace(0.0,1.0,len(data)), np.linspace(0.0,1.0,len(data)), label='Diagonal', marker='', color='black')
        
    if shares is None:
        plt.grid(True)
        plt.plot(data['Cumsum percentage of total returns'], data['Cumsum percentage of total AGI'], label=data_label, marker='.')
        if compare is not None:
            plt.plot(compare['Cumsum percentage of total returns'], compare['Cumsum percentage of total AGI'], label=compare_label, marker='.')
    else:
        plt.plot(data['Cumsum percentage of total returns'], data['Cumsum percentage of total AGI'], label=data_label)
        if compare is not None:
            plt.plot(compare['Cumsum percentage of total returns'], compare['Cumsum percentage of total AGI'], label=compare_label)
        for share in shares:
            # interpolate the top share of total AGI
            y = 1-np.interp(share, data['Cumsum percentage of total returns'], data['Cumsum percentage of total AGI'])
            print(f'Interpolated value for top {((1-share)*100):.2f}% share of total AGI:', y)

            # plot interpolated values on the lorentz curve
            plt.plot(share, 1-y, 'o', label=f'{((1-share)*100):.0f}% share of total AGI')

            # create dotted lines from interpolated values axis
            plt.axhline(1-y, color='grey', linestyle='dotted')
            plt.axvline(share, color='grey', linestyle='dotted')

            # write intersection of dotted lines with axis as text at the axis
            plt.text(share, 0, f'{share*100}%', verticalalignment='bottom', horizontalalignment='right')

            # adjust text position based on value of 1-y
            if 1-y > 0.8:  # adjust threshold as needed
                plt.text(0.41, 1-y+0.015, '{:.2f}%'.format((1-y)*100), verticalalignment='center', horizontalalignment='right')
            else:
                plt.text(0.04, 1-y+0.015, '{:.2f}%'.format((1-y)*100), verticalalignment='center', horizontalalignment='right')
    
    if pchip:
        pchip = PchipInterpolator(data['Cumsum percentage of total returns'], data['Cumsum percentage of total AGI'])
        x_pchip = np.linspace(0.0, 1.00, 1000)
        y_pchip = pchip(x_pchip)
        data_label_pchip = f'{data_label} (PCHIP Interpolated)'
        plt.plot(x_pchip, y_pchip, label=data_label_pchip)

    # set axis to percentage, legend and title
    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: '{:.0f}%'.format(x*100)))
    plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: '{:.0f}%'.format(x*100)))
    plt.title(title)
    plt.xlabel('Filers ranked from poorest to richest')
    plt.ylabel('Share of total AGI')
    plt.legend(loc='upper left')

    # saving the plot
    if save is not None:
        plt.savefig(save)

    plt.show()

def gini_coefficient(data, print_gini=False):
        """
        Function to calculate the Gini coefficient
        
        Args:
        data: pandas DataFrame, data to calculate the Gini coefficient for
        print_gini: bool, if True, print the Gini coefficient
        
        Returns:
        gini: float, Gini coefficient"""

        # calculate the area under between the diagonal and the lorentz curve
        area = (np.trapz(np.linspace(0.0,1.0,len(data)), np.linspace(0.0,1.0,len(data))) # area of the triangle
                - np.trapz(data['Cumsum percentage of total AGI'], data['Cumsum percentage of total returns'])) # area under the lorentz curve
        
        # calculate the Gini coefficient
        gini = area/np.trapz(np.linspace(0.0,1.0,len(data)), np.linspace(0.0,1.0,len(data)))

        if print_gini:
                print('Gini coefficient:', gini)
        
        return gini