import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class ExploratoryDataAnalysis:
    def __init__(self, df):
        self.df = df
        plt.style.use('seaborn')
    
    def basic_info(self):
        """Display basic information about the dataset"""
        print('\nBasic Dataset Information:')
        print('=' * 50)
        print(f'Number of Rows: {self.df.shape[0]}')
        print(f'Number of Columns: {self.df.shape[1]}')
        print('\nColumns:', self.df.columns.tolist())
        print('\nData Types:')
        print(self.df.dtypes)
        
        # Missing values analysis
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print('\nMissing Values:')
            print(missing[missing > 0])
            print('\nMissing Values Percentage:')
            print((missing[missing > 0] / len(self.df) * 100).round(2))
    
    def numerical_analysis(self):
        """Analyze numerical columns"""
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numerical_cols) > 0:
            print('\nNumerical Analysis:')
            print('=' * 50)
            
            # Descriptive statistics
            stats_df = self.df[numerical_cols].describe()
            print('\nDescriptive Statistics:')
            print(stats_df)
            
            # Distribution plots
            for col in numerical_cols:
                plt.figure(figsize=(12, 6))
                
                # Histogram with KDE
                plt.subplot(1, 2, 1)
                sns.histplot(data=self.df, x=col, kde=True)
                plt.title(f'Distribution of {col}')
                
                # Box plot
                plt.subplot(1, 2, 2)
                sns.boxplot(y=self.df[col])
                plt.title(f'Box Plot of {col}')
                
                plt.tight_layout()
                plt.show()
                
                # Skewness and Kurtosis
                print(f'\nSkewness for {col}: {self.df[col].skew():.2f}')
                print(f'Kurtosis for {col}: {self.df[col].kurtosis():.2f}')
    
    def categorical_analysis(self):
        """Analyze categorical columns"""
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) > 0:
            print('\nCategorical Analysis:')
            print('=' * 50)
            
            for col in categorical_cols:
                print(f'\nAnalysis for {col}:')
                value_counts = self.df[col].value_counts()
                print('\nValue Counts:')
                print(value_counts)
                
                print('\nPercentage Distribution:')
                print((value_counts / len(self.df) * 100).round(2))
                
                # Bar plot
                plt.figure(figsize=(10, 6))
                sns.countplot(data=self.df, x=col)
                plt.title(f'Distribution of {col}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
    
    def correlation_analysis(self):
        """Analyze correlations between numerical variables"""
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numerical_cols) > 1:
            print('\nCorrelation Analysis:')
            print('=' * 50)
            
            # Correlation matrix
            corr_matrix = self.df[numerical_cols].corr()
            
            # Heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix')
            plt.tight_layout()
            plt.show()
            
            # Find high correlations
            high_corr = np.where(np.abs(corr_matrix) > 0.7)
            high_corr = [(corr_matrix.index[x], corr_matrix.columns[y], corr_matrix.iloc[x, y])
                         for x, y in zip(*high_corr) if x != y]
            
            if high_corr:
                print('\nHigh Correlations (>0.7):')
                for var1, var2, corr in high_corr:
                    print(f'{var1} - {var2}: {corr:.2f}')
    
    def outlier_analysis(self):
        """Identify outliers in numerical columns"""
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        
        print('\nOutlier Analysis:')
        print('=' * 50)
        
        for col in numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)][col]
            
            if len(outliers) > 0:
                print(f'\nOutliers in {col}:')
                print(f'Number of outliers: {len(outliers)}')
                print(f'Percentage of outliers: {(len(outliers) / len(self.df) * 100):.2f}%')
                print(f'Outlier values: {outliers.values}')
    
    def run_full_analysis(self):
        """Run all analyses"""
        self.basic_info()
        self.numerical_analysis()
        self.categorical_analysis()
        self.correlation_analysis()
        self.outlier_analysis()

# Example usage
if __name__ == '__main__':
    # Load your dataset
    # df = pd.read_csv('your_data.csv')
    # eda = ExploratoryDataAnalysis(df)
    # eda.run_full_analysis()
    
    # Example with sample data
    sample_data = pd.DataFrame({
        'numeric1': np.random.normal(0, 1, 1000),
        'numeric2': np.random.normal(2, 1.5, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    eda = ExploratoryDataAnalysis(sample_data)
    eda.run_full_analysis()