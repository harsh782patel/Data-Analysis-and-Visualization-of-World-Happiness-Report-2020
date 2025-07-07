# Data-Analysis-and-Visualization-of-World-Happiness-Report-2020
![World Happiness Report](assets/id8lGFstdW_1751913596371.svg)

## Overview
This project analyzes the World Happiness Report 2020 dataset, focusing on happiness metrics during the first year of the COVID-19 pandemic. The analysis explores how different factors like GDP, social support, health expectancy, freedom, generosity, and corruption perception contributed to national happiness levels during this challenging period.

The project demonstrates a complete data science workflow including:
- Exploratory data analysis (EDA)
- Statistical analysis and hypothesis testing
- Machine learning modeling
- Model interpretation with SHAP values
- Interactive visualization
- Deployment as a Streamlit web application

## Features

- **Global Happiness Map:** Interactive choropleth map showing happiness scores worldwide
- **Regional Analysis:** Comparison of happiness across different regions
- **Metrics Explanation:** Clear explanations of all happiness factors
- **Key Drivers Analysis:** Identification of most important happiness factors during COVID-19
- **Happiness Predictor:** Interactive tool to predict national happiness based on input parameters
- **Policy Recommendations:** Data-driven policy suggestions based on analysis

[here](https://www.kaggle.com/datasets/londeen/world-happiness-report-2020).

## Technologies Used

- Python 3.9+
- Streamlit (web framework)
- Pandas (data manipulation)
- NumPy (numerical computing)
- Scikit-learn (machine learning)
- SHAP (model interpretation)
- Plotly (interactive visualizations)
- Seaborn (statistical visualizations)
- Pycountry (country code conversion)

## Live Demo

The application is deployed on Streamlit Community Cloud: 

## Installation

To run this project locally:
- Clone the repository:
```bash
git clone https://github.com/yourusername/world-happiness-analysis.git
cd world-happiness-analysis
```
- Install the required packages:
```bash
pip install -r requirements.txt
```
- Run the Streamlit app:
```bash
streamlit run app.py
```
## Key Insights from Analysis

- Social support was 28% more impactful than GDP during the pandemic
- Health expectancy showed strongest correlation with happiness in Western nations
- Nordic countries consistently showed the highest happiness scores
- Economic factors became less predictive of happiness during the pandemic
- A minimum 0.75 social support threshold is needed for happiness resilience

## Usage

After launching the application, navigate through the tabs to explore different aspects of the analysis:
- **Data Overview:** Dataset summary and correlation matrix
- **Metrics Explained:** Detailed explanations of happiness factors
- **Regional Analysis:** Happiness comparisons across regions
- **Key Drivers:** Most important happiness factors during COVID-19
- **Predict Happiness:** Interactive prediction tool
- **World Map:** Global happiness distribution visualization

## Contributing

Contributions are welcome! Please follow these steps:
- Fork the repository
- Create your feature branch (```git checkout -b feature/AmazingFeature```)
- Commit your changes (```git commit -m 'Add some AmazingFeature'```)
- Push to the branch (```git push origin feature/AmazingFeature```)
- Open a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
