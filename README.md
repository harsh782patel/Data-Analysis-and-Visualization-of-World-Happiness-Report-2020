# Happiness Insights 2020
![World Happiness Report](assets/id8lGFstdW_1751913596371.svg)

[![kaggle](https://img.shields.io/badge/Dataset%20WHR%202020-pink?logo=kaggle&logoColor=%23ffffff&logoSize=auto&labelColor=%2329bdfb&color=%2329bdfb&link=https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Flondeen%2Fworld-happiness-report-2020)](https://www.kaggle.com/datasets/londeen/world-happiness-report-2020)

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

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](#)
[![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=fff)](#)
[![NumPy](https://img.shields.io/badge/NumPy-4DABCF?logo=numpy&logoColor=fff)](#)
[![Scikit-learn](https://img.shields.io/badge/-scikit--learn-%23F7931E?logo=scikit-learn&logoColor=white)](#)
[![SHAP](https://img.shields.io/badge/SHAP-pink?logo=shap&label=v0.48.0&labelColor=%23ae13a6&color=%23ae13a6)](#)
[![Plotly](https://img.shields.io/badge/Plotly-red?logo=plotly&labelColor=%23ce2e62&color=%23ce2e62)](#)
[![Matplotlib](https://custom-icon-badges.demolab.com/badge/Matplotlib-71D291?logo=matplotlib&logoColor=fff)](#)
[![Seaborn](https://img.shields.io/badge/Seaborn-Blue?logo=%237db0bc&label=v0.13.2&labelColor=%237db0bc&color=%237db0bc)](#)
[![Pycountry](https://img.shields.io/badge/PyCountry-pink?logo=shap&label=v24.6.1&labelColor=%23008bfb&color=%23008bfb)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Live Demo

The application is deployed on Streamlit Community Cloud: [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://harsh782patel-data-analysis-and-visualization-of--appapp-kmou9w.streamlit.app)

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
