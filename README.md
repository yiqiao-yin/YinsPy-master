# YinsPy: Yin's Python Projects

[![YinsPythonProjects](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://yinscapital.com/research/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

This is Yin's Python Project, YinsPy, github repo  offered by Mr. Yin at [Yin's Capital](www.YinsCapital.com). The site stores sample *.py* scripts and *ipynb* notebooks for software development, statistical analysis, and machine learning applications. 

<p align="center">
  <img width="800" src="https://github.com/yiqiao-yin/YinsPy/blob/master/figs/main.gif">
</p>

- Copyright © Official python projects and products published by Yin's Capital.
- Copyright © 2010 – 2019 Yiqiao Yin
- Contact: Yiqiao Yin
- Email: Yiqiao.Yin@YinsCapital.com

## Why Is It Special Here

I pursue each and every one of my data science, machine learning, or AI projects using the following procedures. There are two phases. Phase I is about end-to-end research and due dilligence. Phase II is about product management and software maintainance as well as client relationship. In brief, they are listed out in the following.

Phase I: 

1. Develops operational procedures for collection, editing, verification, and management of statistical data. 
2. Develops and implements relevant statistical programs to incorporate data from multiple projects. 
3. Designs comprehensive relational databases with working knowledge of the scientific applications impacting on the data analysis and reporting. 
4. Assists faculty, boss, clients and other research professionals with the formulation and description of appropriate statistical methods. 
5. Evaluates research studies and recommends statistical procedures to analyze the data. 
6. Carries out comprehensive statistical analysis for a broad spectrum of types of data and types of studies. 
7. Integrates the research methodologies of multiple projects into bio-statistical analyses, statistical analysis, big data analysis, machine learning experimental design, deep learning architecture design, and so on. 

Phase II:

8. Prepares reports summarizing the analysis of research data, interpreting the findings and providing conclusions and recommendations.
9. Presents talks, seminars, or other oral presentations on the methodology and analysis used in scientific studies.
10. Assists investigators in preparation of research grant applications by writing research methods sections pertaining to acquisition, analysis, relevance and validity of data.
11. Participates in the preparation of manuscripts that are submitted for peer reviewed publication.
12. May supervise and provide training for lower level or less experienced employees.
13. May perform other duties as assigned.
14. Develop statistical packages for clients, debug, product management.

From my experience, the above checklist is important to evaluate for oneself once a while to ensure that proper steps are taken to execute data science, machine learning, and AI projects in an appropriate and efficient manner.

## Money Management

Having a sound long-term investment strategy as a part of a comprehensive money management plan helps investors keep their focus on their personal benchmarks rather than meaningless market benchmarks or indexes, enabling them to ignore short-term market events. It is about personal wealth improvement.

- As the first section discussed on this site, Yin's Timer is always going to be the frontier pipelines of our products. We are proud to present a beta version of python notebook [here](https://github.com/yiqiao-yin/Yins_Python_Projects/blob/master/scripts/python_MM_EDA_YinsTimer.ipynb).

- When it comes to portfolio construction, Markowitz is my go-to guy to discuss. Here is a notebook to discuss his point of view of efficient portfolio. Link is [here](https://github.com/yiqiao-yin/Yins_Python_Projects/blob/master/scripts/python_MM_Markowitz_Portfolio.ipynb).

- The most basic pricing model, Capital Asset Pricing Model (CAPM), is without a doubt an important discussion here on my platform. Here is a python notebook for a quick [discussion](https://github.com/yiqiao-yin/Yins_Python_Projects/blob/master/scripts/python_MM_CAPM.ipynb). 

- After foundation of capital markets, we have some understanding of asset classes risk premiums. How about the risk premiums of parameters of different asset classes or different portfolios? How do we explain these quantitative factors? This [notebook](https://github.com/yiqiao-yin/YinsPy/blob/master/scripts/python_MM_FamaMacbeth.ipynb) I discuss Fama-Macbeth regression and how 17 industry portfolios downloaded live from Fama and French's website can be used to carry out a cross-sectional panel study. 

- An important skill is to conduct simulations when it comes to money management. Monte Carlo Markov Chain is a good method to adopt. Hence, I come up with this [notebook](https://github.com/yiqiao-yin/YinsPy/blob/master/scripts/python_MM_MCMC.ipynb) to execute this idea.

## Data Structures

Data Structures are the key part of many computer algorithms as they allow the programmers to do data management in an efficient way. A right selection of data structure can enhance the efficiency of computer program or algorithm in a better way.

- An important component in data structure is navigation through different formats of data and information extraction. I actually had an interview about this following question (source is provided by [HackerRank](https://www.hackerrank.com/domains/regex?filters%5Bsubdomains%5D%5B%5D=re-applications)): this problem is a derivation of the problems provided by HackerRank (since it is an encrypted interview question, I can't copy/paste here, but I took the idea and provide my own replication of the problem) [Validate and Report IP Address](https://github.com/yiqiao-yin/Yins_Python_Projects/blob/master/scripts/python_DS_Validate_Report_IP_Address.ipynb)

- I have seen arrays and strings being tested in software engineer positions in technical interviews. I start with a simple [Evaluate Palindrome](https://github.com/yiqiao-yin/YinsPy/blob/master/scripts/python_DS_Valid_Palindrome.ipynb) here. 

- The capability to partition a data set is very important and the functionality plays an important role in data structures. I wrote this [notebook](https://github.com/yiqiao-yin/YinsPy/blob/master/scripts/python_DS_Measure_Predictivity.ipynb) to practice coding influence measure by partitioning data set according to discretized variable levels.

## Feature Selection and Feature Engineer

Domain knowledge always gives a data scientist, a machine learning practitioner, or an AI specialist the edge they needed to design the appropriate machine leaning algorithms. The information to represent domain knowledge is to construct informative features. 

- The most common feature engineer methodology is to use k-nearest neighbors and this [notebook](https://github.com/yiqiao-yin/YinsPy/blob/master/scripts/python_FSFE_KMeans.ipynb) I explain how to do that on titanic data set.

- Feature selection is important in a way that it can allow data scientists and machine learning practitioners to directly interact with the informative information that leads to correct model specification. It is essential to let data speak for itself instead of assuming an underlying model which is why conventional methodologies such as step-wise regression by AIC/BIC pose a challenge. To tackle this problem, we introduce a nonparametric feature selection method pioneered by Professor Shawhwa Lo at Columbia University. This [notebook](https://github.com/yiqiao-yin/YinsPy/blob/master/scripts/python_DS_Measure_Predictivity.ipynb) I introduce a vanilla version of influence score, a metric designed to take in $X$ and $y$ and spits out how predictive $X$ influencing $y$. Based on Influence Score (also known as "influence score" or "i-score"), we design a greedy backward dropping algorithm taking full advantage of the unique property of Influence Score (e.g. I-score) which states that I-score is high when selected covariate matrix $X$ has larger impact on target variable $y$ and is low when selected covariate matrix $X$ has lower impact or noise within on $y$. I discuss the Backward Dropping Algorithm in this [notebook](https://github.com/yiqiao-yin/YinsPy/blob/master/scripts/python_FSFE_InfluenceScore.ipynb)

## Machine Learning

Machine Learning is another big component of Yin's Capital product. 

- One should always start with vanilla regression problem to start the journey of machine learning. This is why I start with a simply python notebook of [Housing Price Analysis](https://github.com/yiqiao-yin/YinsPy/blob/master/scripts/python_ML_SimpleLinearRegression_Housing.ipynb).

- As data gets bigger, we can start to see how multiple variables together impose an impact on dependent variable. It is not horribly difficult to make the transition from simple linear regresssion to multivariate linear regression problem. A data science project can be carried out to investigate investment startup profits using multivariate regression model which is [here](https://github.com/yiqiao-yin/YinsPy/blob/master/scripts/python_ML_MultiLinearRegression_StartInvest.ipynb).

- Simple and Multivariate Linear Models may be fast in making predictions. However, the variables can only be assumed to affect the dependent variable marginally. This may miss some interactions that have joint effect to the target variable. This fact provide us motivations to use tree-based learning algorithms. This [notebook](https://github.com/yiqiao-yin/YinsPy/blob/master/scripts/python_ML_DecisionTreeRegressor_StartInvest.ipynb) I discuss Decision Tree Regressor as a leap from conventional regression problem and we are going to use visualization tools to help us understand why the machines do what it does.

- Machine Learning has a lot of components. This [notebook](https://github.com/yiqiao-yin/YinsPy/blob/master/scripts/python_ML_KNNRegressorClassifier.ipynb) I introduce a famous machine learning techniques using K-Nearest Neighborhood and I walk through each and every step in conducting a standardized machine learning projects including but not limiting to training and validating set examination, cross validation, feature selection, explore different loss functions, and performance visualization.

## Deep Learning

A higher level of form of machine learning is deep learning. The mystery of deep learning pose a great potential as well as threat to mankind. Data is unbiased. Algorithms are biased. However, the design of experiment by human can almost never by 100\% impartial. 

- The first notebook I discuss a [basic neural network based regression](impartial) problem predicting fuel efficiency. 

- To take things to another level, I move away from regression problem and attempt a very basic classification problem. The [notebook](https://github.com/yiqiao-yin/YinsPy/blob/master/scripts/python_DL_NN3.ipynb) here conducts a basic 3-layer neural network architecture and lands on software development.

- Object Detection is a higher level of artistic representation of the usage of deep learning. Specifically, there are object detection, facial detection, gender detection, and object localization. In this [notebook](https://github.com/yiqiao-yin/YinsPy/blob/master/scripts/python_DL_SimpleCVLib.ipynb), I use open source *cvlib* as playground to illustrate some basic applications of this style. An advanced version is YOLO algorithm with live camera feed. Fortunately, open source *cvlib* have made the production code fairly easy for customer use and in this [notebook](https://github.com/yiqiao-yin/YinsPy/blob/master/scripts/python_DL_YOLO_Live.ipynb) I explain how to deploy the usage of such technology. An interesting application is posted below.

<p align="center">
  <img width="400" src="https://github.com/yiqiao-yin/YinsPy/blob/master/videos/2019_12_30.gif">
</p>
