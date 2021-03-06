# LogisticRegression_Firth

This set of code uses John Lees' implementation of Firth's penalised logistic regression to make an sklearn estimator class. 
I've used this succesfully inside sklearn and with other packages e.g. SHAP. You can find John's original code here
https://gist.github.com/johnlees/3e06380965f367e4894ea20fbae2b90d. 

## Usage

```python
import LogisticRegression_Firth as FLR

#create fitted estimator 
FLR = FLR.Firth_LogisticRegression()
FLR_fit = FLR.fit(X_train, y) # make sure training data has an intercept (column of 1's)

#perform stat test. Either likelihood ratio test,
FLR.test_likelihoodratio(X_train, y)
#or Wald test
FLR_wald_pvals = FLR.test_wald()

#to use sklearn functions simply use the standard syntex however rmember to remove the dummy 
#column of 1's from your data. e.g.
FLR_fit.predict(X_valid)
FLR_fit.predict_log_proba(X_valid)
FLR_fit.predict_proba(X_valid)
FLR_fit.score(X_valid, y)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

There are some obvious areas for improvement in this code. When I have time I may remove the current need for the user to add an intercept to the data. Another apparant problem is that this estimator does not pass sklearn's check_estimator function. I believe this is because check_estimator attempts to pass a y vector containg {0, 1, 2} to check it works in a multinomial classifiaction system. This won't work because the original function was written using an instance of a binomial logit model smf.Logit(y, X).

## License
[cc]
