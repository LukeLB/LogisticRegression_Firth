# LogisticRegression_Firth

This set of code uses John Lees' implementation of Firth's penalised logistic regression into a sklearn estimator class. 
I've used this siuccesfully inside sklearn and with other packages e.g. SHAP.

## Usage

```python
import LogisticRegression_Firth as FLR

#create fitted estimator 
FLR = FLR.Firth_LogisticRegression()
FLR_fit = FLR.fit(X_train, y) # make sure training data has an intercpet (column of 1's)

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

## License
[cc]
