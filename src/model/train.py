from skopt import BayesSearchCV
import mlflow
import os
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from utils.plots import plot_roc_curve

class MLFlowBayesianTuner:
    def __init__(self, estimator) -> None:
        self.estimator = estimator

    def train_model(self, X, y, params):
        self.estimator.set_params(**params)
        self.estimator.fit(X,y)

    def tune_model(self, X, y, search_spaces, run_name, cv=3, n_iter=50, n_jobs =1, random_state = 42):
        bayes_opt = BayesSearchCV(
            estimator=self.estimator,
            search_spaces=search_spaces,
            n_iter=n_iter,
            scoring='roc_auc',
            n_jobs=n_jobs,
            cv=cv,
            refit=True,
            random_state= random_state,
            return_train_score= True
        )

        with mlflow.start_run(run_name=run_name):
            # log model
            bayes_opt.fit(X,y)
            best_model = bayes_opt.best_estimator_
            mlflow.sklearn.log_model(best_model, 'tuned_model')

            # log params
            model_params = best_model.get_params()
            mlflow.log_params(model_params)

            # log metrics
            scores = cross_val_predict(
                self.estimator,
                X, y, cv=cv,
                method='predict_proba',
                n_jobs=n_jobs,
            )
            y_prob_cv = scores[:,1]
            y_prob_train = best_model.predict_proba(X)[:,1]
            cv_auc = roc_auc_score(y, y_prob_cv)
            train_auc = roc_auc_score(y, y_prob_train)
            mlflow.log_metrics({'CV_ROC_AUC': cv_auc})
            mlflow.log_metrics({'TRAIN_ROC_AUC': train_auc})

            # log roc_curve
            plot_roc_curve(y,y_prob_cv,cv_auc,image_path='roc_curve.png', title=f'ROC Curve - {run_name}')
            mlflow.log_artifact('roc_curve.png')
        os.remove('roc_curve.png')



