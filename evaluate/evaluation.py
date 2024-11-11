import pandas as pd
from inference import inf_model, policy
import numpy as np
import scipy.stats as stats


# Policy and Inference Model
inference_model =  inf_model.NormalKnownVariance(
    
    prior_mean=0, prior_variance=1, variance=1
)
tmps = policy.ThompsonSampling(inference_model, number_of_treatments=2)

class Evaluatemethod:
    def __init__(
        self, 
        t, 
        length,
        df1, 
        pop_mean = [2,0], 
        pop_var= [1,1],
        n_sim = 1000,
        number_of_patients = 100, 
        number_of_actions = 2
    ):
        self.t = t # t define as the next time cycle after the missing value row
        self.length = length
        self.df1 = df1
        self.pop_mean = pop_mean
        self.pop_var = pop_var
        self.n_sim = n_sim
        self.number_of_patients = number_of_patients
        self.number_of_actions = number_of_actions
        
        
    def get_attribute(self):
        return self.t, self.length

    def get_actions(self, filled_dt): ## df1 is the original data frame and filled_dt is the filled data frame

        actions = pd.DataFrame()
        actual_tr = []

        for id in range(0,self.number_of_patients): # check the next treatment for each patient
            # get history for the patient
            actual_tr.append(self.df1[(self.df1['t']== self.t) & (self.df1['patient_id']== id)]['treatment'].values)
            history = filled_dt[filled_dt['patient_id'] == id]
            context = history[['treatment', 'patient_id']].copy()
            
            action = []
            for i in range(0, self.n_sim):
                a = tmps.choose_action(history, context)
                action.append(a)
            actions[f'patien_{id}'] = action
        return actions, actual_tr
    
    def get_action_result(self, filled_dt):
        n_correct_tr = []
        index = []
        
        actions, actual_tr = self.get_actions(filled_dt)
        for i, column in enumerate(actions.columns):
            # number of time selected the same treatment as the actual data
            value_counts = actions[column].value_counts()[actual_tr[0]].values[0]
            n_correct_tr.append(value_counts)
            index.append(i) 
        
        return n_correct_tr, index 
    
    def observe_outcome(self, action):
        treatment_index = action
        return np.random.normal(
                self.pop_mean[treatment_index], np.sqrt(self.pop_var[treatment_index])
            )
    
    def get_newdt(self, filled_dt, missing_patient_id):
    
        new_dt = filled_dt.copy()
        for id in range(0,self.number_of_patients):
            
            if np.isin(id, missing_patient_id).any():
                next_dt = filled_dt[filled_dt['patient_id'] == id].copy()
          
                for i in range(self.t, self.length):
                    history = next_dt.copy()
                    action = tmps.choose_action(history, self.number_of_actions)
                    next_cycle = self.df1[(self.df1['t']>i-1) & (self.df1['t'] <= i) & (self.df1['patient_id'] == id)].copy()
                    
                    # put the choosed action on treatment column
                    next_cycle.loc[next_cycle.index.values[0], 'treatment'] = action
                    
                    # new observed outcome to the outcome column
                    next_cycle.loc[next_cycle.index.values[0], 'outcome'] =self.observe_outcome(action)
                    
                    next_dt = pd.concat([next_dt, next_cycle], axis= 0, ignore_index=False) # add the next cycle to the data
                # add to the main data  
                new_dt =  pd.concat([new_dt, next_dt], axis= 0, ignore_index=False) 
            else:
                next_dt = self.df1[self.df1['patient_id'] == id]
                new_dt =  pd.concat([new_dt, next_dt], axis= 0, ignore_index=False)

        new_dt = new_dt.sort_index()
        return new_dt.drop_duplicates()
    
    def confidence_interval(self, df, column, confidence_level=0.95):
        
        sample = df[column]
        n = len(sample)                       # Sample size
        mean = sample.mean()                  # Mean of the sample
        std_err = sample.sem()                # Standard error of the mean

        # Calculate the margin of error
        margin_of_error = stats.t.ppf((1 + confidence_level) / 2, n - 1) * std_err

        # Calculate confidence interval
        lower_bound = mean - margin_of_error
        upper_bound = mean + margin_of_error

        return (lower_bound, upper_bound)

    def get_mean_var(self, filled_dt, missing_patient_id):
        
        mean_0 = []
        mean_1 = []
        var_0 = []
        var_1 = []
        for i in range(0, self.n_sim):
            new_dt = self.get_newdt(filled_dt, missing_patient_id)
            mean, var = inference_model.update_posterior(new_dt, self.number_of_actions)
            mean_0.append(mean[0])
            mean_1.append(mean[1])
            var_0.append(var[0])
            var_1.append(var[1])
        output = {
            'mean of treatment 0': mean_0,
            'mean of treatment 1': mean_1,
            'variance of treatment 0': var_0,
            'variance of treatment 1': var_1, 
        }
        return pd.DataFrame(output)
    
    