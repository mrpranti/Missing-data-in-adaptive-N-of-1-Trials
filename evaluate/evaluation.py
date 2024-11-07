import pandas as pd
from inference import inf_model, policy



# Policy and Inference Model
inference_model =  inf_model.NormalKnownVariance(
    
    prior_mean=0, prior_variance=1, variance=1
)
tmps = policy.ThompsonSampling(inference_model, number_of_treatments=2)

class Evaluatemethod:
    def __init__(self, n_sim = 100):
        self.n_sim = n_sim
        
    def get_actions(self, filled_dt, df1): ## df1 is the original data frame and filled_dt is the filled data frame

        actions = pd.DataFrame()
        actual_tr = []

        for id in range(0,100): # check the next treatment for each patient
            # get history for the patient
            actual_tr.append(df1[(df1['t']== 5) & (df1['patient_id']== id)]['treatment'].values)
            history = filled_dt[filled_dt['patient_id'] == id]
            context = history[['t', 'patient_id']].copy()
            
            action = []
            for i in range(0, self.n_sim):
                a = tmps.choose_action(history, context)
                action.append(a)
            actions[f'patien_{id}'] = action
        return actions, actual_tr
    
    def get_result(self, filled_dt):
        n_correct_tr = []
        index = []
        
        actions, actual_tr = self.get_actions(filled_dt)
        for i, column in enumerate(actions.columns):
            # number of time selected the same treatment as the actual data
            value_counts = actions[column].value_counts()[actual_tr[0]].values[0]
            n_correct_tr.append(value_counts)
            index.append(i) 
        
        return n_correct_tr, index 