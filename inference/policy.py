
import random



class ThompsonSampling():
    def __init__(
        self,
        inference_model,
        number_of_treatments,
        posterior_update_interval=1,
        **kwargs,
    ):
        self.inference = inference_model
        self.posterior_update_interval = posterior_update_interval
        self.number_of_treatments = number_of_treatments
        self._debug_data = []
        super().__init__(**kwargs)

    @property
    def additional_config(self):
        return {"inference": f"{self.inference}"}

    def __str__(self):
        return f"ThompsonSampling({self.inference})"

    def choose_action(self, history, context):
        if (
            len(history) % self.posterior_update_interval == 0
            or self.inference.trace is None
        ):
            self.inference.update_posterior(history, self.number_of_treatments)

        probability_array = self.inference.approximate_max_probabilities(
            self.number_of_treatments, context
        )
        action = random.choices(
            range(self.number_of_treatments), weights=probability_array
        )[0]
       ## self._debug_information += [
        #    f"Probabilities for picking: {numpy.array_str(numpy.array(probability_array), precision=2, suppress_small=True)}, chose {action}"
       # ]
       # debug_data_from_model = self.inference.debug_data
        #self._debug_data.append(
         #   {**{"probabilities": probability_array}, **debug_data_from_model}
        #)
        return action
    
    @property
    def debug_data(self):
        return self._debug_data