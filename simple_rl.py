import random
import os
from json import dumps
from time import sleep

def normalize(num: int,factor: int) -> float:
    return num/factor
def clamp(num: int | float, min: int, max: int) -> int:
    if num < min: return min
    elif num > max: return max
    return num

class Agent:
    def __init__(self, inputs: int, processors: int, outputs: int,
                 processor_values: list, output_values: list) -> None:
        self._inputs = []
        self.processors = {}
        self.outputs = []
        self.rights = 0
        self.wrongs = 0
        self.doubt = 0
        self.points = 0

        self.output_values = output_values

        _node = {
            'weight': 0,
            'bias': 0,
        }

        self._associate_with = {}

        self._associate = {
            'certainty':0
        }

        for p in range(processors):
            _proc_val = processor_values[p]
            _proc_node = _node.copy()
            _proc_node['weight'] = normalize(random.randint(0,5), 10)
            _proc_node['bias'] = normalize(random.randint(-5, 5),10)
            self.processors[_proc_val] = _proc_node
        
    def predict(self, state) -> tuple[int | str, float, list]:
        outcomes = self.associate_with(state)
        state = normalize(state, 10)
        predictions = []
        for p in self.processors:
            p = self.processors[p]
            _weight = p['weight']
            _bias = p['bias']
            _pred = clamp(state * _weight + _bias, 0, 1)
            predictions.append(_pred)

        if outcomes is not None: 
            _prev = 0
            _curr = 0
            for oc in outcomes.keys():
                certainty, outcome_associate = outcomes[oc]['certainty'], oc
                _curr = certainty
                if _prev > _curr: _curr = _prev
                _prev = _curr
                outcomes[oc]['certainty'] += certainty
                outcomes[oc]['certainty'] = clamp(outcomes[oc]['certainty'], 0.1, 1)
            _highest = _prev
            if _highest > random.randint(0, int(self.doubt))/10 or self.points < 0:
                return outcome_associate, certainty, predictions

        if isinstance(state, int) or isinstance(state, float):
            highest = max(predictions)
            adjacent_node = list(self.processors)[predictions.index(highest)]

            return adjacent_node, highest, predictions
        
    def reward(self, prediction, correct_outcome, predictions: list) -> bool:
        if prediction != correct_outcome:
            _wdata = self.processors[prediction]
            _rdata = self.processors[correct_outcome]
            _wpred = predictions[prediction]
            _rpred = predictions[correct_outcome]
            error_margin = clamp(_wpred - _rpred, 0.1, 0.75)

            _nwdata = _wdata.copy()
            _nwdata['weight'] = _nwdata['weight'] * error_margin - _nwdata['weight']/10
            _nwdata['bias'] = _nwdata['bias'] * error_margin * 0.1
            _nrdata = _rdata.copy()
            _nrdata['weight'] = _nrdata['weight'] / error_margin + _nrdata['weight']/10
            _nrdata['bias'] = _nrdata['bias'] / error_margin * 0.1

            self.processors[prediction] = _nwdata
            self.processors[correct_outcome] = _nrdata
            self.wrongs += 1
            self.points -= 1
            self.doubt = clamp(self.rights/(self.rights + self.wrongs)*10, 2, 8)
            return False, _wpred
        else:
            _rpred = predictions[correct_outcome]
            _rdata = self.processors[prediction]
            _nrdata = _rdata.copy()
            _nrdata['weight'] = _nrdata['weight'] + _nrdata['weight']/10
            _nrdata['bias'] = _nrdata['bias'] + _nrdata['bias']/10
            self.processors[prediction] = _nrdata
            self.rights += 1
            self.points += 1
            self.doubt = clamp(self.rights/(self.rights + self.wrongs)*10, 2, 8)
            return True, _rpred
        
    def associate(self, state, outcome, certainty: int) -> None:
        certainty = normalize(certainty, 10)
        _associate = self._associate.copy()
        _associate['certainty'] += certainty
        _associate['certainty'] = clamp(_associate['certainty'], 0, 1)
        if not state in self._associate_with:
            self._associate_with[state] = {outcome:_associate}
        else:
            _state = self._associate_with[state]
            if not outcome in _state.keys():
                _state[outcome] = _associate
            else:
                self._associate_with[state][outcome]['certainty'] += certainty
                self._associate_with[state][outcome]['certainty'] = clamp(self._associate_with[state][outcome]['certainty'], 0, 1)

    def associate_with(self, state) -> None | dict:
        if not state in self._associate_with:
            return None
        else:
            _state = self._associate_with[state]
            return _state

accs = []
tests = 100
for b in range(tests + 1):
    rights = 0
    wrongs = 0
    agent = Agent(1, 201, 1, [i for i in range(201)], [i for i in range(201)])
    for i in range(1, 1000):
        state = random.randint(0, 200)
        outcome = clamp(state + random.randint(1,50), 0, 200)
        pred, confidence, predictions = agent.predict(state)
        rewarded, conf = agent.reward(pred, outcome, predictions)
        if rewarded: 
            agent.associate(state, outcome, conf/20)
            rights += 1
        elif not rewarded: 
            agent.associate(state, outcome, -conf/20)
            wrongs += 1
    acc = rights/(rights + wrongs)*100
    print(f'Running agent {b} with {round(acc)}% accuracy and {agent.points} points.')
    accs.append(round(acc))

print('Agent predictions completed.')
print(f'Average accuracy over {tests} tests: ~{round(sum(accs)/len(accs))}%')
print(f'Best run at {max(accs)}%. Worst run at {min(accs)}%')