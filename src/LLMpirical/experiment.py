import itertools
from tqdm import tqdm
import os
import ast
import gc
import pandas as pd
from typing import Union, Tuple, Callable, Any
import numpy as np

import torch
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM
from torch.utils.data import DataLoader

#TODO: extract desired minicons functionality into a new scoring module
from minicons import scorer  
import PopulationLM as pop


class Experiment_Set():
    '''
        model list
            perhaps allow setting model list per experiment
            add model
        experiment list
            get list
            add experiment
            remove experiment
        run experiments
    '''
    def __init__(self):
        pass

class Experiment():
    '''
        import data (this feels dubious. Probably instead just accept and allow retrieval of dataframe)
            import with inferred filetype
                    
        export data
            location and filename
            filetype
    '''
    def __init__(
                    self,
                 dataframe: pd.DataFrame,
                 prompt_col: Union[str, int] = 'prompts',
                 query_col: Union[str,int]   = 'queries',
                 target_col: Union[str, int] = 'targets',
                 result_col: str             = 'results',
                 tok_agg_func: Callable      = np.mean,
                 pop_agg_func: Callable      = None,
                 ret_attn: bool              = False,
                 attn_col: str               = 'attn',
                 insert_CoT: bool            = False,
                 ):
        '''
            if pop_agg_func == None, no aggregation by population will occur

            consider allowing/requiring passing 
        '''
        self.df = dataframe.loc[:]
    
    def run(self) -> None:
        pass

    def set_df(self, dataframe: pd.DataFrame) -> None:
        self.df = dataframe.loc[:]

    def get_df(self):
        return self.df.loc[:]

    def __call__(self, *args: Any, **kwds: Any) -> None:
        self.run()

    def __str__(self) -> str:
        pass

    def __repr__(self) -> str:
        pass

def run_experiment(exp_path,
                   transformer,
                   input_file,
                   prompt_col_idx,
                   stim_col_idx,
                   results_loc=None,
                   batch_size=10,
                   num_batches=-1,
                   committee_size=50,
                   debug_early_exit=False,
                   cloze=False
                  ):
    dataset = []
    with open(exp_path + '/' + input_file, "r") as f:
        reader = csv.DictReader(f)
        column_names = reader.fieldnames
        for row in reader:
            dataset.append(list(row.values()))

    results = []
    control_results = []
    conclusion_only = []

    column_names += ["stimulus_prob"]
    with open(results_loc, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(column_names)

    # create a lambda function alias for the method that performs classifications
    if cloze:
        call_me = lambda p1, q1: transformer.cloze_score(p1, q1)
    else:
        call_me = lambda p1, q1: transformer.conditional_score(p1, q1, reduction=lambda x: x.mean(0).item())

    stimuli_loader = DataLoader(dataset, batch_size = batch_size, num_workers=0)
    if num_batches < 0:
        num_batches = len(stimuli_loader)
    for batch in tqdm(stimuli_loader):
        out_dataset = [[] for _ in range(len(batch))]
        stim_scores = []

        if cloze:
            for i in range(len(batch)):
                for cell in batch[i]:
                    for _ in range(4): #TODO: Change this to be not hardcoded number of copies
                        out_dataset[i].append(cell)
        else:
            for i in range(len(batch)):
                out_dataset[i].extend(batch[i])

        results = {'stimulus_prob': [], 'is_correct': [], 'stimulus': []}
        p_list = list(batch[prompt_col_idx])
        if cloze:
            s_list = [ast.literal_eval(s) for s in batch[stim_col_idx]]
        else:
            s_list = list(batch[stim_col_idx])

        population = pop.generate_dropout_population(transformer.model, lambda: call_me(p_list, s_list), committee_size=committee_size)
        outs = [item for item in pop.call_function_with_population(transformer.model, population, lambda: call_me(p_list, s_list))]

        if cloze:
            stim_scores = [[[None for _ in range(committee_size)] for _ in range(len(s_list[b]))] for b in range(len(batch[0]))]
            for b in range(len(batch[0])):
                for c in range(len(s_list[b])):
                    for s in range(committee_size):
                        stim_scores[b][c][s] = outs[s][b][c]

            answer_choices_per_question = 4
            modified_outs = []
            tmp_correct = []
            for question in range(len(stim_scores)):
                for choice in range(answer_choices_per_question):
                    modified_outs.append(stim_scores[question][choice])

                results['is_correct'].extend([int(batch[2][question] == s) for s in ast.literal_eval(batch[1][question])])

                results['stimulus'].extend(s_list[question])

            results['stimulus_prob'].extend(modified_outs)

        else:
            transposed_outs = [[row[i] for row in outs] for i in range(len(outs[0]))]

            stim_scores = [score for score in transposed_outs]

            results['stimulus_prob'].extend(stim_scores)

        out_dataset.append(results['stimulus_prob'])
        if cloze:
            out_dataset[1] = results['stimulus']
            out_dataset[2] = results['is_correct']
        with open(results_loc, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerows(list(zip(*out_dataset)))

        if debug_early_exit:
            break

    del population

    print('\nResults saved to: ', results_loc)

def run_all_experiments(model,
                        lm_type,
                        input_file,
                        prompt_col_idx,
                        stim_col_idx,
                        batch_size=10,
                        committee_size=50,
                        run_base_model=False,
                        drive_loc=None,
                        save_name=None,
                        token=None,
                        debug_early_exit=False):
    # Detect file_not_found errors before loading model to save time and RAM
    for exp in experiments:
          for ds, cloze in datasets:
            exp_path = drive_loc  + '/' + ds + '/' + exp + '/' + input_file
            if not os.path.isfile(exp_path):
                raise FileNotFoundError(exp_path)

    base_model_name = model
    device='cuda'

    # Load the model
    if lm_type == "masked" or lm_type == "mlm":
        try:
          transformer = scorer.MaskedLMScorer(base_model_name,
                                              device=device,
                                              local_files_only=False,
                                              low_cpu_mem_usage=True,
                                              torch_dtype=torch.float16,
                                              device_map="auto",
                                              token=token)
        except:
          transformer = scorer.MaskedLMScorer(base_model_name, device=device, token=token)
    elif lm_type == "incremental" or lm_type == "causal":
        try:
          transformer = scorer.IncrementalLMScorer(base_model_name,
                                                   device=device,
                                                   local_files_only=False,
                                                   low_cpu_mem_usage=True,
                                                   torch_dtype=torch.float16,
                                                   device_map="auto",
                                                   token=token)
        except:
          transformer = scorer.IncrementalLMScorer(base_model_name, device=device, token=token)

    #Overwrite local model with base model (handles local loading limitation in minicons)
    if save_name is not None:
        model_name = save_name
    else:
        model_name = base_model_name

    if "/" in model_name:
      model_name = model_name.replace("/", "_")

    try:
        if run_base_model:
            for exp in experiments:
              for ds, cloze in datasets:
                print(f'Running experiment (base): {exp}')
                exp_path = drive_loc  + '/' + ds + '/' + exp
                run_experiment(exp_path,
                              transformer,
                              input_file,
                              prompt_col_idx,
                              stim_col_idx,
                              results_loc=exp_path + f"/{model_name}_base_results.csv",
                              batch_size=batch_size,
                              committee_size=1,
                              debug_early_exit=debug_early_exit,
                              cloze=cloze
                              )
                gc.collect()


        if committee_size > 0:
            pop.DropoutUtils.add_new_dropout_layers(transformer.model)
            pop.DropoutUtils.convert_dropouts(transformer.model)
            pop.DropoutUtils.activate_mc_dropout(transformer.model, activate=True, random=0.1)

            for exp in experiments:
              for ds, cloze in datasets:
                print(f'Running experiment (population): {exp}')
                exp_path = drive_loc  + '/' + ds + '/' + exp
                run_experiment(exp_path,
                              transformer,
                              input_file,
                              prompt_col_idx,
                              stim_col_idx,
                              results_loc=exp_path + f"/{model_name}_pop{committee_size}_results.csv",
                              batch_size=batch_size,
                              committee_size=committee_size,
                              debug_early_exit=debug_early_exit,
                              cloze=cloze
                              )
                gc.collect()
    except Exception as e:
        print(e)
    finally:
        del transformer
        torch.cuda.empty_cache()
        gc.collect()
    
#Replace with experiment base directory
drive_loc='/content/drive/MyDrive/evaluating_fan_effects_in_large_language_models/Experiments/'

#Name of the file where prompts are stored. Assumed to be a csv file with '|' delimiters
input_file = 'prompts.csv'

#Each experiment name should be the name a directory that descends from drive_loc
experiments = [
                'alternate',
              ]

datasets = [
    ('Misra_Typicality_Rosch', False),
]

# batch_size should be carefully chosen based on planned analysis. Because each
# batch uses a separately-generated population (limitation enforced by pytorch
# implementation details), any trials that are to be compared should be, where
# possible, included in the same batch. For example, MMLU uses multiple choice
# questions with 4 options. Because comparisons are to be made across answers
# to the same question, we should avoid splitting a question across multiple
# batches. To avoid this, batch_size is chosen to be a multiple of 4.
batch_size = 50

# committee_size is arbitrarily set at 50. No need to change until future
# investigation into appropriate population sizes. If committee_size = 0,
# then no population will be used (useful if only running the base model).
# Output file name will be of the form
#   {drive_loc}/{exp_name}/{model_name}_pop{committee_size}_results.csv
committee_size = 50

# if run_base_model is True, then the base model (no dropout) is tested in addition
# to the model. Output file name will be of the form
#   {drive_loc}/{exp_name}/{model_name}_base_results.csv
run_base_model = False

#Each model should be defined using the format (model_name, model_type, save_name)
#model_name should be the name used to load the model using the transformers library (i.e. either a file location or the huggingface name)
#model_type should be one of 'masked' or 'incremental', depending on the model structure
#save_name is where the results will be the name used when creating the results file. Results file name will be <drive_loc>/<experiment_name>/results_<save_name>.csv
models = [
    ('mistralai/Mistral-7B-v0.1',           'incremental', 'Mistral-7B',      ),
    ('upstage/SOLAR-10.7B-v1.0',            'incremental', 'SOLAR-10-7B',     ),
]
#phi-2 resumt counterfactual at hs_macroecon
#control broke for gpt-2 and gemma and phi-2

#These are the zero-based index of the prompt and the stimulus in the input csv file
prompt_col_index = 0
stim_col_index = 5

#Forces above cells to only run one iteration for testing purposes. Set to False when running full experiment
debug_EE = False

# Handles my_token undefined. If you want to use a huggingface token (necessary
# for LLaMa models among others), run the cell above this one
try:
    my_token = google.colab.userdata.get('hf_token')
except userdata.SecretNotFoundError:
    print('/************************************ALERT************************************\\')
    print('|    Token not found in secrets. Either add huggingface token to secrets or   |')
    print('|    check that the secret name matches the argumnet to userdata.get. If      |')
    print('|    the models being used do not need a token, this alert can be ignored.    |')
    print('\\************************************ALERT************************************/')

    my_token = None

# format: (model_name, model_type, save_name)
for mn, mt, sn in models:
    run_all_experiments(mn, mt,
                        input_file,
                        prompt_col_index,
                        stim_col_index,
                        batch_size=batch_size,
                        committee_size=committee_size,
                        run_base_model=run_base_model,
                        save_name=sn,
                        drive_loc=drive_loc,
                        token=my_token,
                        debug_early_exit=debug_EE
                        )