#%%
from tool_codes import configure
from tool_codes import Gait_data, Gait_model
from tool_codes import GaitDataManager, merge_gait_data
from tool_codes import GaitCSVList
from IPython.display import clear_output
from tool_codes import Hyperparameters, Parameters
from tool_codes import default_hyperparameters, coarse_hyperparameters, fine_hyperparameters, fine_hyperparameters2, postprocess_hyperparameters
from typing import Any
from copy import deepcopy
import os
import datetime
import pandas as pd
import pickle

clear_output(wait=True)

RUN={'data_preparation':True,
     'model_configuration':True,
     'model_training':True,
     'model_testing':True,
     'search_execution':'single' # 'single', 'hyper', 'hyper_post'
    }

global num_classes
num_classes=6

# ==================================================

# region Data Preparation
def data_preparation(sequence_length:int):
    global data, lists
    lists = GaitCSVList("temp/lists")
    data = GaitDataManager(num_classes=num_classes, sequence_length=sequence_length)
    data.add(lists.normal)
    data.split()
    data.process()
    data.summary()
# endregion Data Preparation

# region Model Configuration
def model_configuration(num_layer:int, dropout:float, learning_rate:float, summary:bool=True):
    global model
    model=Gait_model(example_data=data.data[0],num_layer=num_layer,dropout=dropout,learning_rate=learning_rate)
    if summary:
        model.summary()


    global merged_train, merged_val
    merged_train=merge_gait_data(data.train_data,name='merged_train')
    merged_val=merge_gait_data(data.val_data,name='merged_val')
# endregion Model Configuration

# region Model Training
def model_training(epochs:int, batch_size:int,will_save:bool=True,use_weight:bool=False):
    model.fit(train=merged_train,val=merged_val,
              epochs=epochs,batch_size=batch_size,shuffle=True,
              use_early_stopping=True,patience=3,
              use_checkpoint=True, use_weight=use_weight)
    if will_save:
        model.save('saved_models','gait_model')
        model.log('model_logs','gait_model')
        model.markdown('model_markdown','gait_model')
# endregion Model Training

# region Model Testing
def model_testing(show:bool=True,plot:bool=True,use_postprocess:bool=False,postprocess_kwargs:dict|None=None) -> list[Any]:
    results = []
    for test in data.test_data:
        returned=model.test(data=test,show=show,plot=plot,use_postprocess=use_postprocess, postprocess_kwargs=postprocess_kwargs)
        results.append(returned)
    return results
# endregion Model Testing

# ==================================================

# region parameters
test_parametars=Parameters(
    sequence_length=[130],
    num_layer=[30],
    dropout=[0.3],
    epochs=[10],
    batch_size=[64],
    learning_rate=[0.0005]
)
use_parameters=fine_hyperparameters2
use_single_parameters=default_hyperparameters
use_pp_parameters=postprocess_hyperparameters
# endregion parameters

# region Execution
if RUN['search_execution']=='hyper':
    records=[]
    acc_records=[]
    for sequence_length in use_parameters.sequence_length:
        data_preparation(int(sequence_length))
        for num_layer in use_parameters.num_layer:
            for dropout in use_parameters.dropout:
                for learning_rate in use_parameters.learning_rate:
                    for epochs in use_parameters.epochs:
                        for batch_size in use_parameters.batch_size:
                            params=Parameters(
                                sequence_length=int(sequence_length),
                                num_layer=int(num_layer),
                                dropout=float(dropout),
                                epochs=int(epochs),
                                batch_size=int(batch_size),
                                learning_rate=float(learning_rate)
                            )
                        
                            model_configuration(params.num_layer,params.dropout,params.learning_rate,False)
                            model_training(params.epochs,params.batch_size,False)
                            model_result=model_testing(False,False)
                            records.append((params,model_result))
                        
                            accs=[float(res.accuracy) for res in model_result]
                            if len(accs) == 0:
                                acc_mean = float("nan")
                                acc_std = float("nan")
                                acc_min = float("nan")
                                acc_max = float("nan")
                            else:
                                acc_mean=sum(accs)/len(accs)
                                acc_std=(sum((x - acc_mean) ** 2 for x in accs) / len(accs)) ** 0.5
                                acc_min=min(accs)
                                acc_max=max(accs)
                            acc_records.append({
                                "sequence_length":int(sequence_length),
                                "num_layer":int(num_layer),
                                "dropout":float(dropout),
                                "learning_rate":float(learning_rate),
                                "epochs":int(epochs),
                                "batch_size":int(batch_size),
                                "accuracy_mean":acc_mean,
                                "accuracy_std":acc_std,
                                "accuracy_min":acc_min,
                                "accuracy_max":acc_max
                            })
    # --- timestamp ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- directory setup ---
    base_dir = "grid_search_results"
    run_dir = f"{base_dir}/run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    # --- accuracy save ---
    df_acc = pd.DataFrame(acc_records)
    acc_path = f"{run_dir}/accuracy_results.csv"
    df_acc.to_csv(acc_path, index=False)

    # --- records save (params + raw TestResult 리스트) ---
    records_path = f"{run_dir}/records.pkl"
    with open(records_path, "wb") as f:
        pickle.dump(records, f)

    print(f"[INFO] Saved accuracy to: {acc_path}")
    print(f"[INFO] Saved raw records to: {records_path}")
    
if RUN['search_execution']=='single':
    accuracies=[]
    for i in range(5):
        data_preparation(use_single_parameters.sequence_length[0])
        model_configuration(use_single_parameters.num_layer[0],use_single_parameters.dropout[0],use_single_parameters.learning_rate[0])
        model_training(use_single_parameters.epochs[0],use_single_parameters.batch_size[0],use_weight=True)
        pp_kwargs={'k':2,'tau':0.4,'tau_trans':0.65,'force_tau':0.8}
        test_results=model_testing(use_postprocess=True,postprocess_kwargs=pp_kwargs, plot=False,show=True)
        accs=[float(res.accuracy) for res in test_results]
        accuracies.append(accs)
    all_accs=[acc for sublist in accuracies for acc in sublist]
    # --- timestamp ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- directory setup ---
    base_dir = "single_run_results"
    run_dir = f"{base_dir}/run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    # --- accuracy save ---
    df_acc = pd.DataFrame(all_accs, columns=["accuracy"])
    acc_path = f"{run_dir}/accuracy_results.csv"
    df_acc.to_csv(acc_path, index=False)
        
        
if RUN['search_execution']=='hyper_post':
    data_preparation(test_parametars.sequence_length[0])
    model=Gait_model.load('saved_models/gait_model_251124_022728.keras',example_data=data.data[0])
    acc_records=[]
    for k in use_pp_parameters['k']:
        for tau in use_pp_parameters['tau']:
            for tau_trans in use_pp_parameters['tau_trans']:
                for force_tau in use_pp_parameters['force_tau']:
                    print(f"[INFO] Post-processing with k={ k }, tau={ tau }, tau_trans={ tau_trans }, force_tau={ force_tau }")
                    postprocess_kwargs={'k':k,'tau':tau,'tau_trans':tau_trans,'force_tau':force_tau}
                    test_results=model_testing(show=True,plot=False,use_postprocess=True,postprocess_kwargs=postprocess_kwargs)
                    
                    accs=[float(res.accuracy) for res in test_results]
                    if len(accs) == 0:
                        acc_mean = float("nan")
                        acc_std = float("nan")
                        acc_min = float("nan")
                        acc_max = float("nan")
                    else:
                        acc_mean=sum(accs)/len(accs)
                        acc_std=(sum((x - acc_mean) ** 2 for x in accs) / len(accs)) ** 0.5
                        acc_min=min(accs)
                        acc_max=max(accs)
                    acc_records.append({
                        "k":k,
                        "tau":tau,
                        "tau_trans":tau_trans,
                        "force_tau":force_tau,
                        "accuracy_mean":acc_mean,
                        "accuracy_std":acc_std,
                        "accuracy_min":acc_min,
                        "accuracy_max":acc_max
                    })
    # --- timestamp ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- directory setup ---
    base_dir = "pp_grid_search_results"
    run_dir = f"{base_dir}/run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    # --- accuracy save ---
    df_acc = pd.DataFrame(acc_records)
    acc_path = f"{run_dir}/accuracy_results.csv"
    df_acc.to_csv(acc_path, index=False)
    
    print(f"[INFO] Saved accuracy to: {acc_path}")
# endregion Execution
# %%
