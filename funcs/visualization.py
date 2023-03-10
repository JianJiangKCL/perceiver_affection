import pandas as pd
import numpy as np
import wandb
import itertools
api = wandb.Api()


# sorted runs by modalities
# runs = sorted(runs, key=lambda x: x.config["modalities"])
def latex_table(fn):
    def wrapper(*args, **kwargs):
        # print content from table_template.tex
        with open("latex/table_template_v3.tex", "r") as f:
            # print if not \bottomrule
            # iterate over lines
            line = ''
            pre_table = ''
            caption_name = kwargs['caption_name']
            while "bottomrule" not in line:

                if "\\caption{" in line and caption_name is not None:
                    line = caption_name + "\n"
                pre_table += line
                line = f.readline()
            print(pre_table)
            fn(*args, **kwargs)
            print(line)
            # read() the rest of the file
            print(f.read())
            print('\n \n')
            return None
    return wrapper


def find_already_fair_runs(target_group):
    sensitive_group = ['gender', 'age']
    sensitive_group.remove(target_group)
    sensitive_group = sensitive_group[0]
    with open(f"latex/{sensitive_group}_second_stage.tex", "r") as f:
        search_range = [i for i in range(5, 10)] if target_group =='gender' else [i for i in range(13, 18)]

        to_keep = {}
        line = f.readline()
        while line:

            items = line.split('&')
            name = items[0]
            # join items to string based on search_range
            items = [items[i] for i in search_range]
            to_search = ''.join(items)

            if '%' in line:
                to_keep[name] = line

            elif 'textcolor' not in to_search:
                to_keep[name] = line
            line = f.readline()

        return to_keep



@latex_table
def print_sorted_runs(sorted_modalities_names, sorted_items, previous_results_to_keep=None, caption_name=None, ):
    def modifiy_item(item):
        # replace audio with Audio
        item = item.replace("%", "")
        item = item.replace("audio", "Audio")
        # replace facebody with FaceBody
        item = item.replace("facebody", "Video")
        # replace senti,speech,time with Textual
        item = item.replace("senti,speech,time", "Textual")
        # replace senti,speech,text,time with  BERT, Textual
        item = item.replace("senti,speech,text,time", "BERT, Textual")
        # # replace text with BERT, except \textcolor
        item = item.replace("text", "BERT")
        item = item.replace("\BERT", "\\text")
        return item
    for name, tmp in zip(sorted_modalities_names, sorted_items):

        item = ''
        # covert name to string
        modalities = ','.join(name)
        item += modalities + ' & '
        item += tmp
        item = modifiy_item(item)
        modified_name = item.split('&')[0]
        if previous_results_to_keep is not None and modified_name in previous_results_to_keep.keys():
            to_get_name = "%" + modified_name
            item = previous_results_to_keep[to_get_name]
            item = modifiy_item(item)

        print(item)


def record_runs(to_report_metrics, to_separate_metrics, filter_configs, to_report_configs, runs, highlight_flag=True, print_flag=True):
    items = []
    modalities_names = []
    for run in runs:
        flag_filtered = False
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        item = ''
        # filter out runs that do not match the filter metrics
        for key in filter_configs:
            if key in run.config:
                if run.config[key] != filter_configs[key]:
                    # print("key", key, "value", run.config[key], "not match", filter_configs[key])
                    flag_filtered = True
                    break

        if not flag_filtered:
            for key in to_report_configs:
                if key in run.config:
                    # process the modality name
                    if key == "modalities":
                        modalities = run.config[key]
                        # if is a list, then get the first element

                        if isinstance(modalities, list):
                            modalities = modalities[0]
                        # if contains _ , then split and sort
                        if '_' in modalities:
                            modalities = modalities.split('_')
                            # sort the modalities based alphabetical order
                            modalities = sorted(modalities)
                            # combine them with ','
                            # modalities = ','.join(modalities)
                        # item += modalities + ' & '
                        modalities_names.append(modalities)
                    else:
                        item += str(run.config[key]) + ' & '

            for key in to_report_metrics:
                if key in run.summary._json_dict:

                    if key in to_separate_metrics:
                    # if "val_mse" in key:
                        # keep 4 decimal places
                        item += str(round(run.summary._json_dict[key], 4)) + ' & '
                    # times 100, and
                    else:
                        value = round(run.summary._json_dict[key] * 100, 2)
                        if value < 80 and highlight_flag:
                            #  \textcolor{red}{0.7907}
                            item += '\\textcolor{red}{' + str(value) + '} & '
                        else:
                            item += str(value) + ' & '
            # remove the last &
            item = item[:-2]
            # plus \\
            item += '\\\\'
            items.append(item)

    # print(modalities_names)
    # make each element a list, if it a string add it to a list
    modalities_names = [x if isinstance(x, list) else [x] for x in modalities_names]
    # print(modalities_names)
    # sort with number of elements, in default sorted with alphabetical order
    lengths = [len(x) for x in modalities_names]
    sorted_indices = sorted(range(len(lengths)), key=lambda k: lengths[k])
    sorted_modalities_names = [modalities_names[i] for i in sorted_indices]
    sorted_items = [items[i] for i in sorted_indices]
    if print_flag:
        print_sorted_runs(sorted_modalities_names, sorted_items)
    return sorted_modalities_names, sorted_items



# for multiple random seeds
def multiple_runs(seeds, runs, to_report_metrics, to_separate_metrics, filter_configs, to_report_configs, caption_prefix, previous_results_to_keep=None):
    all_items = []
    for seed in seeds:
        filter_configs["seed"] = seed
        sorted_modalities_names, sorted_items = record_runs(to_report_metrics, to_separate_metrics, filter_configs,
                                                            to_report_configs, runs, highlight_flag=False,
                                                            print_flag=False)
        # split by &
        sorted_items = [x.split(' & ') for x in sorted_items]
        # # remove \\\\ for the last item in each list
        for i in range(len(sorted_items)):
            sorted_items[i][-1] = sorted_items[i][-1].replace('\\\\', '')
        # convert str to float
        for i in range(len(sorted_items)):
            for j in range(len(sorted_items[i])):
                sorted_items[i][j] = float(sorted_items[i][j])
        all_items.append(sorted_items)

    k = 1
    latex_table = []
    all_items = np.array(all_items)
    to_sep = [0, 1, 2, 3, 9, 10, 11]
    # to_sep = [0, 1]
    # compute the average of all items
    for i in range(len(all_items[0])):
        avged_items = ''
        for j in range(len(all_items[0][i])):
            avg = np.mean(all_items[:, i, j])
            if j in to_sep:
                avged_items += str(round(avg, 2)) + ' & '
            else:
                if avg < 80:
                    #  \textcolor{red}{0.7907}
                    avged_items += '\\textcolor{red}{' + str(round(avg, 2)) + '} & '
                else:
                    avged_items += str(round(avg, 2)) + ' & '
        avged_items = avged_items[:-2] + '\\\\'
        latex_table.append(avged_items)
    # caption_name is the combination of seeds and target_sensitive_group
    target_sensitive_group = filter_configs["target_sensitive_group"]
    caption_name = caption_prefix + '\_'.join([str(x) for x in seeds])
    caption_name = '\caption{'+ caption_name +  target_sensitive_group
    # plus gamma
    # caption_name += '  gamma\_' + str(filter_configs["gamma"])
    # # plus beta
    # caption_name += '  beta\_' + str(filter_configs["beta"])
    # end }
    caption_name += '}'
    print_sorted_runs(sorted_modalities_names, latex_table, caption_name=caption_name, previous_results_to_keep=previous_results_to_keep)

print('dada')

to_report_configs = ["modalities"]
# to_report_metrics = [ "val_mse", "gender_val_DIR_O", "gender_val_DIR_C", "gender_val_DIR_E", "gender_val_DIR_A", "gender_val_DIR_N", "age_val_DIR_O", "age_val_DIR_C", "age_val_DIR_E", "age_val_DIR_A", "age_val_DIR_N"]
to_report_metrics = [ "val_mse", "gender_val_MSE_1", "gender_val_MSE_0", "gender_val_MSE_gap", "gender_val_DIR_O", "gender_val_DIR_C", "gender_val_DIR_E", "gender_val_DIR_A", "gender_val_DIR_N", "age_val_MSE_1", "age_val_MSE_0", "age_val_MSE_gap", "age_val_DIR_O", "age_val_DIR_C", "age_val_DIR_E", "age_val_DIR_A", "age_val_DIR_N"]
to_separate_metrics = ["val_mse", "gender_val_MSE_gap", "age_val_MSE_gap", "age_val_MSE_1", "age_val_MSE_0", "gender_val_MSE_1", "gender_val_MSE_0"]
# replace val with test
key_name_test = 'test'
to_report_metrics = [x.replace('val', key_name_test) for x in to_report_metrics]
to_separate_metrics = [x.replace('val', key_name_test) for x in to_separate_metrics]

# for baseline
# filter_configs = {"depth":3, "lr": 0.004, "num_latents": 128, "epochs":50, "results_dir": "results/trainval_bul_baseline_right_uniqueMean"}
# runs = api.runs("jianjiang/perceiver_affection_baseline_trainval_bul_right")
# runs = api.runs("jianjiang/perceiver_affection_v100_age26_trainval")
# # 1996 and 6
# filter_configs = {"depth":5, "lr": 0.004, "num_latents": 128, "epochs":60, "seed":1996}
# runs = api.runs("jianjiang/perceiver_affection_baseline_trainval_a5000")
# filter_configs = {"cpc_layers":2, "lr": 0.001, "dropout_prj": 0.3, "epochs":30, "sigma":0.1, "scheduler": "multistep"}
# runs = api.runs("jianjiang/mmim_affection_hyper_trainval")
# for second
# runs = api.runs("jianjiang/perceiver_affection_spd_trainval_3090")
# filter_configs = { "lr": 0.004, "gamma": 5, "epochs": 5, "num_latents": 128, "seed": 1995, "target_sensitive_group": "age"}
# runs = api.runs("jianjiang/perceiver_affection_spd_v100_age26_trainval")
# runs = api.runs("jianjiang/perceiver_affection_spd_trainval_a5000")
# filter_configs = { "lr": 0.001, "gamma": 5, "epochs": 1, }
# runs = api.runs("jianjiang/mmim_affection_spd_trainval_3090")
# for test
# to_report_metrics = [ "test_loss", "gender_test_DIR_O", "gender_test_DIR_C", "gender_test_DIR_E", "gender_test_DIR_A", "gender_test_DIR_N", "age_test_DIR_O", "age_test_DIR_C", "age_test_DIR_E", "age_test_DIR_A", "age_test_DIR_N"]
# to_separate_metrics = ["test_loss"]
# runs = api.runs("jianjiang/perceiver_affection_test")
# filter_configs = {}

# for third

# to_separate_metrics = ["val_mse"]
# runs = api.runs("jianjiang/perceiver_affection_spd_third_trainval_bul")
# filter_configs = {"beta": 0.1, "results_dir": "results/sweep_spd_trainval_devicebul_third_ablation", "gamma":10}
# runs = api.runs("jianjiang/perceiver_affection_third_trainval_a5000")
# runs = api.runs("jianjiang/perceiver_affection_third_trainval_v100_age26")
# filter_configs = { "lr": 0.004, "beta": 1, "epochs":5, "num_latents": 128, "seed": 1995, "gamma":5 , "target_sensitive_group": "gender"}


####### final mimm
# filter_configs = {"cpc_layers":2, "lr": 0.001, "dropout_prj": 0.3, "epochs": 30, "sigma":0.1}
# runs = api.runs("jianjiang/mmim_affection_base_trainval")

# epoch 5; dropout 0.3 or 0 will let age debiase gender more
# epoch 1 dropout 0; just did the job
# filter_configs = {"cpc_layers": 2, "lr": 0.001, "dropout_prj": 0.3, "epochs": 1, "sigma": 0.1, "gamma": 5, "beta": 2}
# runs = api.runs("jianjiang/mmim_affection_spd_trainval_3090")
# runs = api.runs("jianjiang/mmim_affection_third_trainval_3090_final")
#############final perceiver
# filter_configs = {"depth":5, "lr": 0.004, "num_latents": 128, "epochs":60, "seed":1996}
# runs = api.runs("jianjiang/perceiver_affection_baseline_trainval_3090_final")


# filter_configs = {"depth": 5, "lr": 0.004,  "epochs": 5, "seed":1996, "gamma": 5, "beta": 0.5}
# runs = api.runs("jianjiang/perceiver_affection_ablation_test")
# runs = api.runs("jianjiang/perceiver_affection_spd_trainval_3090_final")
# runs = api.runs("jianjiang/perceiver_affection_ablation_trainval_3090_final")
# filter_configs = {"depth": 5, "lr": 0.004,  "epochs": 5, "seed":1996, "gamma": 3, "beta": 10}
# runs = api.runs("jianjiang/perceiver_affection_third_trainval_3090_final")
# filter_configs = {"depth": 5, "lr": 0.004,  "epochs": 5, "seed":1996, "gamma": 3, "beta": 0.5, "results_dir": "/DATA/jj/affection/results/trainval_kd_ablation_final", "is_baseline": 0}
# runs = api.runs("jianjiang/perceiver_affection_ablation_trainval_3090_final")

filter_configs = { "results_dir": '/DATA/jj/affection/results/trainval_3090_baseline_several_biased', "bias_sensitive": "gender"}
runs = api.runs("jianjiang/perceiver_affection_baseline_trainval_3090_biased")

#######################


### test dir(1-)
# filter_configs = {"depth": 5, "lr": 0.004,  "epochs": 5, "seed":1996, "gamma": 8, "results_dir": "/DATA/jj/affection/results/trainval_kd_test"}
# runs = api.runs("jianjiang/perceiver_affection_ablation_test")

# record_runs(to_report_metrics, to_separate_metrics, filter_configs, to_report_configs, runs)
# 0 6 1995 1996 1997
# seeds = [0, 6, 1995, 1996, 1997]
seeds = [6, 1995, 1996]
#  three of them is a group, calculate multiple_runs for all groups

# for gamma in [2, 3, 4]:
#     filter_configs["gamma"] = gamma



caption_prefix = "all biased baseline "
# caption_prefix = "second"
# for beta in [2, 5, 8]:
#     filter_configs["beta"] = beta
for group in itertools.combinations(seeds, 3):
    for target_group in ['age', 'gender']:
        # to_keep = find_already_fair_runs(target_group)
        to_keep = None
        filter_configs["target_sensitive_group"] = target_group
        multiple_runs(group, runs, to_report_metrics, to_separate_metrics, filter_configs, to_report_configs, caption_prefix=caption_prefix, previous_results_to_keep=to_keep)


k=1








# runs_df.to_csv("project.csv")