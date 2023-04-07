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
        with open("latex/table_template_personality.tex", "r") as f:
        # with open("latex/table_template_v3.tex", "r") as f:
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
    # line start with % are test set
    # otherwise, it is train set
    sensitive_group = ['gender', 'ethnicity']
    sensitive_group.remove(target_group)
    sensitive_group = sensitive_group[0]
    # for biased dataset use another template
    with open(f"latex/{sensitive_group}_second_stage_personality_biased.tex", "r") as f:
        search_range = [i for i in range(3, 18, 3)] if target_group == 'gender' else [i for i in range(4, 18, 3)]
        to_keep = {}
        line = f.readline()
        while line:
            if line == '\n':
                line = f.readline()
                continue

            items = line.split('&')
            name = items[0]
            k=1
            # join items to string based on search_range
            items = [items[i] for i in search_range]
            to_search = ''.join(items)

            if '%' in line:
                to_keep[name] = line

            elif 'textcolor' not in to_search:
                to_keep[name] = line
            line = f.readline()

        return to_keep


def sort_runs_dict(runs_dict, key):
    # sort runs_dict based on key
    sorted_runs_dict = {}
    base_key = runs_dict[key]
    # sort base_key and get the index based on  the length of item and alphabet
    lengths = [len(x) for x in base_key]
    sorted_base_key_indices = sorted(range(len(lengths)), key=lambda k: lengths[k])

    # print(sorted_base_key_indices)
    for k, v in runs_dict.items():
        # print(k, v)
        # if v is not empty list
        if v:
            # print(k, v)
            sorted_runs_dict[k] = [v[i] for i in sorted_base_key_indices]
    return sorted_runs_dict


@latex_table
def print_sorted_runs(sorted_runs_dict, to_report_list, to_separate_metrics, previous_results_to_keep=None, caption_name=None, highlight_flag=True):
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

    # get the i th item for each key
    items = []
    for i in range(len(sorted_runs_dict['modalities'])):
        item = ''
        for key in to_report_list:
            v = sorted_runs_dict[key]

            if key == 'modalities':
                item += v[i] + ' & '
                continue
            if key in to_separate_metrics:
                item += str(round(v[i] * 100, 2)) + ' & '
            else:
                value = round(v[i] * 100, 2)
                if value < 80 and highlight_flag:
                    item += '\\textcolor{red}{' + str(value) + '} & '
                else:
                    item += str(value) + ' & '
                # remove the last &
        item = item[:-2]
        # plus \\
        item += '\\\\'
        items.append(item)

        item = modifiy_item(item)
        modified_name = item.split('&')[0]
        if previous_results_to_keep is not None and modified_name in previous_results_to_keep.keys():
            to_get_name = "%" + modified_name
            item = previous_results_to_keep[to_get_name]
            item = modifiy_item(item)
        print(item)


def record_runs(runs, filter_configs, to_report_configs, to_avg_configs, to_report_metrics, to_separate_metrics, highlight_flag=True, print_flag=False):
    items = []
    modalities_names = []
    recorded_personality = []
    # combine two list of to_report_metrics and to_report_configs

    combined_list = to_report_metrics + to_report_configs + to_avg_configs
    runs_dict = {key: [] for key in combined_list}
    for run in runs:
        flag_filtered = False

        for key in filter_configs:
            if key in run.config:
                if isinstance(filter_configs[key], list):
                    if run.config[key] not in filter_configs[key]:
                        # print("not in")
                        flag_filtered = True
                        break
                elif run.config[key] != filter_configs[key]:
                    flag_filtered = True
                    break

        if not flag_filtered:
            for key in to_report_configs + to_avg_configs:
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
                            # sort the name of a certain combination of modalities based alphabetical order
                            modalities = sorted(modalities)
                            # combine them with ','
                            modalities = ','.join(modalities)
                        # item += modalities + ' & '
                        runs_dict[key].append(modalities)
                    else:
                        runs_dict[key].append(run.config[key])
                        # item += str(run.config[key]) + ' & '

            for key in to_report_metrics:
                if key in run.summary._json_dict:
                    runs_dict[key].append(run.summary._json_dict[key])

    # sort_runs_dict(runs_dict, "modalities")
    if print_flag:
        sorted_runs_dict = sort_runs_dict(runs_dict, "modalities")
        print_sorted_runs(sorted_runs_dict, combined_list, to_separate_metrics, previous_results_to_keep=None, caption_name=None)
    return runs_dict


# for multiple random seeds
def multiple_runs(runs, filter_configs, to_report_configs, to_avg_configs, to_report_metrics,  to_separate_metrics, caption_prefix, previous_results_to_keep=None):

    def averethnicity_runs_seeds(runs_dict):
        # averethnicity each dict based on the seeds
        unique_modalities = list(set(runs_dict["modalities"]))
        avged_runs_dict = {}

        for mod in unique_modalities:
            # get the index of the seed
            index = [i for i, x in enumerate(runs_dict["modalities"]) if x == mod]
            # get the dict based on the index
            mod_runs_dict = {key: [runs_dict[key][i] for i in index] for key in runs_dict.keys()}
            # averethnicity the dict
            avged_runs_dict[mod] = {
                key: sum(mod_runs_dict[key]) / len(mod_runs_dict[key]) if not isinstance(mod_runs_dict[key][0],
                                                                                         str) else mod_runs_dict[key][0]
                for key in mod_runs_dict.keys()}

        modified_avged_runs_dict = {key: [] for key in runs_dict.keys()}

        for key in avged_runs_dict:
            modified_avged_runs_dict["modalities"].append(key)
            for k in avged_runs_dict[key]:
                if k != "modalities":
                    modified_avged_runs_dict[k].append(avged_runs_dict[key][k])

        return modified_avged_runs_dict

    if 'target_personality' in to_avg_configs:
        all_items = []
        for personality in range(0, 5):
            filter_configs['target_personality'] = personality
            runs_dict = record_runs(runs, filter_configs, to_report_configs, to_avg_configs, to_report_metrics,  to_separate_metrics)
            modified_avged_runs_dict = averethnicity_runs_seeds(runs_dict)

            all_items.append(modified_avged_runs_dict)

        final_runs_dict = {key: [] for key in runs_dict.keys()}
        # addition_to_report_metrics = []


        for i, p in enumerate(['O', 'C', 'E', 'A', 'N']):
            # addition_to_report_metrics.append(f'mse_{p}')
            final_runs_dict[f'mse_{p}'] = all_items[i]['test_mse']
        for key in to_report_metrics + to_report_configs:
            if 'mse' in key or 'MSE' in key:
                # averethnicity mathcing the modalities and the target personality;
                # i is personality, j is modality

                final_runs_dict[key] = [sum([all_items[i][key][j] for i in range(0, 5)]) / 5 for j in range(len(all_items[0][key]))]
            elif key == 'modalities':
                final_runs_dict[key] = all_items[0][key]
            else:
                for i, p in enumerate(['O', 'C', 'E', 'A', 'N']):
                    if p in key:
                        final_runs_dict[key] = all_items[i][key]

        to_report_metrics = ["val_mse", "mse_O", "gender_val_DIR_O", "ethnicity_val_DIR_O", "mse_C", "gender_val_DIR_C",
                             "ethnicity_val_DIR_C", "mse_E",
                             "gender_val_DIR_E", "ethnicity_val_DIR_E", "mse_A", "gender_val_DIR_A", "ethnicity_val_DIR_A", "mse_N",
                             "gender_val_DIR_N", "ethnicity_val_DIR_N"]
        to_separate_metrics = ["val_mse", "mse_O", "mse_C", "mse_E", "mse_A", "mse_N"]
        key_name_test = 'test'
        to_report_metrics = [x.replace('val', key_name_test) for x in to_report_metrics]
        to_separate_metrics = [x.replace('val', key_name_test) for x in to_separate_metrics]
        modified_avged_runs_dict = final_runs_dict
        k=1
    else:
        runs_dict = record_runs(runs, filter_configs, to_report_configs, to_avg_configs, to_report_metrics,  to_separate_metrics,  highlight_flag=False, print_flag=False)
        if "train_mse" in runs_dict:
            runs_dict['train_mse'] = [100 for i in range(len(runs_dict["modalities"]))]

        modified_avged_runs_dict = averethnicity_runs_seeds(runs_dict)

    # seeds
    modified_avged_runs_dict = sort_runs_dict(modified_avged_runs_dict, "modalities")
    # modified_avged_runs_dict = sort_runs_dict(runs_dict, "modalities")
    tmp=1


    # personality are matched based on modality and target_personality
    # # caption_name is the combination of seeds and target_sensitive_group
    # target_sensitive_group = filter_configs["target_sensitive_group"]
    caption_name = caption_prefix + '\_'.join([str(x) for x in seeds])
    # caption_name = '\caption{'+ caption_name +  target_sensitive_group
    # # plus gamma
    # # caption_name += '  gamma\_' + str(filter_configs["gamma"])
    # # # plus beta
    # # caption_name += '  beta\_' + str(filter_configs["beta"])
    # # end }
    # caption_name += '}'
    print_sorted_runs(modified_avged_runs_dict, to_report_configs + to_report_metrics, to_separate_metrics, previous_results_to_keep=previous_results_to_keep,  caption_name=caption_name)#, previous_results_to_keep=previous_results_to_keep)

print('dada')

to_report_configs = ["modalities"]
to_avg_configs = ["seed"]

to_separate_metrics = ["val_mse", "val_mse_O", "val_mse_C", "val_mse_E", "val_mse_A", "val_mse_N", "val_MSE_O", "val_MSE_C", "val_MSE_E", "val_MSE_A", "val_MSE_N"]
# replace val with test


# for baseline



to_report_metrics_second = ["val_mse", "val_mse_O", "gender_val_DIR_O","ethnicity_val_DIR_O", "val_mse_C", "gender_val_DIR_C", "ethnicity_val_DIR_C", "val_mse_E", "gender_val_DIR_E", "ethnicity_val_DIR_E", "val_mse_A", "gender_val_DIR_A", "ethnicity_val_DIR_A", "val_mse_N", "gender_val_DIR_N",  "ethnicity_val_DIR_N"]
to_report_metrics_per = ["val_mse",  "gender_val_DIR_O","ethnicity_val_DIR_O",  "gender_val_DIR_C", "ethnicity_val_DIR_C",  "gender_val_DIR_E", "ethnicity_val_DIR_E",  "gender_val_DIR_A", "ethnicity_val_DIR_A",  "gender_val_DIR_N",  "ethnicity_val_DIR_N"]

# caption_prefix = " baseline fiv2"
# runs = api.runs("jianjiang/perceiver_affection_baseline_trainval_3090_fiv2")



to_report_metrics = to_report_metrics_second


filter_configs = {"gamma": 10}


# runs = api.runs("jianjiang/perceiver_affection_spd_trainval_3090_fiv2")
# caption_prefix = f"second fiv"
runs = api.runs("jianjiang/perceiver_affection_third_trainval_3090_fiv2")
caption_prefix = f"third fiv gamma{filter_configs['gamma']}"

seeds = [6, 1995, 1996]
# caption_prefix = "second"
# for beta in [2, 5, 8]:
#     filter_configs["beta"] = beta
# for group in itertools.combinations(seeds, 3):

key_name_test = 'test'
to_report_metrics = [x.replace('val', key_name_test) for x in to_report_metrics]
to_separate_metrics = [x.replace('val', key_name_test) for x in to_separate_metrics]
for target_group in ['gender', 'ethnicity']:
    # to_keep = find_already_fair_runs(target_group)
    to_keep = None
    filter_configs["target_sensitive_group"] = target_group
    multiple_runs(runs, filter_configs, to_report_configs, to_avg_configs, to_report_metrics,  to_separate_metrics, caption_prefix=caption_prefix+f" {target_group}", previous_results_to_keep=to_keep)


k=1








# runs_df.to_csv("project.csv")