import pandas as pd
import numpy as np
import wandb
api = wandb.Api()


# sorted runs by modalities
# runs = sorted(runs, key=lambda x: x.config["modalities"])
def latex_table(fn):
    def wrapper(*args, **kwargs):
        # print content from table_template.tex
        with open("table_template.tex", "r") as f:
            # print if not \bottomrule
            # iterate over lines
            line = ''
            pre_table = ''
            while "bottomrule" not in line:
                pre_table += line
                line = f.readline()
            print(pre_table)
            fn(*args, **kwargs)
            print(line)
            # read() the rest of the file
            print(f.read())
            return None
    return wrapper


@latex_table
def print_sorted_runs(sorted_modalities_names, sorted_items):
    for name, tmp in zip(sorted_modalities_names, sorted_items):
        item = ''
        # covert name to string
        modalities = ','.join(name)
        item += modalities + ' & '
        item += tmp
        # replace audio with Audio
        item = item.replace("audio", "Audio")
        # replace facebody with FaceBody
        item = item.replace("facebody", "FaceBody")
        # replace senti,speech,time with Textual
        item = item.replace("senti,speech,time", "Textual")
        # replace senti,speech,text,time with  BERT, Textual
        item = item.replace("senti,speech,text,time", "BERT, Textual")
        # # replace text with BERT, except \textcolor
        item = item.replace("text", "BERT")
        item = item.replace("\BERT", "\\text")

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
def multiple_runs(seeds, runs, to_report_metrics, to_separate_metrics, filter_configs, to_report_configs):
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
    # compute the average of all items
    for i in range(len(all_items[0])):
        avged_items = ''
        for j in range(len(all_items[0][i])):
            avg = np.mean(all_items[:, i, j])
            if j == 0:
                avged_items += str(round(avg, 4)) + ' & '
            else:
                if avg < 80:
                    #  \textcolor{red}{0.7907}
                    avged_items += '\\textcolor{red}{' + str(round(avg, 2)) + '} & '
                else:
                    avged_items += str(round(avg, 2)) + ' & '
        avged_items = avged_items[:-2] + '\\\\'
        latex_table.append(avged_items)

    print_sorted_runs(sorted_modalities_names, latex_table)


to_report_configs = ["modalities"]
to_report_metrics = [ "val_mse", "gender_val_DIR_O", "gender_val_DIR_C", "gender_val_DIR_E", "gender_val_DIR_A", "gender_val_DIR_N", "age_val_DIR_O", "age_val_DIR_C", "age_val_DIR_E", "age_val_DIR_A", "age_val_DIR_N"]
to_separate_metrics = ["val_mse"]

# for baseline
# filter_configs = {"depth":3, "lr": 0.004, "num_latents": 128, "epochs":50, "results_dir": "results/trainval_bul_baseline_right_uniqueMean"}
# runs = api.runs("jianjiang/perceiver_affection_baseline_trainval_bul_right")
runs = api.runs("jianjiang/perceiver_affection_v100_age26_trainval")
# 1996 and 6
filter_configs = {"depth":5, "lr": 0.004, "num_latents": 128, "epochs":60, "seed":1995}
# runs = api.runs("jianjiang/perceiver_affection_baseline_trainval_a5000")

# for second
# runs = api.runs("jianjiang/perceiver_affection_spd_trainval_3090")
# filter_configs = { "lr": 0.004, "gamma": 5, "epochs": 5, "num_latents": 128, "seed": 1995, "target_sensitive_group": "gender"}
# runs = api.runs("jianjiang/perceiver_affection_spd_trainval_a5000")
# for test
# to_report_metrics = [ "test_loss", "gender_test_DIR_O", "gender_test_DIR_C", "gender_test_DIR_E", "gender_test_DIR_A", "gender_test_DIR_N", "age_test_DIR_O", "age_test_DIR_C", "age_test_DIR_E", "age_test_DIR_A", "age_test_DIR_N"]
# to_separate_metrics = ["test_loss"]
# runs = api.runs("jianjiang/perceiver_affection_test")
# filter_configs = {}

# for third
# to_report_metrics = [ "val_mse", "gender_val_DIR_O", "gender_val_DIR_C", "gender_val_DIR_E", "gender_val_DIR_A", "gender_val_DIR_N", "age_val_DIR_O", "age_val_DIR_C", "age_val_DIR_E", "age_val_DIR_A", "age_val_DIR_N"]
# to_separate_metrics = ["val_mse"]
# runs = api.runs("jianjiang/perceiver_affection_spd_third_trainval_bul")
# filter_configs = {"beta": 0.1, "results_dir": "results/sweep_spd_trainval_devicebul_third_ablation", "gamma":10}
# runs = api.runs("jianjiang/perceiver_affection_third_trainval_a5000")
# filter_configs = { "lr": 0.004, "beta": 1, "epochs":5}
record_runs(to_report_metrics, to_separate_metrics, filter_configs, to_report_configs, runs)


k=1








# runs_df.to_csv("project.csv")